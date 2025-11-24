import os
import random
from datetime import datetime
from pathlib import Path

import dotenv
import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from timm.scheduler import CosineLRScheduler

import wandb
from loss import MaskedLoss
from model.model_nar_ssl import Lip2SpeechLightWeight, Lip2SpeechSSL
from utils import (
    check_mel_nar,
    count_params,
    fix_random_seed,
    get_path_train_raw,
    make_train_val_loader_with_external_data_raw,
    save_loss,
    set_config,
)


dotenv.load_dotenv()
wandb.login(key=os.environ['WANDB_API_KEY'])
current_time = datetime.now().strftime("%Y:%m:%d_%H-%M-%S")


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss_list,
    train_mae_loss_list,
    train_mse_loss_list,
    val_loss_list,
    val_mae_loss_list,
    val_mse_loss_list,
    epoch,
    ckpt_path,
):
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "random": random.getstate(),
            "np_random": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            "cuda_random": torch.cuda.get_rng_state(),
            "train_loss_list": train_loss_list,
            "train_mae_loss_list": train_mae_loss_list,
            "train_mse_loss_list": train_mse_loss_list,
            "val_loss_list": val_loss_list,
            "val_mae_loss_list": val_mae_loss_list,
            "val_mse_loss_list": val_mse_loss_list,
            "epoch": epoch,
        },
        ckpt_path,
    )


def make_generator(
    cfg,
    device,
):
    if cfg.model.model_name == "lightweight":
        model = Lip2SpeechLightWeight(cfg)
    else:
        Generator = Lip2SpeechSSL(cfg)
    count_params(Generator, "model")
    return Generator.to(device)

# --- 変更: より明確なメルスペクトログラム用識別器（Patch/patch-style） ---
class MelDiscriminator(nn.Module):
    """
    Mel-spectrogram 用のPatchGAN風判別器（Patch判定）。
    入力想定: Generator出力 (B, T, C) または (B, C, T)。
    内部で (B,1,C,T) に整形して2D畳み込みを行い、patch map (B,1,H,W) を返す。
    """
    def __init__(self, in_mel_channels=80, base_ch=64):
        super(MelDiscriminator, self).__init__()
        self.in_mel_channels = in_mel_channels
        self.net = nn.Sequential(
            nn.Conv2d(1, base_ch, kernel_size=(3,3), stride=(1,2), padding=(1,1)),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch, base_ch*2, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=(3,3), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_ch*4, 1, kernel_size=(3,3), stride=(1,1), padding=(1,1)),
            # 出力は patch map（後続で平均等してスカラーにできます）
        )

    def forward(self, x):
        # x: (B, T, C) または (B, C, T) または (B,1,C,T)
        if x.dim() == 3:
            # 判定: 末尾の次元がメルチャネル数なら (B,T,C) とみなす
            if x.shape[-1] == self.in_mel_channels:
                # (B, T, C) -> (B, C, T)
                x = x.permute(0,2,1)
                x = x.unsqueeze(1)  # -> (B,1,C,T)
            else:
                # (B, C, T) とみなす
                x = x.unsqueeze(1)  # -> (B,1,C,T)
        elif x.dim() == 4:
            # 既に (B,1,C,T) の場合はそのまま、別の形式なら可能な変換を試みるが基本は想定外
            pass
        return self.net(x)  # (B,1,H,W)


def make_discriminator(cfg, device):
    D = MelDiscriminator(in_mel_channels=cfg.model.n_mel_channels, base_ch=64)
    count_params(D, "discriminator")
    return D.to(device)
# --- 変更ここまで ---

def train_one_epochG(
    model,
    train_loader,
    optimizer,
    scaler,
    loss_f,
    device,
    cfg,
    ckpt_time,
    discriminator = None,
):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("Generator-train")
    model.train()

    gan_weight = getattr(cfg.train, "gan_loss_weight", 0.0)
    mse = nn.MSELoss()

    for batch in train_loader:
        print(f"iter {iter_cnt}/{all_iter}")
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            emo_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        ) = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        emo_emb = emo_emb.to(device) if cfg.train.use_emo_label else None
        speaker_idx = speaker_idx.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = model(
                lip=lip,
                audio=None,
                lip_len=lip_len,
                spk_emb=spk_emb,
                emo_emb=emo_emb,
            )
            mae_loss = loss_f.mae_loss(
                output, feature, feature_len, max_len=output.shape[-1]
            )
            mse_loss = loss_f.mse_loss(
                output, feature, feature_len, max_len=output.shape[-1]
            )

            adv_loss = 0.0
            if gan_weight > 0.0 and discriminator is not None:
                discriminator.eval()
                # output は (B,T,C) → 判別器へそのまま渡す
                pred = discriminator(output)
                real_target = torch.full_like(pred, 0.9, device=device)
                adv_loss = mse(pred, real_target)


            loss = mae_loss + gan_weight * adv_loss

            epoch_mae_loss += mae_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_loss += loss.item()
            wandb.log({"train_mae_loss": mae_loss})
            wandb.log({"train_mse_loss": mse_loss})
            wandb.log({"train_loss": loss})
            if gan_weight > 0.0:
                wandb.log({"train_adv_loss": adv_loss})

            loss = loss / cfg.train.iters_to_accumulate

        scaler.scale(loss).backward()
        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (
            all_iter - 1
        ) == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(
                    feature[0], output[0], cfg, "mel_train", current_time, ckpt_time
                )
                break

        if iter_cnt % (all_iter - 1) == 0:
            check_mel_nar(
                feature[0], output[0], cfg, "mel_train", current_time, ckpt_time
            )

    epoch_loss /= iter_cnt
    epoch_mae_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    return epoch_loss, epoch_mae_loss, epoch_mse_loss


def val_one_epochG(
    model,
    val_loader,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    epoch_loss = 0
    epoch_mae_loss = 0
    epoch_mse_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("Generator-validation")
    model.eval()

    for batch in val_loader:
        print(f"iter {iter_cnt}/{all_iter}")
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            emo_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        ) = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        emo_emb = emo_emb.to(device) if cfg.train.use_emo_label else None

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            with torch.no_grad():
                output = model(
                    lip=lip,
                    audio=None,
                    lip_len=lip_len,
                    spk_emb=spk_emb,
                    emo_emb=emo_emb,
                )

            mae_loss = loss_f.mae_loss(
                output, feature, feature_len, max_len=output.shape[-1]
            )
            mse_loss = loss_f.mse_loss(
                output, feature, feature_len, max_len=output.shape[-1]
            )
            loss = mae_loss
            epoch_mae_loss += mae_loss.item()
            epoch_mse_loss += mse_loss.item()
            epoch_loss += loss.item()
            wandb.log({"val_mae_loss": mae_loss})
            wandb.log({"val_mse_loss": mse_loss})
            wandb.log({"val_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(
                    feature[0],
                    output[0],
                    cfg,
                    "mel_validation",
                    current_time,
                    ckpt_time,
                )
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                check_mel_nar(
                    feature[0],
                    output[0],
                    cfg,
                    "mel_validation",
                    current_time,
                    ckpt_time,
                )
        else:
            check_mel_nar(
                feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time
            )

    epoch_loss /= iter_cnt
    epoch_mae_loss /= iter_cnt
    epoch_mse_loss /= iter_cnt
    result = (epoch_loss, epoch_mae_loss, epoch_mse_loss)
    return result

# --------------- 追加: Discriminator 学習関数 (LSGAN / MSE) ---------------
def train_one_epochD(
    generator,
    discriminator,
    train_loader,
    optimizerD,
    device,
    cfg,
):
    """
    Discriminator を1エポック学習（Generatorは固定）。LSGAN (MSE) を使用。
    入力の real feature は train_loader から得られる feature (B, T, C)。
    Generator 出力は (B, T, C) を想定。
    """
    discriminator.train()
    generator.eval()
    mse = nn.MSELoss()
    epoch_loss = 0.0
    iter_cnt = 0
    all_iter = len(train_loader)
    real_label_val = 0.9  # label smoothing
    fake_label_val = 0.0
    print("Discriminator-train" )

    for batch in train_loader:
        print(f"iter {iter_cnt}/{all_iter}")
        (
            wav,
            lip,
            feature,
            feature_avhubert,
            spk_emb,
            emo_emb,
            feature_len,
            lip_len,
            speaker,
            speaker_idx,
            filename,
            lang_id,
            is_video,
        ) = batch

        # real mel: (B, T, C) -> (B, T, C)
        real_mel = feature.to(device)

        # real forward
        real_pred = discriminator(real_mel)  # returns (B,1,H,W)
        real_target = torch.full_like(real_pred, real_label_val, device=device)
        loss_real = mse(real_pred, real_target)

        # fake mel: generator で生成
        lip = lip.to(device)
        lip_len = lip_len.to(device)
        spk_emb = spk_emb.to(device)
        emo_emb = emo_emb.to(device) if cfg.train.use_emo_label else None

        with torch.no_grad():
            fake_mel = generator(lip=lip, audio=None, lip_len=lip_len, spk_emb=spk_emb, emo_emb=emo_emb)
        # fake_mel: (B,T,C) expected; pass to D directly
        fake_pred = discriminator(fake_mel.detach())
        fake_target = torch.full_like(fake_pred, fake_label_val, device=device)
        loss_fake = mse(fake_pred, fake_target)

        lossD = 0.5 * (loss_real + loss_fake)

        optimizerD.zero_grad()
        lossD.backward()
        optimizerD.step()

        epoch_loss += lossD.item()
        iter_cnt += 1

    return epoch_loss / max(1, iter_cnt)
# --------------- ここまで ---------------


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)

    wandb_cfg = OmegaConf.to_container(
        cfg,
        #resolve=True,
        #throw_on_missing=True, #?throw_on_missingがないぜって言われてる。バージョンも互換性も確認済みーーーコメントアウトによりスルー
    )

    #print(OmegaConf.to_yaml(cfg.model))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")

    video_dir, audio_dir, ckpt_path, save_path, ckpt_time = get_path_train_raw(
        cfg, current_time
    )

    """
    print(f"video_dir = {video_dir}")
    print(f"audio_dir = {audio_dir}")
    breakpoint()
    """

    train_loader, val_loader, train_dataset, val_dataset = (
        make_train_val_loader_with_external_data_raw(cfg, video_dir, audio_dir)
    )

    loss_f = MaskedLoss()
    train_loss_list = []
    train_mae_loss_list = []
    train_mse_loss_list = []
    val_loss_list = []
    val_mae_loss_list = []
    val_mse_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(
        **cfg.wandb_conf.setup,
        config=wandb_cfg,
        settings=wandb.Settings(start_method="fork"),
    ) as run:
        Generator = make_generator(cfg, device)
        Discriminator = make_discriminator(cfg, device)
        print(f"{cfg.model.avhubert_config.model_size=}")
        print(f"{cfg.model.avhubert_config.load_pretrained_weight=}")
        #breakpoint()
        #print("-----------------------以下モデル構造--------------------------")
        #print(model)
        #print("----------------------------以上-------------------------------")

        if cfg.train.which_optim == "adam":
            optimizerG = torch.optim.Adam(
                params=Generator.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )

            optimizerD = torch.optim.Adam(
                params=Discriminator.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )

        elif cfg.train.which_optim == "adamw":
            optimizerG = torch.optim.AdamW(
                params=Generator.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )
            optimizerD = torch.optim.AdamW(
                params=Discriminator.parameters(),
                lr=cfg.train.lr,
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,
            )

        if cfg.train.which_scheduler == "exp":
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizerG,
                gamma=cfg.train.lr_decay_exp,
            )
        elif cfg.train.which_scheduler == "warmup":
            scheduler = CosineLRScheduler(
                optimizer=optimizerG,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )

        scaler = torch.cuda.amp.GradScaler()

        last_epoch = 0

        wandb.watch(Generator, **cfg.wandb_conf.watch)

        #!train_one_epoch_D→,train_one_epoch_G→val_one_epoch_Gの流れをつくりたい
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            # 1) Discriminatorを1エポック学習（Generator固定）
            if getattr(cfg.train, "use_gan", False) and getattr(cfg.train, "gan_loss_weight", 0.0) > 0.0:
                lossD_epoch = train_one_epochD(
                    generator=Generator,
                    discriminator=Discriminator,
                    train_loader=train_loader,
                    optimizerD=optimizerD,
                    device=device,
                    cfg=cfg,
                )
                wandb.log({"train_discriminator_loss": lossD_epoch, "epoch": current_epoch})

            # 2) Generatorを1エポック学習（Discriminator固定）
            epoch_loss, epoch_mae_loss, epoch_mse_loss = train_one_epochG(
                model=Generator,
                train_loader=train_loader,
                optimizer=optimizerG,
                scaler=scaler,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_mae_loss_list.append(epoch_mae_loss)
            train_mse_loss_list.append(epoch_mse_loss)

            epoch_loss, epoch_mae_loss, epoch_mse_loss = val_one_epochG(
                model=Generator,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_mae_loss_list.append(epoch_mae_loss)
            val_mse_loss_list.append(epoch_mse_loss)

            if cfg.train.which_scheduler == "exp":
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == "warmup":
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]["lr"]})
                scheduler.step(epoch)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=Generator,
                    optimizer=optimizerG,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    train_mae_loss_list=train_mae_loss_list,
                    train_mse_loss_list=train_mse_loss_list,
                    val_loss_list=val_loss_list,
                    val_mae_loss_list=val_mae_loss_list,
                    val_mse_loss_list=val_mse_loss_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt"),
                )

            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_mae_loss_list, val_mae_loss_list, save_path, "mae_loss")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")

    wandb.finish()


if __name__ == "__main__":
    main()




"""!以下、学習コードの参考
G_losses = []
D_losses = []
D_x_out = []
D_G_z1_out = []

# 学習のループ
for epoch in range(n_epoch):
    for itr, data in enumerate(dataloader):
        real_image = data[0].to(device)     # 本物画像
        sample_size = real_image.size(0)    # 画像枚数
        
        # 標準正規分布からノイズを生成
        noise = torch.randn(sample_size, nz, 1, 1, device=device)
        # 本物画像に対する識別信号の目標値「1」
        real_target = torch.full((sample_size,), 1., device=device)
        # 生成画像に対する識別信号の目標値「0」
        fake_target = torch.full((sample_size,), 0., device=device) 
        
        ############################
        # 識別器Dの更新
        ###########################
        netD.zero_grad()    # 勾配の初期化

        output = netD(real_image)   # 識別器Dで本物画像に対する識別信号を出力
        errD_real = criterion(output, real_target)  # 本物画像に対する識別信号の損失値
        D_x = output.mean().item()  # 本物画像の識別信号の平均

        fake_image = netG(noise)    # 生成器Gでノイズから生成画像を生成
        
        output = netD(fake_image.detach())  # 識別器Dで生成画像に対する識別信号を出力
        errD_fake = criterion(output, fake_target)  # 生成画像に対する識別信号の損失値
        D_G_z1 = output.mean().item()  # 生成画像の識別信号の平均

        errD = errD_real + errD_fake    # 識別器Dの全体の損失
        errD.backward()    # 誤差逆伝播
        optimizerD.step()   # Dのパラメーターを更新

        ############################
        # 生成器Gの更新
        ###########################
        netG.zero_grad()    # 勾配の初期化
        
        output = netD(fake_image)   # 更新した識別器Dで改めて生成画像に対する識別信号を出力
        errG = criterion(output, real_target)   # 生成器Gの損失値。Dに生成画像を本物画像と誤認させたいため目標値は「1」
        errG.backward()     # 誤差逆伝播
        D_G_z2 = output.mean().item()  # 更新した識別器Dによる生成画像の識別信号の平均

        optimizerG.step()   # Gのパラメータを更新

        if itr % display_interval == 0: 
            print('[{}/{}][{}/{}] Loss_D: {:.3f} Loss_G: {:.3f} D(x): {:.3f} D(G(z)): {:.3f}/{:.3f}'
                  .format(epoch + 1, n_epoch,
                          itr + 1, len(dataloader),
                          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        if epoch == 0 and itr == 0:     # 初回に本物画像を保存する
            vutils.save_image(real_image, '{}/real_samples.png'.format(outf),
                              normalize=True, nrow=10)

        # ログ出力用データの保存
        D_losses.append(errD.item())
        G_losses.append(errG.item())
        D_x_out.append(D_x)
        D_G_z1_out.append(D_G_z1)

    ############################
    # 確認用画像の生成
    ############################
    fake_image = netG(fixed_noise)  # 1エポック終了ごとに確認用の生成画像を生成する
    vutils.save_image(fake_image.detach(), '{}/fake_samples_epoch_{:03d}.png'.format(outf, epoch + 1),
                      normalize=True, nrow=10)

    ############################
    # モデルの保存
    ############################
    if (epoch + 1) % 10 == 0:   # 10エポックごとにモデルを保存する
        torch.save(netG.state_dict(), '{}/netG_epoch_{}.pth'.format(outf, epoch + 1))
        torch.save(netD.state_dict(), '{}/netD_epoch_{}.pth'.format(outf, epoch + 1))
"""
