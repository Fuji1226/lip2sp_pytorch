"""
非自己回帰decoderの学習用
"""
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from librosa.display import specshow

import torch
from torch.nn.utils import clip_grad_norm_

from model.model_nar import Lip2SP_NAR
from loss import MaskedLoss
from train_default import make_train_val_loader, check_feat_add

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(model, optimizer, scheduler, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def save_loss(train_loss_list, val_loss_list, save_path, filename):
    loss_save_path = save_path / f"{filename}.png"
    plt.figure()
    plt.plot(np.arange(len(train_loss_list)), train_loss_list)
    plt.plot(np.arange(len(train_loss_list)), val_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["train loss", "validation loss"])
    plt.grid()
    plt.savefig(str(loss_save_path))
    plt.close("all")
    # wandb.log({f"{filename}": plt})
    # wandb.log({f"Image {filename}": wandb.Image(os.path.join(save_path, f"{filename}.png"))})
    wandb.log({f"loss {filename}": wandb.plot.line_series(
        xs=np.arange(len(train_loss_list)), 
        ys=[train_loss_list, val_loss_list],
        keys=["train loss", "validation loss"],
        title=f"{filename}",
        xname="epoch",
    )})


def make_model(cfg, device):
    model = Lip2SP_NAR(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_inner_channels=cfg.model.tc_inner_channels,
        dec_kernel_size=cfg.model.tc_kernel_size,
        feat_add_channels=cfg.model.tc_feat_add_channels,
        feat_add_layers=cfg.model.tc_feat_add_layers,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        apply_first_bn=cfg.train.apply_first_bn,
        multi_task=cfg.train.multi_task,
        add_feat_add=cfg.train.add_feat_add,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        reduction_factor=cfg.model.reduction_factor,
        use_gc=cfg.train.use_gc,
    )

    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def check_mel(target, output, cfg, filename):
    target = target.to('cpu').detach().numpy().copy()
    output = output.to('cpu').detach().numpy().copy()

    plt.close("all")
    plt.figure()
    ax = plt.subplot(2, 1, 1)
    specshow(
        data=target, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length,
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("target")
    
    ax = plt.subplot(2, 1, 2, sharex=ax, sharey=ax)
    specshow(
        data=output, 
        x_axis="time", 
        y_axis="mel", 
        sr=cfg.model.sampling_rate, 
        hop_length=cfg.model.hop_length, 
        fmin=cfg.model.f_min,
        fmax=cfg.model.f_max,
        cmap="viridis",
    )
    plt.colorbar(format="%+2.f dB")
    plt.xlabel("Time[s]")
    plt.ylabel("Frequency[Hz]")
    plt.title("output")

    plt.tight_layout()
    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(str(save_path / f"{filename}.png"))
    wandb.log({f"{filename}": wandb.Image(str(save_path / f"{filename}.png"))})


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        output, feat_add_out = model(lip=lip, data_len=data_len)
        B, C, T = output.shape
        
        if cfg.train.multi_task:
            loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"train_iter_loss_feat_add": loss_feat_add})
            loss_feat_add.backward(retain_graph=True)

        loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        epoch_loss += loss.item()
        wandb.log({"train_iter_loss": loss})
        loss.backward()

        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_train")
                    if cfg.train.multi_task:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train")
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_train")
                if cfg.train.multi_task:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_train")

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss, epoch_loss_feat_add


def calc_val_loss(model, val_loader, loss_f, device, cfg):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("\ncalc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        
        with torch.no_grad():
            output, feat_add_out = model(lip=lip, data_len=data_len)

        B, C, T = output.shape
        
        if cfg.train.multi_task:
            loss_feat_add = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=T)
            epoch_loss_feat_add += loss_feat_add.item()
            wandb.log({"val_iter_loss_feat_add": loss_feat_add})

        loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
        epoch_loss += loss.item()
        wandb.log({"val_iter_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel(feature[0], output[0], cfg, "mel_validation")
                    if cfg.train.multi_task:
                        check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_validation")
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel(feature[0], output[0], cfg, "mel_validation")
                if cfg.train.multi_task:
                    check_feat_add(feature[0], feat_add_out[0], cfg, "feat_add_validation")
            
    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss, epoch_loss_feat_add


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

    # 口唇動画か顔かの選択
    lip_or_face = cfg.train.face_or_lip
    if lip_or_face == "face":
        data_root = cfg.train.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    elif lip_or_face == "lip":
        data_root = cfg.train.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path
    elif lip_or_face == "lip_128128":
        data_root = cfg.train.lip_pre_loaded_path_128128
        mean_std_path = cfg.train.lip_mean_std_path_128128
    elif lip_or_face == "lip_9696":
        data_root = cfg.train.lip_pre_loaded_path_9696
        mean_std_path = cfg.train.lip_mean_std_path_9696
    elif lip_or_face == "lip_9696_time_only":
        data_root = cfg.train.lip_pre_loaded_path_9696_time_only
        mean_std_path = cfg.train.lip_mean_std_path_9696_time_only
    
    data_root = Path(data_root).expanduser()
    mean_std_path = Path(mean_std_path).expanduser()

    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")

    # check point
    ckpt_path = Path(cfg.train.ckpt_path).expanduser()
    ckpt_path = ckpt_path / lip_or_face / current_time
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = Path(cfg.train.save_path).expanduser()
    save_path = save_path / lip_or_face / current_time
    os.makedirs(save_path, exist_ok=True)

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    loss_f = MaskedLoss()
    train_loss_list = []
    train_feat_add_loss_list = []
    val_loss_list = []
    val_feat_add_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        # scheduler
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer,
            milestones=cfg.train.multi_lr_decay_step,
            gamma=cfg.train.lr_decay_rate,
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = cfg.train.start_ckpt_path
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = epoch + last_epoch
            print(f"##### {current_epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            train_epoch_loss, train_epoch_loss_feat_add = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
            )
            train_loss_list.append(train_epoch_loss)
            train_feat_add_loss_list.append(train_epoch_loss_feat_add)

            # validation
            val_epoch_loss, val_epoch_loss_feat_add = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
            )
            val_loss_list.append(val_epoch_loss)
            val_feat_add_loss_list.append(val_epoch_loss_feat_add)
        
            # 学習率の更新
            scheduler.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "loss_feat_add")
                
        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()