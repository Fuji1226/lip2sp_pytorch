"""
wandbを使って記録するバージョン

最近はこっちをメインで使ってます
train.pyはあまり触ってません
また,こっちもdiscriminatorはいじってません
分岐が多くなってごちゃごちゃしてきたので,こっちはdiscriminatorを使わない場合のtrain,使う場合はtrain_d.pyを使用するようにしていこうと考えてます

datasetを作成するとき,事前に作ったnpzファイルを読み込むように変更しました
なので,data_process/make_npz.pyでnpzを作ってから出ないと実行できなくなっていると思います
KablabDatasetのdata_root,mean_std_pathです
"""

from omegaconf import DictConfig, OmegaConf
import hydra
import mlflow
import wandb

from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# 自作
from get_dir import get_datasetroot, get_data_directory
# from model.dataset_no_chainer import KablabDataset, KablabTransform
from model.dataset_npz import KablabDataset, KablabTransform
from model.models import Lip2SP
from loss import masked_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from mf_writer import MlflowWriter
from data_process.feature import delta_feature

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# check pointでの保存
def save_checkpoint(model, optimizer, schedular, epoch, ckpt_path):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def make_train_val_loader(cfg):
    trans = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    dataset = KablabDataset(
        # data_root=cfg.train.train_path,
        data_root=cfg.train.pre_loaded_path,    # npzファイルまでのパス
        mean_std_path=cfg.train.mean_std_path,
        name=cfg.model.name,
        train=True,
        transforms=trans,
        cfg=cfg,
        debug=cfg.train.debug
    )

    # 学習用と検証用にデータセットを分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.95)
    val_size = n_samples - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    val_dataset.train = False

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=1,   
        shuffle=False,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=None,
    )
    return train_loader, val_loader, dataset


# def make_test_loader(cfg):
#     trans = KablabTransform(
#         length=cfg.model.length,
#         delta=cfg.model.delta
#     )
#     datasets = KablabDataset(
#         data_root=cfg.train.test_path,
#         train=False,
#         transforms=trans,
#         cfg=cfg,
#     )
#     test_loader = DataLoader(
#         dataset=datasets,
#         # batch_size=cfg.train.batch_size,
#         batch_size=1,   
#         shuffle=True,
#         num_workers=cfg.train.num_workers,      
#         pin_memory=False,
#         drop_last=True,
#         collate_fn=None,
#     )
#     return test_loader, datasets
    

def train_one_epoch(model: nn.Module, train_loader, optimizer, loss_f_mse, loss_f_train, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()
    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        # output : postnet後の出力
        # dec_output : postnet前の出力
        output, dec_output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
        )                      
        
        loss_default = loss_f_mse(output[:, :, :-2], target[:, :, 2:]) + loss_f_mse(dec_output[:, :, :-2], target[:, :, 2:])
        # パディング部分をマスクした損失を計算。postnet前後の出力両方を考慮。
        loss = loss_f_train.mse_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + loss_f_train.mse_loss(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                + loss_f_train.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        loss.backward()

        # gradient clipping（max_norm=3.0）
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        optimizer.step()
        epoch_loss += loss.item()
        
        wandb.log({"train_iter_loss": loss.item()})
    
    # epoch_loss /= data_cnt
    epoch_loss /= iter_cnt
    return epoch_loss


def train_one_epoch_with_d(model: nn.Module, discriminator, train_loader, optimizer, optimizer_d, loss_f_mse, loss_f_train, device, cfg):
    assert cfg.model.which_d is not None, "discriminatorが設定されていません!"
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    for batch in train_loader:
        model.train()
        discriminator.train()
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size

        #====================================================
        # discriminatorの最適化
        # generator1回に対して複数回最適化するコードもあり。とりあえず1回で実装。
        #====================================================
        # discriminatorのパラメータ更新を行うように設定
        for param in discriminator.parameters():
            param.requires_grad = True

        # output : postnet後の出力
        # dec_output : postnet前の出力
        # generatorのパラメータが更新されないように設定
        with torch.no_grad():
            output, dec_output = model(
                lip=lip,
                data_len=data_len,
                prev=target,
            )                      
        
        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output[:, :, :-2])   # 生成データを入力
            out_r, fmaps_r = discriminator(target[:, :, 2:])    # 実データを入力
            # 損失計算
            loss_d = loss_f_train.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="d")
        except Exception:
            print("error")
        
        # 最適化
        loss_d.backward()
        optimizer_d.step()
        epoch_loss_d += loss_d.item()

        #====================================================
        # generatorの最適化
        #====================================================
        # discriminatorのパラメータ更新を行わないように設定
        for param in discriminator.parameters():
            param.requires_grad = False
        
        # generatorでデータ生成
        output, dec_output = model(
            lip=lip,
            data_len=data_len,
            prev=target,
        )

        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output[:, :, :-2])   # 生成データを入力
            out_r, fmaps_r = discriminator(target[:, :, 2:])    # 実データを入力
            # 損失計算
            loss_g_ls = loss_f_train.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="g")
            loss_g_fm = loss_f_train.fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d)
        except Exception:
            print("error")

        # loss_default = loss_f_mse(output[:, :, :-2], target[:, :, 2:]) + loss_f_mse(dec_output[:, :, :-2], target[:, :, 2:])
        # loss_mask = loss_f_train.masked_mse(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
        #     + loss_f_train.masked_mse(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # loss_delta = loss_f_train.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 

        # postnet前後のmse_lossと、GAN関連のlossの和（実際は重みづけが必要そう）
        loss = loss_f_train.mse_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
            + loss_f_train.mse_loss(dec_output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                + loss_f_train.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) \
                    + loss_g_ls + loss_g_fm
        
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    # epoch_loss /= data_cnt
    # epoch_loss_d /= data_cnt
    epoch_loss /= iter_cnt
    epoch_loss_d /= iter_cnt
    return epoch_loss, epoch_loss_d


def calc_val_loss(model: nn.Module, val_loader, loss_f_test, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc test loss")
    model.eval()
    for batch in val_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip,
            )                      
        
        loss = loss_f_test.mse_loss(output, target, data_len, max_len=model.max_len * 2) \
            + loss_f_test.mse_loss(dec_output, target, data_len, max_len=model.max_len * 2) \
                + loss_f_test.delta_loss(output[:, :, :-2], target[:, :, 2:], data_len, max_len=model.max_len * 2) 
        epoch_loss += loss.item()

        wandb.log({"val_iter_loss": loss.item()})
    
    # epoch_loss /= data_cnt
    epoch_loss /= iter_cnt
    return epoch_loss


def calc_test_loss_with_d(model: nn.Module, discriminator, test_loader, loss_f_test, device, cfg):
    epoch_loss = 0
    epoch_loss_d = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(test_loader)
    print("calc test loss")
    for batch in test_loader:
        model.eval()
        discriminator.eval()
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        (lip, target, feat_add), data_len, label = batch
        lip, target, feat_add, data_len = lip.to(device), target.to(device), feat_add.to(device), data_len.to(device)
        batch_size = lip.shape[0]
        data_cnt += batch_size

        # generatorでデータ生成
        with torch.no_grad():
            output, dec_output = model.inference(
                lip=lip,
            )    
    
        # discriminatorへ入力
        try:
            out_f, fmaps_f = discriminator(output)   # 生成データを入力
            out_r, fmaps_r = discriminator(target)    # 実データを入力
            # 損失計算
            # 最適化しないのでdiscriminator、generator同時に求めています
            loss_d = loss_f_test.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="d")
            loss_g_ls = loss_f_test.ls_loss(out_f, out_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d, which_loss="g")
            loss_g_fm = loss_f_test.fm_loss(fmaps_f, fmaps_r, data_len, max_len=model.max_len * 2, which_d=cfg.model.which_d)
        except Exception:
            print("error")

        loss = loss_f_test.mse_loss(output, target, data_len, max_len=model.max_len * 2) \
            + loss_f_test.mse_loss(dec_output, target, data_len, max_len=model.max_len * 2) \
                + loss_f_test.delta_loss(output, target, data_len, max_len=model.max_len * 2) \
                    + loss_g_ls + loss_g_fm

        epoch_loss += loss.item()
        epoch_loss_d += loss_d.item()
    
    # epoch_loss /= data_cnt
    # epoch_loss_d /= data_cnt
    epoch_loss /= iter_cnt
    epoch_loss_d /= iter_cnt
    return epoch_loss, epoch_loss_d


def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    if cfg.train.debug:
        cfg.train.max_epoch = 10

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    print(os.cpu_count())

    #resultの表示
    result_path = 'results'
    # os.makedirs(result_path, exist_ok=True)

    # check point
    ckpt_path = os.path.join(cfg.train.ckpt_path, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = os.path.join(cfg.train.train_save_path, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    train_loader, val_loader, _ = make_train_val_loader(cfg)
    # test_loader, _ = make_test_loader(cfg)
    
    # 損失関数
    loss_f_mse = nn.MSELoss()
    loss_f_mae = nn.L1Loss()
    loss_f_train = masked_loss(train=True)
    loss_f_test = masked_loss(train=False)
    train_loss_list = []
    
    # training
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg) as run:
        #インスタンス作成
        model = Lip2SP(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            res_layers=cfg.model.res_layers,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_head=cfg.model.n_head,
            glu_inner_channels=cfg.model.d_model,
            glu_layers=cfg.model.glu_layers,
            pre_in_channels=cfg.model.pre_in_channels,
            pre_inner_channels=cfg.model.pre_inner_channels,
            post_inner_channels=cfg.model.post_inner_channels,
            n_position=cfg.model.length * 5,
            max_len=cfg.model.length // 2,
            which_encoder=cfg.model.which_encoder,
            which_decoder=cfg.model.which_decoder,
            training_method=cfg.train.training_method,
            num_passes=cfg.train.num_passes,
            mixing_prob=cfg.train.mixing_prob,
            dropout=cfg.train.dropout,
            reduction_factor=cfg.model.reduction_factor,
            use_gc=cfg.train.use_gc
        ).to(device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay     # 1.0e-6
        )

        # schedular
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.train.max_epoch // 4, 
            gamma=cfg.train.lr_decay_rate       # 0.5
        )

        # discriminator
        if cfg.model.which_d is not None:
            if cfg.model.which_d == "unet":
                discriminator = UNetDiscriminator()
            elif cfg.model.which_d == "jcu":
                discriminator = JCUDiscriminator(
                    in_channels=cfg.model.out_channels,
                    out_channels=1,
                    use_gc=cfg.train.use_gc,
                    emb_in=cfg.train.batch_size,
                )
            optimizer_d = torch.optim.Adam(
                discriminator.parameters(), lr=cfg.train.lr, betas=(cfg.train.beta_1, cfg.train.beta_2)
            )
            discriminator.to(device)
        else:
            discriminator = None

        wandb.watch(model, **cfg.wandb_conf.watch)

        if cfg.model.which_d is None:
            for epoch in range(cfg.train.max_epoch):
                print(f"##### {epoch} #####")
                epoch_loss = train_one_epoch(model, train_loader, optimizer, loss_f_mse, loss_f_train, device, cfg)
                train_loss_list.append(epoch_loss)
                print(f"epoch_loss = {epoch_loss}")
                print(f"train_loss_list = {train_loss_list}")

                # 検証用データ
                if epoch % cfg.train.display_val_loss_step == 0:
                    epoch_loss_test = calc_val_loss(model, val_loader, loss_f_test, device, cfg)
                    print(f"epoch_loss_test = {epoch_loss_test}")
                
                # 学習率の更新
                scheduler.step()

                # check point
                if epoch % cfg.train.ckpt_step == 0:
                    save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        schedular=scheduler,
                        epoch=epoch,
                        ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.cpt")
                    )
                    artifact_ckpt = wandb.Artifact('ckpt', type='ckpt')
                    artifact_ckpt.add_file(os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.cpt"))
                    wandb.log_artifact(artifact_ckpt)

            # モデルの保存
            torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}.pth"))
            artifact_model = wandb.Artifact('model', type='model')
            artifact_model.add_file(os.path.join(save_path, f"model_{cfg.model.name}.pth"))
            wandb.log_artifact(artifact_model)
            
        else:
            for epoch in range(cfg.train.max_epoch):
                print(f"##### {epoch} #####")
                epoch_loss, epoch_loss_d = train_one_epoch_with_d(model, discriminator, train_loader, optimizer, optimizer_d, loss_f_mse, loss_f_train, device, cfg)
                train_loss_list.append(epoch_loss)
                print(f"epoch_loss = {epoch_loss}")
                print(f"epoch_loss_d = {epoch_loss_d}")
                print(f"train_loss_list = {train_loss_list}")

                if epoch % cfg.train.display_test_loss_step == 0:
                    epoch_loss_test, epoch_loss_d_test = calc_test_loss_with_d(model, discriminator, val_loader, loss_f_test, device, cfg)
                    print(f"epoch_loss_test = {epoch_loss_test}")
                    print(f"epoch_loss_d_test = {epoch_loss_d_test}")
            
            save_result(train_loss_list, result_path+'/train_loss.png')
            torch.save(model.state_dict(), save_path+f'/model_{cfg.model.name}_{cfg.model.which_d}.pth')

    wandb.finish()


if __name__=='__main__':
    main()