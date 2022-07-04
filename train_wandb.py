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
from pyparsing import col
import wandb
import copy
from pathlib import Path
import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random

# pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly

# 自作
from get_dir import get_datasetroot, get_data_directory
from model.dataset_npz import KablabDataset, KablabTransform, KablabTransform_val,  MySubset, collate_fn_padding
from model.models import Lip2SP
from loss import masked_loss
from model.discriminator import UNetDiscriminator, JCUDiscriminator
from mf_writer import MlflowWriter
from data_process.feature import delta_feature
from data_check import save_mspec, save_world

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def save_checkpoint(model, optimizer, schedular, epoch, ckpt_path):
	torch.save({'model': model.state_dict(),
				'optimizer': optimizer.state_dict(),
                'schedular': schedular.state_dict(),
                "random": random.getstate(),
                "np_random": np.random.get_state(), 
                "torch": torch.get_rng_state(),
                "torch_random": torch.random.get_rng_state(),
                'cuda_random' : torch.cuda.get_rng_state(),
				'epoch': epoch}, ckpt_path)


def make_train_val_loader(cfg, data_path, mean_std_path):
    """
    pytorchのrandom_splitとかで分けると元のデータセットのtransformを共有してしまうので,Subsetを作ります
    学習用データにはdata augmentationとtime adjust
    検証用データにはtime adjustのみを適用しています(損失計算の並列化のため)
    """
    # 学習用，検証用それぞれに対してtransformを作成
    trans_train = KablabTransform(
        length=cfg.model.length,
        delta=cfg.model.delta
    )
    trans_val = KablabTransform_val(
        length=cfg.model.length,
        delta=cfg.model.delta
    )

    # 元となるデータセットの作成(transformは必ずNoneでお願いします)
    dataset = KablabDataset(
        data_path=data_path,    
        mean_std_path=mean_std_path,
        name=cfg.model.name,
        train=True,
        cfg=cfg,
        debug=cfg.train.debug,
        transform=None,
    )

    # 学習用と検証用にデータセットを分割
    n_samples = len(dataset)
    train_size = int(n_samples * 0.95)
    indices = np.arange(n_samples)
    train_dataset = MySubset(
        dataset=dataset,
        indices=indices[:train_size],
        transform=trans_train,
    )
    val_dataset = MySubset(
        dataset=dataset,
        indices=indices[train_size:],
        transform=trans_val,
    )

    # それぞれのdata loaderを作成
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


def make_model(cfg, device):
    model = Lip2SP(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            res_layers=cfg.model.res_layers,
            d_model=cfg.model.d_model,
            n_layers=cfg.model.n_layers,
            n_head=cfg.model.n_head,
            dec_n_layers=cfg.model.dec_n_layers,
            dec_d_model=cfg.model.dec_d_model,
            glu_inner_channels=cfg.model.d_model,
            glu_layers=cfg.model.glu_layers,
            pre_in_channels=cfg.model.pre_in_channels,
            pre_inner_channels=cfg.model.pre_inner_channels,
            post_inner_channels=cfg.model.post_inner_channels,
            n_position=cfg.model.length * 5,
            # n_position=cfg.model.length,
            max_len=cfg.model.length // 2,
            which_encoder=cfg.model.which_encoder,
            which_decoder=cfg.model.which_decoder,
            training_method=cfg.train.training_method,
            num_passes=cfg.train.num_passes,
            mixing_prob=cfg.train.mixing_prob,
            apply_first_bn=cfg.train.apply_first_bn,
            dropout=cfg.train.dropout,
            reduction_factor=cfg.model.reduction_factor,
            use_gc=cfg.train.use_gc,
        ).to(device)
    return model
    

def train_one_epoch(model: nn.Module, train_loader, optimizer, loss_f_train, device, cfg, training_method):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    # with detect_anomaly():
    model.train()
    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        # lip, feature, feat_add, data_len, speaker, label, max_len = batch
        # lip, feature, feat_add, data_len, max_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), max_len.to(device)

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        # output : postnet後の出力
        # dec_output : postnet前の出力
        output, dec_output, enc_output = model(
            lip=lip,
            data_len=data_len,
            prev=feature,
            training_method=training_method,
        )                

        rf = cfg.model.reduction_factor      
        
        # パディング部分をマスクした損失を計算。postnet前後の出力両方を考慮。
        # output_loss = loss_f_train.mse_loss(output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # dec_output_loss = loss_f_train.mse_loss(dec_output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # delta_loss = loss_f_train.delta_loss(output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2)
            
        #  decoderは自己回帰なのでずらすけど，outputはpostnetを通すだけなのでずらしちゃだめだったかも
        # output_loss = loss_f_train.mse_loss(output, feature, data_len, max_len=model.max_len * 2) 
        # dec_output_loss = loss_f_train.mse_loss(dec_output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # delta_loss = loss_f_train.delta_loss(output, feature, data_len, max_len=model.max_len * 2)

        # transformerのdecoder出力の時点でシフトすることにより，最終的な損失ではシフトしないでいいようにした
        # output_loss = loss_f_train.mse_loss(output, feature[..., rf:], data_len - rf, max_len=model.max_len * rf - rf) 
        # dec_output_loss = loss_f_train.mse_loss(dec_output, feature[..., rf:], data_len - rf, max_len=model.max_len * rf - rf) 
        # delta_loss = loss_f_train.delta_loss(output, feature[..., rf:], data_len - rf, max_len=model.max_len * rf - rf)

        
        output_loss = loss_f_train.mse_loss(output, feature, data_len, max_len=model.max_len * rf) 
        dec_output_loss = loss_f_train.mse_loss(dec_output, feature, data_len, max_len=model.max_len * rf) 
        delta_loss = loss_f_train.delta_loss(output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur)

        loss = output_loss + dec_output_loss + delta_loss
        
        # if torch.isnan(loss):
        #     print("loss is nan!")
        #     if torch.isnan(output_loss):
        #         print("output_loss is nan!")
        #     if torch.isnan(dec_output_loss):
        #         print("dec_output_loss is nan!")
        #     if torch.isnan(delta_loss):
        #         print("delta_loss is nan!")
        #     if torch.isnan(output):
        #         print("output is nan!")
        #     if torch.isnan(feature):
        #         print("feature is nan!")
        #     if torch.isnan(data_len):
        #         print("data_len is nan!")
        #     if torch.isnan(lip):
        #         print("lip is nan!")

        #     assert loss is not None
        #     model = make_model(cfg, device)
        #     model.load_state_dict(prev_model)
        #     optimizer = torch.optim.Adam(model.parameters())
        #     optimizer.load_state_dict(prev_optimizer)
            
        # else:
        #     prev_model = model.state_dict()
        #     prev_optimizer = optimizer.state_dict()

        # 勾配の初期化
        optimizer.zero_grad()

        loss.backward()

        # gradient clipping（max_norm=3.0）
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        optimizer.step()
        epoch_loss += loss.item()
        
        wandb.log({"train_output_loss": output_loss.item()})
        wandb.log({"train_dec_output_loss": dec_output_loss.item()})
        wandb.log({"train_delta_loss": delta_loss.item()})
        wandb.log({"train_total_loss": loss.item()})

    # epoch_loss /= data_cnt
    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model: nn.Module, val_loader, loss_f_val, device, cfg):
    epoch_loss = 0
    data_cnt = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()
    for batch in val_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')
        
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)
        batch_size = lip.shape[0]
        data_cnt += batch_size
        
        with torch.no_grad():
            # output, dec_output, enc_output = model.inference(
            #     lip=lip,
            # )            
            output, dec_output, enc_output = model(lip)

        rf = cfg.model.reduction_factor
        output_loss = loss_f_val.mse_loss(output, feature, data_len, max_len=model.max_len * rf) 
        dec_output_loss = loss_f_val.mse_loss(dec_output, feature, data_len, max_len=model.max_len * rf) 
        delta_loss = loss_f_val.delta_loss(output, feature, data_len, max_len=model.max_len * rf, device=device, blur=cfg.train.blur)

        # output_loss = loss_f_val.mse_loss(output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # dec_output_loss = loss_f_val.mse_loss(dec_output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        # delta_loss = loss_f_val.delta_loss(output[:, :, :-2], feature[:, :, 2:], data_len, max_len=model.max_len * 2) 
        loss = output_loss + dec_output_loss + delta_loss
        
        epoch_loss += loss.item()

        wandb.log({"val_output_loss": output_loss.item()})
        wandb.log({"val_dec_output_loss": dec_output_loss.item()})
        wandb.log({"val_delta_loss": delta_loss.item()})
        wandb.log({"val_total_loss": loss.item()})
    
    # epoch_loss /= data_cnt
    epoch_loss /= iter_cnt
    return epoch_loss


def save_result(loss_list, save_path):
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.savefig(save_path)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    assert cfg.model.which_d is None

    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    torch.backends.cudnn.benchmark = True

    # 口唇動画か顔かの選択
    lip_or_face = cfg.train.face_or_lip
    assert lip_or_face == "face" or "lip"
    if lip_or_face == "face":
        data_path = cfg.train.face_pre_loaded_path
        mean_std_path = cfg.train.face_mean_std_path
    elif lip_or_face == "lip":
        data_path = cfg.train.lip_pre_loaded_path
        mean_std_path = cfg.train.lip_mean_std_path

    print("--- data directory check ---")
    print(f"data_path = {data_path}")
    print(f"mean_std_path = {mean_std_path}")

    #resultの表示
    result_path = 'results'
    # os.makedirs(result_path, exist_ok=True)

    # check point
    ckpt_path = os.path.join(cfg.train.ckpt_path, lip_or_face, current_time)
    os.makedirs(ckpt_path, exist_ok=True)

    # モデルパラメータの保存先を指定
    save_path = os.path.join(cfg.train.train_save_path, lip_or_face, current_time)
    os.makedirs(save_path, exist_ok=True)
    
    # Dataloader作成
    train_loader, val_loader, _ = make_train_val_loader(cfg, data_path, mean_std_path)
    # test_loader, _ = make_test_loader(cfg)
    
    # 損失関数
    loss_f_train = masked_loss(train=True)
    loss_f_val = masked_loss(train=False)
    train_loss_list = []
    
    # training
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg) as run:
        # model
        model = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        # schedular
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.train.max_epoch // cfg.train.lr_decay_step, 
            gamma=cfg.train.lr_decay_rate    
        )

        if cfg.train.check_point_start:
            checkpoint_path = "/home/usr4/r70264c/lip2sp_pytorch/check_point/default/2022:06:24_10-36-39/mspec_40.ckpt"
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["schedular"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])

        wandb.watch(model, **cfg.wandb_conf.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch

        # teacher forcingとscheduled samplingの切り替え(田口さんがやっていた)
        training_method_change_step = max_epoch * cfg.train.tm_change_step
        
        for epoch in range(max_epoch):
            print(f"##### {epoch} #####")

            if epoch < training_method_change_step:
                training_method = "tf"  # teacher forcing
            else:
                training_method = "ss"  # scheduled sampling
            print(f"training_method : {training_method}")

            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f_train=loss_f_train, 
                device=device, 
                cfg=cfg, 
                training_method=training_method,
            )
            train_loss_list.append(epoch_loss)
            print(f"epoch_loss = {epoch_loss}")
            print(f"train_loss_list = {train_loss_list}")

            # 検証用データ
            if epoch % cfg.train.display_val_loss_step == 0:
                epoch_loss_test = calc_val_loss(
                    model=model, 
                    val_loader=val_loader, 
                    loss_f_val=loss_f_val, 
                    device=device, 
                    cfg=cfg,
                )
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
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.ckpt")
                )
                # wandb.save(os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.cpt"), base_path="/check_point")
                # artifact_ckpt = wandb.Artifact('ckpt', type='ckpt')
                # artifact_ckpt.add_file(os.path.join(ckpt_path, f"{cfg.model.name}_{epoch}.cpt"))
                # wandb.log_artifact(artifact_ckpt)
                
        # モデルの保存(wandbのartifactはうまくいってません)
        torch.save(model.state_dict(), os.path.join(save_path, f"model_{cfg.model.name}.pth"))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(os.path.join(save_path, f"model_{cfg.model.name}.pth"))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()