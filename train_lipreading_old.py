"""
Lipreadingの学習用
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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from timm.scheduler import CosineLRScheduler

# 自作
from dataset.dataset_lipreading import LipReadingDataset, LipReadingTransform, get_data_simultaneously, collate_time_adjust
from data_process.phoneme_encode import get_classes, IGNORE_INDEX, SOS_INDEX, EOS_INDEX
from model.model_lipreading import Lip2Text
from data_process.phoneme_encode import get_keys_from_value

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(7)
torch.manual_seed(7)
torch.cuda.manual_seed_all(7)
random.seed(7)


def save_checkpoint(model, optimizer, schedular, epoch, ckpt_path):
	torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'schedular': schedular.state_dict(),
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
    # wandb.log({f"{filename}": plt})
    # wandb.log({f"Image {filename}": wandb.Image(os.path.join(save_path, f"{filename}.png"))})
    wandb.log({f"loss {filename}": wandb.plot.line_series(
        xs=np.arange(len(train_loss_list)), 
        ys=[train_loss_list, val_loss_list],
        keys=["train loss", "validation loss"],
        title=f"{filename}",
        xname="epoch",
    )})


def make_train_val_loader(cfg, data_root, mean_std_path):
    data_path = get_data_simultaneously(
        data_root=data_root,
        name=cfg.model.name,
    )
    classes = get_classes(data_path)
    n_samples = len(data_path)
    train_size = int(n_samples * 0.95)
    train_data_path = data_path[:train_size]
    val_data_path = data_path[train_size:]

    train_trans = LipReadingTransform(
        cfg=cfg,
        train_val_test="train",
    )
    val_trans = LipReadingTransform(
        cfg=cfg,
        train_val_test="val",
    )

    train_dataset = LipReadingDataset(
        data_path=train_data_path,
        mean_std_path = mean_std_path,
        transform=train_trans,
        cfg=cfg,
        test=False,
        classes=classes,
    )
    val_dataset = LipReadingDataset(
        data_path=val_data_path,
        mean_std_path=mean_std_path,
        transform=val_trans,
        cfg=cfg,
        test=False,
        classes=classes,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=cfg.train.num_workers,      
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_time_adjust,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,   
        shuffle=True,
        num_workers=0,      
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_time_adjust,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def make_model(cfg, device):
    model = Lip2Text(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.n_classes,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        dec_n_layers=cfg.model.dec_n_layers,
        dec_d_model=cfg.model.dec_d_model,
        dec_n_head=cfg.model.dec_n_head,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        apply_first_bn=cfg.train.apply_first_bn,
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
    

def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, training_method, mixing_prob, epoch, dataset):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    model.train()

    classes_index = dataset.classes_index

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch

        # phoneme_indexにsosとeosを追加
        p_list = phoneme_index.tolist()
        for p in p_list:
            p.insert(0, SOS_INDEX)
            p.append(EOS_INDEX)
        phoneme_index = torch.tensor(p_list)

        # print(f"lip = {lip.shape}, phoneme_index = {phoneme_index.shape}")
        # print(f"phoneme_index = {phoneme_index}")

        lip, phoneme_index, data_len = lip.to(device), phoneme_index.to(device), data_len.to(device)

        # sosからがmodelへのinput
        phoneme_index_input = phoneme_index[:, :-1]    # (B, T)

        # eosまでがoutputに対してのlabel
        phoneme_index_output = phoneme_index[:, 1:]    # (B, T)

        output = model(lip, phoneme_index_input, data_len, training_method, mixing_prob)    # (B, C, T)

        loss = loss_f(output, phoneme_index_output)
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_loss += loss.item()
        wandb.log({"train_iter_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break

        if iter_cnt > (all_iter - 1):
            with torch.no_grad():
                # softmaxを適用して確率に変換
                output = torch.softmax(output, dim=1)
                
                # Onehot
                output = torch.distributions.OneHotCategorical(output).sample()
                
                # 最大値(Onehotの1のところ)のインデックスを取得
                output = output.max(dim=1)[1]   # (B, T)

                phoneme_index_output = phoneme_index_output[0]
                output = output[0]

                # 音素を数値列から元の音素ラベルに戻す
                phoneme_answer = []
                for i in phoneme_index_output:
                    phoneme_answer.append(get_keys_from_value(classes_index, i))
                phoneme_answer = " ".join(phoneme_answer)
                
                phoneme_predict = []
                for i in output:
                    phoneme_predict.append(get_keys_from_value(classes_index, i))
                phoneme_predict = " ".join(phoneme_predict)

                save_path = Path("~/lip2sp_pytorch/data_check/lipreading").expanduser()
                save_path = save_path / current_time
                os.makedirs(save_path, exist_ok=True)
                
                with open(str(save_path / "phoneme.txt"), "a") as f:
                    f.write(f"\n--- epoch {epoch} : iter {iter_cnt} ---\n")
                    f.write("answer\n")
                    f.write(f"{phoneme_answer}\n")
                    f.write("\npredict\n")
                    f.write(f"{phoneme_predict}\n")

    epoch_loss /= iter_cnt
    return epoch_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg, training_method, mixing_prob):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        wav, lip, feature, feat_add, phoneme_index, data_len, speaker, label = batch

        p_list = phoneme_index.tolist()
        for p in p_list:
            p.insert(0, SOS_INDEX)
            p.append(EOS_INDEX)
        phoneme_index = torch.tensor(p_list)

        lip, phoneme_index, data_len = lip.to(device), phoneme_index.to(device), data_len.to(device)

        phoneme_index_input = phoneme_index[:, :-1]
        phoneme_index_output = phoneme_index[:, 1:]

        with torch.no_grad():
            output = model(lip, phoneme_index_input, data_len, training_method, mixing_prob)

        loss = loss_f(output, phoneme_index_output)
        epoch_loss += loss.item()
        wandb.log({"val_iter_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                break
            
    epoch_loss /= iter_cnt
    return epoch_loss


@hydra.main(config_name="config", config_path="conf")
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

    # 口唇動画か顔かの選択
    lip_or_face = cfg.train.face_or_lip
    assert lip_or_face == "face" or "lip"
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

    print("--- data directory check ---")
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
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader(cfg, data_root, mean_std_path)
    
    # 損失関数
    loss_f = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    train_loss_list = []
    val_loss_list = []
    
    # training
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
        )

        # scheduler
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(
        #     optimizer=optimizer,
        #     milestones=cfg.train.multi_lr_decay_step,
        #     gamma=cfg.train.lr_decay_rate,
        # )
        scheduler = CosineLRScheduler(
            optimizer, 
            t_initial=cfg.train.max_epoch, 
            lr_min=cfg.train.lr / 10, 
            warmup_t=20, 
            warmup_lr_init=1e-5, 
            warmup_prefix=True,
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = cfg.train.start_ckpt_path
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["schedular"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)

        if cfg.train.debug:
            max_epoch = cfg.train.debug_max_epoch
        else:
            max_epoch = cfg.train.max_epoch
        
        for epoch in range(max_epoch - last_epoch):
            current_epoch = epoch + last_epoch
            print(f"##### {current_epoch} #####")

            # 学習方法の変更
            if current_epoch < cfg.train.tm_change_step:
                training_method = "tf"  # teacher forcing
            else:
                training_method = "ss"  # scheduled sampling

            # mixing_probの変更
            if cfg.train.change_mixing_prob:
                if current_epoch >= cfg.train.mp_change_step:
                    if cfg.train.fixed_mixing_prob:
                        mixing_prob = 0.1
                    else:
                        mixing_prob = torch.randint(10, 50, (1,)) / 100     # [0.1, 0.5]でランダム
                        mixing_prob = mixing_prob.item()
                else:
                    mixing_prob = cfg.train.mixing_prob
            else:
                mixing_prob = cfg.train.mixing_prob

            print(f"training_method : {training_method}")
            print(f"mixing_prob = {mixing_prob}")
            # print(f"learning_rate = {scheduler.get_last_lr()}")
            print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # training
            train_epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                training_method=training_method,
                mixing_prob=mixing_prob,
                epoch=current_epoch,
                dataset=train_dataset,
            )
            train_loss_list.append(train_epoch_loss)

            # validation
            val_epoch_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                training_method=training_method,
                mixing_prob=mixing_prob,
            )
            val_loss_list.append(val_epoch_loss)
        
            # 学習率の更新
            # scheduler.step()
            scheduler.step(current_epoch + 1)

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    schedular=scheduler,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
                
        # モデルの保存(wandbのartifactはうまくいってません)
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()