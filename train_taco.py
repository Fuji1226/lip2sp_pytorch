from socketserver import ThreadingMixIn
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import random
from functools import partial
from librosa.display import specshow

import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly

from utils import make_train_val_loader, get_path_train, save_loss, check_feat_add, check_mel_default, count_params, set_config, calc_class_balance
from model.model_taco import Lip2SPTaco
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

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


def make_model(cfg, device):
    model = Lip2SPTaco(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        norm_type=cfg.model.norm_type_lip,
        inv_up_scale=cfg.model.inv_up_scale,
        sq_r=cfg.model.sq_r,
        md_n_groups=cfg.model.md_n_groups,
        c_attn=cfg.model.c_attn,
        s_attn=cfg.model.s_attn,
        which_res=cfg.model.which_res,
        which_encoder=cfg.model.which_encoder,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        rnn_hidden_channels=cfg.model.rnn_hidden_channels,
        rnn_n_layers=cfg.model.rnn_n_layers,
        dec_n_layers=cfg.model.taco_dec_n_layers,
        dec_hidden_channels=cfg.model.taco_dec_hidden_channels,
        dec_conv_channels=cfg.model.taco_dec_conv_channels,
        dec_conv_kernel_size=cfg.model.taco_dec_conv_kernel_size,
        dec_use_attention=cfg.model.taco_use_attention,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        post_n_layers=cfg.model.post_n_layers,
        post_kernel_size=cfg.model.post_kernel_size,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        reduction_factor=cfg.model.reduction_factor,
    )

    count_params(model, "model")
    count_params(model.ResNet_GAP, "ResNet")
    count_params(model.encoder, "encoder")
    count_params(model.decoder, "decoder")
    count_params(model.postnet, "postnet")

    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, training_method, threshold, ckpt_time):
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        if cfg.train.use_gc:
            output, dec_output = model(lip=lip, data_len=data_len, target=feature, training_method=training_method, threshold=threshold, gc=speaker)               
        else:
            output, dec_output = model(lip=lip, data_len=data_len, target=feature, training_method=training_method, threshold=threshold)               

        B, C, T = output.shape

        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T, speaker=speaker) 
        epoch_output_loss += output_loss.item()
        wandb.log({"train_output_loss": output_loss})
        output_loss.backward(retain_graph=True)

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T, speaker=speaker) 
        epoch_dec_output_loss += dec_output_loss.item()
        wandb.log({"train_dec_output_loss": dec_output_loss})
        dec_output_loss.backward()
        
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)

    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    return epoch_output_loss, epoch_dec_output_loss


def val_one_epoch(model, val_loader, loss_f, device, cfg, training_method, threshold, ckpt_time):
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            if cfg.train.use_gc:
                output, dec_output = model(lip=lip, data_len=data_len, target=feature, training_method=training_method, threshold=threshold, gc=speaker)               
            else:
                output, dec_output = model(lip=lip, data_len=data_len, target=feature, training_method=training_method, threshold=threshold)               

        B, C, T = output.shape

        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T, speaker=speaker) 
        epoch_output_loss += output_loss.item()
        wandb.log({"val_output_loss": output_loss})

        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T, speaker=speaker) 
        epoch_dec_output_loss += dec_output_loss.item()
        wandb.log({"val_dec_output_loss": dec_output_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    return epoch_output_loss, epoch_dec_output_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
        
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True

    # path
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    if len(cfg.train.speaker) > 1:
        class_weight = calc_class_balance(cfg, data_root, device)
    else:
        class_weight = None
    loss_f = MaskedLoss(weight=class_weight, use_weighted_mse=cfg.train.use_weighted_mse)

    train_output_loss_list = []
    train_dec_output_loss_list = []
    val_output_loss_list = []
    val_dec_output_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay    
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
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            if current_epoch < cfg.train.tm_change_step:
                training_method = "tf"
            else:
                training_method = "ss"

            if current_epoch >= cfg.train.threshold_change_step:
                threshold = cfg.train.max_threshold
            else:
                threshold = cfg.train.min_threshold

            epoch_output_loss, epoch_dec_output_loss = train_one_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                training_method=training_method,
                threshold=threshold,
                ckpt_time=ckpt_time,
            )
            train_output_loss_list.append(epoch_output_loss)
            train_dec_output_loss_list.append(epoch_dec_output_loss)

            epoch_output_loss, epoch_dec_output_loss = val_one_epoch(
                model=model,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                training_method=None,
                threshold=None,
                ckpt_time=ckpt_time,
            )
            val_output_loss_list.append(epoch_output_loss)
            val_dec_output_loss_list.append(epoch_dec_output_loss)

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
            
            save_loss(train_output_loss_list, val_output_loss_list, save_path, "output_loss")
            save_loss(train_dec_output_loss_list, val_dec_output_loss_list, save_path, "dec_output_loss")

        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
    
    wandb.finish()


if __name__ == "__main__":
    main()