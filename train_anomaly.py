from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import torch

from utils import make_train_val_loader_tts, save_loss, get_path_train, check_mel_nar, count_params, set_config
from model.anomaly_detector import AEDetector
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    model, optimizer, scheduler,
    train_loss_list,
    val_loss_list,
    epoch, ckpt_path):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'train_loss_list': train_loss_list,
        'val_loss_list': val_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    model = AEDetector(
        in_channels=cfg.model.n_mel_channels,
        hidden_channels_list=cfg.model.ae_detector_hidden_channels_list,
        n_rnn_layers=cfg.model.ae_detector_n_rnn_layers,
    )
    count_params(model, "model")
    model = model.to(device)
    return model


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("start training")
    model.train()
    
    for batch in train_loader:
        print(f"iter {iter_cnt}/{all_iter}")
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        feature = feature.to(device)
        feature_len = feature_len.to(device)
        
        output = model(feature, feature_len)
        
        loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss += loss.item()
        wandb.log({"train_loss" : loss})
        
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    return epoch_loss


def val_one_epoch(model, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("start validation")
    model.eval()
    
    for batch in val_loader:
        print(f"iter {iter_cnt}/{all_iter}")
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, label = batch
        feature = feature.to(device)
        feature_len = feature_len.to(device)
        
        with torch.no_grad():
            output = model(feature, feature_len)
        
        loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
        
        epoch_loss += loss.item()
        wandb.log({"val_loss" : loss})
        
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_mel_nar(feature[0], output[0], cfg, "mel_validation", current_time, ckpt_time)
            
    epoch_loss /= iter_cnt
    return epoch_loss


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
    torch.backends.cudnn.deterministic = True

    # path
    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader
    train_loader, val_loader, _, _ = make_train_val_loader_tts(cfg, train_data_root, val_data_root)
    
    loss_f = MaskedLoss()
    train_loss_list = []
    val_loss_list = []
    
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        model = make_model(cfg, device)
        
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=cfg.train.lr_decay_exp,
        )
        
        last_epoch = 0
        if cfg.train.check_point_start:
            print("load check point")
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint["train_loss_list"]
            val_loss_list = checkpoint["val_loss_list"]
            last_epoch = checkpoint["epoch"]
            
        wandb.watch(model, **cfg.wandb_conf.watch)
    
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")
            
            epoch_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            
            epoch_loss = val_one_epoch(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            
            scheduler.step()
            
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_loss_list=train_loss_list,
                    val_loss_list=val_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()
            
            
if __name__ == "__main__":
    main()