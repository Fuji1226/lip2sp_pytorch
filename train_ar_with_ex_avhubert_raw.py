from omegaconf import OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random
import torch
from timm.scheduler import CosineLRScheduler

from utils import (
    count_params,
    set_config,
    save_loss,
    check_mel_ar,
    check_attention_weight,
    requires_grad_change,
    fix_random_seed,
    set_requires_grad_by_name,
    get_path_train_raw,
    make_train_val_loader_with_external_data_raw,
)
from model.model_ar_avhubert import Lip2SP_AR_AVHubert
from loss import MaskedLoss

wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    scaler,
    train_loss_list,
    train_output_mse_loss_list,
    train_dec_output_mse_loss_list,
    val_loss_list,
    val_output_mse_loss_list,
    val_dec_output_mse_loss_list,
    epoch,
    ckpt_path,
):
    torch.save(
        {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            "random": random.getstate(),
            "np_random": np.random.get_state(), 
            "torch": torch.get_rng_state(),
            "torch_random": torch.random.get_rng_state(),
            'cuda_random' : torch.cuda.get_rng_state(),
            'train_loss_list': train_loss_list,
            'train_output_mse_loss_list': train_output_mse_loss_list, 
            'train_dec_output_mse_loss_list': train_dec_output_mse_loss_list, 
            'val_loss_list': val_loss_list, 
            'val_output_mse_loss_list': val_output_mse_loss_list, 
            'val_dec_output_mse_loss_list': val_dec_output_mse_loss_list, 
            'epoch': epoch, 
        },
        ckpt_path,
    )


def make_model(cfg, device):
    model = Lip2SP_AR_AVHubert(
        avhubert_config=cfg.model.avhubert_config,
        avhubert_model_size=cfg.model.avhubert_model_size,
        avhubert_return_res_output=cfg.model.avhubert_return_res_output,
        load_avhubert_pretrained_weight=cfg.model.load_avhubert_pretrained_weight,
        avhubert_layer_loaded=cfg.model.avhubert_layer_loaded,
        reduction_factor=cfg.model.reduction_factor,
        which_decoder=cfg.model.which_decoder,
        out_channels=cfg.model.out_channels,
        dec_dropout=cfg.train.dec_dropout,
        use_spk_emb=cfg.train.use_spk_emb,
        spk_emb_dim=cfg.model.spk_emb_dim,
        pre_inner_channels=cfg.model.pre_inner_channels,
        glu_layers=cfg.model.glu_layers,
        glu_kernel_size=cfg.model.glu_kernel_size,
        dec_channels=cfg.model.taco_dec_channels,
        dec_atten_conv_channels=cfg.model.taco_dec_conv_channels,
        dec_atten_conv_kernel_size=cfg.model.taco_dec_conv_kernel_size,
        dec_atten_hidden_channels=cfg.model.taco_dec_atten_hidden_channels,
        prenet_hidden_channels=cfg.model.taco_dec_prenet_hidden_channels,
        prenet_inner_channels=cfg.model.taco_dec_prenet_inner_channels,
        prenet_dropout=cfg.model.taco_lip_prenet_dropout,
        lstm_n_layers=cfg.model.taco_dec_n_layers,
        post_inner_channels=cfg.model.post_inner_channels,
        post_n_layers=cfg.model.post_n_layers,
        post_kernel_size=cfg.model.post_kernel_size,
        use_attention=cfg.model.taco_use_attention,
    )
    model.decoder.training_method = cfg.train.training_method
    count_params(model, "model")
    return model.to(device)


def train_one_epoch(
    model,
    train_loader,
    optimizer,
    scaler,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    epoch_loss = 0
    epoch_output_mse_loss = 0
    epoch_dec_output_mse_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("training") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output, dec_output, att_w = model(
                lip=lip,
                audio=None,
                lip_len=lip_len,
                spk_emb=spk_emb,
                feature_target=feature,
            )
            output_mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
            dec_output_mse_loss = loss_f.mse_loss(dec_output, feature, feature_len, max_len=output.shape[-1])
            loss = output_mse_loss + dec_output_mse_loss
            epoch_loss += loss.item()
            epoch_output_mse_loss += output_mse_loss.item()
            epoch_dec_output_mse_loss += dec_output_mse_loss.item()
            wandb.log({"train_loss": loss})
            wandb.log({"train_output_mse_loss": output_mse_loss})
            wandb.log({"train_dec_output_mse_loss": dec_output_mse_loss})

            loss = loss / cfg.train.iters_to_accumulate

        scaler.scale(loss).backward()

        if (iter_cnt + 1) % cfg.train.iters_to_accumulate == 0 or (iter_cnt + 1) % (all_iter - 1) == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_ar(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)
                check_attention_weight(att_w[0], cfg, 'att_w_train', current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            check_mel_ar(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)
            check_attention_weight(att_w[0], cfg, 'att_w_train', current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_output_mse_loss /= iter_cnt
    epoch_dec_output_mse_loss /= iter_cnt
    return epoch_loss, epoch_output_mse_loss, epoch_dec_output_mse_loss


def val_one_epoch(
    model,
    val_loader,
    loss_f,
    device,
    cfg,
    ckpt_time,
):
    epoch_loss = 0
    epoch_output_mse_loss = 0
    epoch_dec_output_mse_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("validation")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, feature, feature_avhubert, spk_emb, feature_len, lip_len, speaker, speaker_idx, filename, lang_id, is_video = batch
        lip = lip.to(device)
        feature = feature.to(device)
        feature_avhubert = feature_avhubert.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                output, dec_output, att_w = model(
                    lip=lip,
                    audio=None,
                    lip_len=lip_len,
                    spk_emb=spk_emb,
                    feature_target=None,
                )
            output_mse_loss = loss_f.mse_loss(output, feature, feature_len, max_len=output.shape[-1])
            dec_output_mse_loss = loss_f.mse_loss(dec_output, feature, feature_len, max_len=output.shape[-1])
            loss = output_mse_loss + dec_output_mse_loss
            epoch_loss += loss.item()
            epoch_output_mse_loss += output_mse_loss.item()
            epoch_dec_output_mse_loss += dec_output_mse_loss.item()
            wandb.log({"val_loss": loss})
            wandb.log({"val_output_mse_loss": output_mse_loss})
            wandb.log({"val_dec_output_mse_loss": dec_output_mse_loss})
            
        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_mel_ar(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                check_attention_weight(att_w[0], cfg, 'att_w_validation', current_time, ckpt_time)
                break

        if all_iter - 1 > 0:
            if iter_cnt % (all_iter - 1) == 0:
                check_mel_ar(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                check_attention_weight(att_w[0], cfg, 'att_w_validation', current_time, ckpt_time)
        else:
            check_mel_ar(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
            check_attention_weight(att_w[0], cfg, 'att_w_validation', current_time, ckpt_time)
    
    epoch_loss /= iter_cnt
    epoch_output_mse_loss /= iter_cnt
    epoch_dec_output_mse_loss /= iter_cnt
    return epoch_loss, epoch_output_mse_loss, epoch_dec_output_mse_loss


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    set_config(cfg)
    fix_random_seed(cfg.train.random_seed)
    
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")

    video_dir, audio_dir, ckpt_path, save_path, ckpt_time= get_path_train_raw(cfg, current_time)
    train_loader, val_loader, train_dataset, val_dataset = make_train_val_loader_with_external_data_raw(cfg, video_dir, audio_dir)
    
    loss_f = MaskedLoss()
    train_loss_list = []
    train_output_mse_loss_list = []
    train_dec_output_mse_loss_list = []
    val_loss_list = []
    val_output_mse_loss_list = []
    val_dec_output_mse_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        model = make_model(cfg, device)
        
        if cfg.train.which_optim == 'adam':
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )
        elif cfg.train.which_optim == 'adamw':
            optimizer = torch.optim.AdamW(
                params=model.parameters(),
                lr=cfg.train.lr, 
                betas=(cfg.train.beta_1, cfg.train.beta_2),
                weight_decay=cfg.train.weight_decay,    
            )

        if cfg.train.which_scheduler == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=cfg.train.lr_decay_exp,
            )
        elif cfg.train.which_scheduler == 'warmup':
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=cfg.train.max_epoch,
                lr_min=cfg.train.warmup_lr_min,
                warmup_t=int(cfg.train.max_epoch * cfg.train.warmup_t_rate),
                warmup_lr_init=cfg.train.warmup_lr_init,
                warmup_prefix=True,
            )

        scaler = torch.cuda.amp.GradScaler()

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
            scaler.load_state_dict(checkpoint["scaler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]
            train_loss_list = checkpoint['train_loss_list']
            train_output_mse_loss_list = checkpoint['train_output_mse_loss_list']
            train_dec_output_mse_loss_list = checkpoint['train_dec_output_mse_loss_list']
            val_loss_list = checkpoint['val_loss_list']
            val_output_mse_loss_list = checkpoint['val_output_mse_loss_list']
            val_dec_output_mse_loss_list = checkpoint['val_dec_output_mse_loss_list']

        if cfg.train.check_point_start_separate_save_dir:
            print("load check point (separate save dir)")
            checkpoint_path = Path(cfg.train.start_ckpt_path_separate_save_dir).expanduser()
            if torch.cuda.is_available():
                checkpoint = torch.load(checkpoint_path)
            else:
                checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(checkpoint["model"])
        
        if len(cfg.train.module_is_fixed) != 0:
            print('\n--- Fix Model Parameters ---')
            count_params(model, 'model')

            for module in cfg.train.module_is_fixed:
                if module == 'avhubert':
                    set_requires_grad_by_name(model, lambda name: name.startswith('avhubert.'), requires_grad=False)
                elif module == 'avhubert_transformer':
                    set_requires_grad_by_name(model, lambda name: name.startswith('avhubert.encoder.'), requires_grad=False)
                elif module == 'avhubert_resnet':
                    set_requires_grad_by_name(model, lambda name: not name.startswith('avhubert.encoder'), requires_grad=False)

            print('--- Number of Learnable Parameters ---')
            count_params(model, 'model')
            print()
            
        wandb.watch(model, **cfg.wandb_conf.watch)
        
        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            if cfg.train.training_method == 'scheduled_sampling':
                model.decoder.scheduled_sampling_thres = 100
                # if current_epoch == 1:
                #     model.decoder.scheduled_sampling_thres = 0
                # else:
                #     model.decoder.scheduled_sampling_thres = np.clip((current_epoch - 1) ** 3 / 100, a_min=0, a_max=100)
            wandb.log({"scheduled_sampling_thres": model.decoder.scheduled_sampling_thres})

            epoch_loss, epoch_output_mse_loss, epoch_dec_output_mse_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                scaler=scaler,
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_output_mse_loss_list.append(epoch_output_mse_loss)
            train_dec_output_mse_loss_list.append(epoch_dec_output_mse_loss)

            epoch_loss, epoch_output_mse_loss, epoch_dec_output_mse_loss = val_one_epoch(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_output_mse_loss_list.append(epoch_output_mse_loss)
            val_dec_output_mse_loss_list.append(epoch_dec_output_mse_loss)

            if cfg.train.which_scheduler == 'exp':
                wandb.log({"learning_rate": scheduler.get_last_lr()[0]})
                scheduler.step()
            elif cfg.train.which_scheduler == 'warmup':
                wandb.log({"learning_rate": scheduler.optimizer.param_groups[0]['lr']})
                scheduler.step(epoch)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    train_loss_list=train_loss_list,
                    train_output_mse_loss_list=train_output_mse_loss_list,
                    train_dec_output_mse_loss_list=train_dec_output_mse_loss_list,
                    val_loss_list=val_loss_list,
                    val_output_mse_loss_list=val_output_mse_loss_list,
                    val_dec_output_mse_loss_list=val_dec_output_mse_loss_list,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{current_epoch}.ckpt")
                )
            
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_output_mse_loss_list, val_output_mse_loss_list, save_path, "output_mse_loss")
            save_loss(train_dec_output_mse_loss_list, val_dec_output_mse_loss_list, save_path, "dec_output_mse_loss")

    wandb.finish()


if __name__ == '__main__':
    main()