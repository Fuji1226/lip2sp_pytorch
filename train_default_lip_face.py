from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
from pathlib import Path
import os
from datetime import datetime
import numpy as np
import random

import torch
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly

from utils import make_train_val_loader_lip_face, get_path_train, save_loss, check_mel_ss, check_mel_default, count_params, \
    set_config, mixing_prob_controller, mixing_prob_controller_f0, check_f0
from model.model_default import Lip2SP
from loss import MaskedLoss

# wandbへのログイン
wandb.login(key="ba729c3f218d8441552752401f49ba3c0c0e2b9f")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(
    model, optimizer, scheduler,
    train_output_loss_list,
    train_dec_output_loss_list,
    train_classifier_loss_list,
    val_output_loss_list,
    val_dec_output_loss_list,
    val_classifier_loss_list,
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
        "train_output_loss_list" : train_output_loss_list,
        "train_dec_output_loss_list" : train_dec_output_loss_list,
        "train_classifier_loss_list" : train_classifier_loss_list,
        "val_output_loss_list" : val_output_loss_list,
        "val_dec_output_loss_list" : val_dec_output_loss_list,
        "val_classifier_loss_list" : val_classifier_loss_list,
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    model = Lip2SP(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_inner_channels=cfg.model.res_inner_channels,
        which_res=cfg.model.which_res,
        use_lip_and_face=cfg.model.use_lip_and_face,
        trans_enc_n_layers=cfg.model.trans_enc_n_layers,
        trans_enc_n_head=cfg.model.trans_enc_n_head,
        rnn_n_layers=cfg.model.rnn_n_layers,
        rnn_which_norm=cfg.model.rnn_which_norm,
        glu_layers=cfg.model.glu_layers,
        glu_kernel_size=cfg.model.glu_kernel_size,
        trans_dec_n_layers=cfg.model.trans_dec_n_layers,
        trans_dec_n_head=cfg.model.trans_dec_n_head,
        use_f0_predicter=cfg.model.use_f0_predicter,
        f0_predicter_inner_channels=cfg.model.f0_predicter_inner_channels,
        f0_predicter_rnn_n_layers=cfg.model.f0_predicter_rnn_n_layers,
        f0_predicter_trans_enc_n_layers=cfg.model.f0_predicter_trans_enc_n_layers,
        f0_predicter_trans_enc_n_head=cfg.model.f0_predicter_trans_enc_n_head,
        f0_predicter_which_encoder=cfg.model.f0_predicter_which_encoder,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        use_spk_emb=cfg.train.use_spk_emb,
        where_spk_emb=cfg.train.where_spk_emb,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        post_n_layers=cfg.model.post_n_layers,
        post_kernel_size=cfg.model.post_kernel_size,
        n_position=int(cfg.model.input_lip_sec * cfg.model.fps),
        which_encoder=cfg.model.which_encoder,
        which_decoder=cfg.model.which_decoder,
        dec_dropout=cfg.train.dec_dropout,
        res_dropout=cfg.train.res_dropout,
        rnn_dropout=cfg.train.rnn_dropout,
        f0_predicter_dropout=cfg.train.f0_predicter_dropout,
        reduction_factor=cfg.model.reduction_factor,
    )

    count_params(model, "model")
    # multi GPU
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return model.to(device)


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, mixing_prob, ckpt_time):
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_classifier_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    model.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        wav, lip, face, feature, feat_add, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        lip = lip.to(device)
        face = face.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        output, dec_output, mixed_prev, fmaps, classifier_out, f0_pred = model(lip, lip_len, spk_emb, feature, mixing_prob, face=face)

        output_loss = loss_f.mse_loss(output, feature, feature_len, max_len=feature.shape[-1]) 
        dec_output_loss = loss_f.mse_loss(dec_output, feature, feature_len, max_len=feature.shape[-1]) 

        if cfg.train.use_spk_emb:
            classifier_loss = loss_f.cross_entropy_loss(classifier_out, speaker_idx, ignore_index=-100)
        else:
            classifier_loss = torch.tensor(0)

        loss = output_loss * cfg.train.output_loss_weight + dec_output_loss * cfg.train.dec_output_loss_weight \
            + classifier_loss * cfg.train.classifier_weight
        loss.backward()
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_output_loss += output_loss.item()
        epoch_dec_output_loss += dec_output_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        wandb.log({"train_output_loss": output_loss})
        wandb.log({"train_dec_output_loss": dec_output_loss})
        wandb.log({"train_classifier_loss": classifier_loss})

        iter_cnt += 1

        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)

                    if mixed_prev is not None:
                        check_mel_ss(feature[0], mixed_prev[0], cfg, "mel_ss_train", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)

                if mixed_prev is not None:
                    check_mel_ss(feature[0], mixed_prev[0], cfg, "mel_ss_train", current_time, ckpt_time)

    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    return epoch_output_loss, epoch_dec_output_loss, epoch_classifier_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg, mixing_prob, ckpt_time):
    epoch_output_loss = 0
    epoch_dec_output_loss = 0
    epoch_classifier_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')

        wav, lip, face, feature, feat_add, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename, label = batch
        lip = lip.to(device)
        face = face.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)
        speaker_idx = speaker_idx.to(device)
        
        with torch.no_grad():
            output, dec_output, mixed_prev, fmaps, classifier_out, f0_pred = model(lip, lip_len, spk_emb, feature, mixing_prob, face=face)

        output_loss = loss_f.mse_loss(output, feature, feature_len, max_len=feature.shape[-1]) 
        dec_output_loss = loss_f.mse_loss(dec_output, feature, feature_len, max_len=feature.shape[-1]) 

        if cfg.train.use_spk_emb:
            classifier_loss = loss_f.cross_entropy_loss(classifier_out, speaker_idx, ignore_index=-100)
        else:
            classifier_loss = torch.tensor(0)

        epoch_output_loss += output_loss.item()
        epoch_dec_output_loss += dec_output_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        wandb.log({"val_output_loss": output_loss})
        wandb.log({"val_dec_output_loss": dec_output_loss})
        wandb.log({"val_classifier_loss": classifier_loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)

                    if mixed_prev is not None:
                        check_mel_ss(feature[0], mixed_prev[0], cfg, "mel_ss_validation", current_time, ckpt_time)
                break

        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                
                if mixed_prev is not None:
                    check_mel_ss(feature[0], mixed_prev[0], cfg, "mel_ss_validation", current_time, ckpt_time)
            
    epoch_output_loss /= iter_cnt
    epoch_dec_output_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    return epoch_output_loss, epoch_dec_output_loss, epoch_classifier_loss


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
    train_data_root, val_data_root, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"train_data_root = {train_data_root}")
    print(f"val_data_root = {val_data_root}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader_lip_face(cfg, train_data_root, val_data_root)

    # 損失関数
    loss_f = MaskedLoss()

    train_output_loss_list = []
    train_dec_output_loss_list = []
    train_classifier_loss_list = []
    val_output_loss_list = []
    val_dec_output_loss_list = []
    val_classifier_loss_list = []

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
            train_output_loss_list = checkpoint["train_output_loss_list"]
            train_dec_output_loss_list = checkpoint["train_dec_output_loss_list"]
            val_output_loss_list = checkpoint["val_output_loss_list"]
            val_dec_output_loss_list = checkpoint["val_dec_output_loss_list"]
            last_epoch = checkpoint["epoch"]

        wandb.watch(model, **cfg.wandb_conf.watch)

        prob_list = mixing_prob_controller(cfg)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            
            mixing_prob = prob_list[current_epoch - 1]
            wandb.log({"mixing_prob": mixing_prob})

            print(f"mixing_prob = {mixing_prob}")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            epoch_output_loss, epoch_dec_output_loss, epoch_classifier_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                mixing_prob=mixing_prob,
                ckpt_time=ckpt_time,
            )
            train_output_loss_list.append(epoch_output_loss)
            train_dec_output_loss_list.append(epoch_dec_output_loss)
            train_classifier_loss_list.append(epoch_classifier_loss)

            # validation
            epoch_output_loss, epoch_dec_output_loss, epoch_classifier_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                mixing_prob=mixing_prob,
                ckpt_time=ckpt_time,
            )
            val_output_loss_list.append(epoch_output_loss)
            val_dec_output_loss_list.append(epoch_dec_output_loss)
            val_classifier_loss_list.append(epoch_classifier_loss)
        
            scheduler.step()

            # check point
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_output_loss_list=train_output_loss_list,
                    train_dec_output_loss_list=train_dec_output_loss_list,
                    train_classifier_loss_list=train_classifier_loss_list,
                    val_output_loss_list=val_output_loss_list,
                    val_dec_output_loss_list=val_dec_output_loss_list,
                    val_classifier_loss_list=val_classifier_loss_list,
                    epoch=current_epoch,
                    ckpt_path=os.path.join(ckpt_path, f"{cfg.model.name}_{current_epoch}.ckpt")
                )
            
            save_loss(train_output_loss_list, val_output_loss_list, save_path, "output_loss")
            save_loss(train_dec_output_loss_list, val_dec_output_loss_list, save_path, "dec_output_loss")
            save_loss(train_classifier_loss_list, val_classifier_loss_list, save_path, "classifier_loss")
                
        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()