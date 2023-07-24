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
import librosa
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from model.model_ae import VoiceConversionNetAE, LipEncoder
from loss import MaskedLoss
from model.mi_estimater import MyCLUBSample
from model.nar_decoder import FeadAddPredicter
from utils import make_train_val_loader, make_test_loader, check_mel_nar, check_feat_add, get_path_train, save_loss

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(vcnet, mi_estimater, feat_add_predicter, optimizer, optimizer_mi, scheduler, epoch, ckpt_path):
# def save_checkpoint(vcnet, mi_estimater, feat_add_predicter, optimizer, optimizer_mi, epoch, ckpt_path):
	torch.save({
        'vcnet': vcnet.state_dict(),
        'mi_estimater': mi_estimater.state_dict(),
        "feat_add_predicter" : feat_add_predicter.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer_mi': optimizer_mi.state_dict(),
        'scheduler': scheduler.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def make_model(cfg, device):
    vcnet = VoiceConversionNetAE(
        out_channels=cfg.model.out_channels,
        tc_d_model=cfg.model.tc_d_model,
        tc_n_attn_layer=cfg.model.tc_n_attn_layer,
        tc_n_head=cfg.model.tc_n_head,
        dec_n_layers=cfg.model.tc_n_layers,
        dec_inner_channels=cfg.model.tc_inner_channels,
        dec_kernel_size=cfg.model.tc_kernel_size,
        feat_add_channels=cfg.model.tc_feat_add_channels,
        feat_add_layers=cfg.model.tc_feat_add_layers,
        ae_emb_dim=cfg.model.ae_emb_dim,
        spk_emb_dim=cfg.model.spk_emb_dim,
        n_speaker=len(cfg.train.speaker),
        norm_type_audio=cfg.model.norm_type_audio,
        content_d_model=cfg.model.content_d_model,
        content_n_attn_layer=cfg.model.content_n_attn_layer,
        content_n_head=cfg.model.content_n_head,
        which_spk_enc=cfg.model.which_spk_enc,
        use_feat_add=cfg.train.use_feat_add,
        phoneme_classes=cfg.model.n_classes,
        use_phoneme=cfg.train.use_phoneme,
        use_dec_attention=cfg.train.use_dec_attention,
        upsample_method=cfg.train.upsample_method,
        compress_rate=cfg.train.compress_rate,
        dec_dropout=cfg.train.dec_dropout,
        reduction_factor=cfg.model.reduction_factor,
    )
    lip_enc = LipEncoder(
        in_channels=cfg.model.in_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        ae_emb_dim=cfg.model.ae_emb_dim,
        apply_first_bn=cfg.train.apply_first_bn,
        compress_rate=cfg.train.compress_rate,
        which_encoder=cfg.model.which_encoder,
        res_dropout=cfg.train.res_dropout,
        reduction_factor=cfg.model.reduction_factor,
        norm_type_lip=cfg.model.norm_type_lip,
        which_res=cfg.model.which_res,
        separate_frontend=cfg.train.separate_frontend,
    )
    params = 0
    for p in vcnet.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"vcnet parameter = {params}")
    
    params = 0
    for p in lip_enc.parameters():
        if p.requires_grad:
            params += p.numel()
    print(f"lip_enc parameter = {params}")

    # multi GPU
    if torch.cuda.device_count() > 1:
        vcnet = torch.nn.DataParallel(vcnet)
        lip_enc = torch.nn.DataParallel(lip_enc)
        print(f"\nusing {torch.cuda.device_count()} GPU")
    return vcnet.to(device), lip_enc.to(device)


def train_one_epoch(vcnet, mi_estimater, feat_add_predicter, train_loader, optimizer, optimizer_mi, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    epoch_mse_loss = 0
    epoch_mi_loss = 0
    epoch_classifier_loss = 0
    epoch_estimater_loss = 0
    epoch_feat_add_predicter_loss = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    vcnet.train()
    mi_estimater.train()
    feat_add_predicter.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        ### update mi estimater ###
        with torch.no_grad():
            output, _, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)
        
        estimater_loss = - mi_estimater.loglikeli(spk_emb, enc_output)
        estimater_loss.backward()
        clip_grad_norm_(mi_estimater.parameters(), cfg.train.max_norm)
        optimizer_mi.step()
        optimizer_mi.zero_grad()
        epoch_estimater_loss += estimater_loss.item()
        wandb.log({"train_estimater_loss": estimater_loss})

        ### update feat_add_predicter ###
        feat_add_out = feat_add_predicter(out_upsample)
        loss_feat_add_predicter = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=feat_add.shape[-1])
        loss_feat_add_predicter.backward()
        clip_grad_norm_(feat_add_predicter.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()
        epoch_feat_add_predicter_loss += loss_feat_add_predicter.item()
        wandb.log({"train_feat_add_predicter_loss": loss_feat_add_predicter})

        ### update vcnet ###
        output, _, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)

        with torch.no_grad():
            mi_loss = mi_estimater(spk_emb, enc_output)
            feat_add_out = feat_add_predicter(out_upsample)
            
        loss_feat_add = loss_f.mse_loss(feat_add_out, torch.zeros_like(feat_add_out), data_len, max_len=feat_add.shape[-1])
        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=feature.shape[-1])
        classifier_loss = F.cross_entropy(spk_class, speaker)
        
        loss = mse_loss + cfg.train.mi_weight * mi_loss + cfg.train.classifier_weight * classifier_loss + cfg.train.feat_add_weight * loss_feat_add
        loss.backward()
        clip_grad_norm_(vcnet.parameters(), cfg.train.max_norm)
        optimizer.step()
        optimizer.zero_grad()

        epoch_mse_loss += mse_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        epoch_loss_feat_add += loss_feat_add.item()
        epoch_loss += loss.item()
        wandb.log({"train_mse_loss": mse_loss})
        wandb.log({"train_mi_loss": mi_loss})
        wandb.log({"train_classifier_loss": classifier_loss})
        wandb.log({"train_loss_feat_add": loss_feat_add})
        wandb.log({"train_total_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                    check_feat_add(feat_add[0], feat_add_out[0], cfg, "feat_add_train", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_train", current_time, ckpt_time)
                check_feat_add(feat_add[0], feat_add_out[0], cfg, "feat_add_train", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    epoch_feat_add_predicter_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss, epoch_feat_add_predicter_loss


def val_one_epoch(vcnet, mi_estimater, feat_add_predicter, val_loader, loss_f, device, cfg, ckpt_time):
    epoch_loss = 0
    epoch_loss_feat_add = 0
    epoch_mse_loss = 0
    epoch_mi_loss = 0
    epoch_classifier_loss = 0
    epoch_estimater_loss = 0
    epoch_feat_add_predicter_loss = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    vcnet.eval()
    mi_estimater.eval()
    feat_add_predicter.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        feature, feat_add, data_len, speaker = feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        rand_index = torch.randperm(feature.shape[0])
        feature_ref = feature[rand_index]

        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)
            estimater_loss = - mi_estimater.loglikeli(spk_emb, enc_output)
            mi_loss = mi_estimater(spk_emb, enc_output)
            feat_add_out = feat_add_predicter(out_upsample)

        loss_feat_add_predicter = loss_f.mse_loss(feat_add_out, feat_add, data_len, max_len=feat_add.shape[-1])
        mse_loss = loss_f.mse_loss(output, feature, data_len, max_len=feature.shape[-1])
        classifier_loss = F.cross_entropy(spk_class, speaker)
        loss_feat_add = loss_f.mse_loss(feat_add_out, torch.zeros_like(feat_add_out), data_len, max_len=feat_add.shape[-1])
        loss = mse_loss + cfg.train.mi_weight * mi_loss + cfg.train.classifier_weight * classifier_loss + cfg.train.feat_add_weight * loss_feat_add

        epoch_estimater_loss += estimater_loss.item()
        epoch_feat_add_predicter_loss += loss_feat_add_predicter.item()
        epoch_mse_loss += mse_loss.item()
        epoch_mi_loss += mi_loss.item()
        epoch_classifier_loss += classifier_loss.item()
        epoch_loss_feat_add += loss_feat_add.item()
        epoch_loss += loss.item()
        wandb.log({"val_estimater_loss": estimater_loss})   
        wandb.log({"val_feat_add_predicter_loss": loss_feat_add_predicter})
        wandb.log({"val_mse_loss": mse_loss})
        wandb.log({"val_mi_loss": mi_loss})  
        wandb.log({"val_classifier_loss": classifier_loss})
        wandb.log({"val_loss_feat_add": loss_feat_add})
        wandb.log({"val_total_loss": loss})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_nar(feature[0], output[0], cfg, "mel_val", current_time, ckpt_time)
                    check_feat_add(feat_add[0], feat_add_out[0], cfg, "feat_add_val", current_time, ckpt_time)
                break
        
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_nar(feature[0], output[0], cfg, "mel_val", current_time, ckpt_time)
                check_feat_add(feat_add[0], feat_add_out[0], cfg, "feat_add_val", current_time, ckpt_time)

    epoch_loss /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    epoch_mse_loss /= iter_cnt
    epoch_mi_loss /= iter_cnt
    epoch_classifier_loss /= iter_cnt
    epoch_estimater_loss /= iter_cnt
    epoch_feat_add_predicter_loss /= iter_cnt
    return epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss, epoch_feat_add_predicter_loss


def calc_dem(vcnet, loader_F01, loader_M01, cfg, device):
    spk_emb_f_list = []
    spk_emb_m_list = []
    enc_output_f_list = []
    enc_output_m_list = []

    for _ in range(len(loader_F01)):
        wav_f, lip_f, feature_f, feat_add_f, upsample_f, data_len_f, speaker_f, label_f = loader_F01.next()
        wav_m, lip_m, feature_m, feat_add_m, upsample_m, data_len_m, speaker_m, label_m = loader_M01.next()
        feature_f, feature_m = feature_f.to(device), feature_m.to(device)
        
        with torch.no_grad():
            output_f, feat_add_out_f, phoneme_f, spk_emb_f, enc_output_f, spk_class_f, out_upsample_f = vcnet(feature=feature_f, feature_ref=feature_m)
            output_m, feat_add_out_m, phoneme_m, spk_emb_m, enc_output_m, spk_class_m, out_upsample_m = vcnet(feature=feature_m, feature_ref=feature_f)

        spk_emb_f = spk_emb_f[0].to('cpu').detach().numpy().copy()    # (C,)
        spk_emb_m = spk_emb_m[0].to('cpu').detach().numpy().copy()    # (C,)
        enc_output_f = enc_output_f[0].to('cpu').detach().numpy().copy()    # (T, C)
        enc_output_m = enc_output_m[0].to('cpu').detach().numpy().copy()    # (T, C)

        spk_emb_f_list.append(spk_emb_f)
        spk_emb_m_list.append(spk_emb_m)
        enc_output_f_list.append(enc_output_f)
        enc_output_m_list.append(enc_output_m)

    spk_emb_f_list_shuffle = random.sample(spk_emb_f_list, len(spk_emb_f_list))
    spk_emb_m_list_shuffle = random.sample(spk_emb_m_list, len(spk_emb_m_list))
    enc_output_f_list_shuffle = random.sample(enc_output_f_list, len(enc_output_f_list))
    enc_output_m_list_shuffle = random.sample(enc_output_m_list, len(enc_output_m_list))

    # 異なる話者で同じ発話内容の場合
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb_f, emb_m, enc_f, enc_m in zip(spk_emb_f_list, spk_emb_m_list, enc_output_f_list, enc_output_m_list):
        # spk_emb
        cos_sim_emb = np.dot(emb_f, emb_m) / (np.linalg.norm(emb_f) * np.linalg.norm(emb_m))
        cos_sim_emb_list.append(cos_sim_emb)

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc_f.T, enc_m.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            f_frame = enc_f[wp[i, 0]]
            m_frame = enc_m[wp[i, 1]]
            cos_sim_enc = np.dot(f_frame, m_frame) / (np.linalg.norm(f_frame) * np.linalg.norm(m_frame))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)

    dem_emb_both = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_both = sum(cos_sim_enc_list) / len(cos_sim_enc_list)

    # 同じ話者で異なる発話内容の場合
    # F01_kablab
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb, emb_shuffle, enc, enc_shuffle in zip(spk_emb_f_list, spk_emb_f_list_shuffle, enc_output_f_list, enc_output_f_list_shuffle):
        # spk_emb
        cos_sim_emb = np.dot(emb, emb_shuffle) / (np.linalg.norm(emb) * np.linalg.norm(emb_shuffle))
        cos_sim_emb_list.append(cos_sim_emb)        

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc.T, enc_shuffle.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            frame = enc[wp[i, 0]]
            frame_shuffle = enc_shuffle[wp[i, 1]]
            cos_sim_enc = np.dot(frame, frame_shuffle) / (np.linalg.norm(frame) * np.linalg.norm(frame_shuffle))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)
    
    dem_emb_F01 = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_F01 = sum(cos_sim_enc_list) / len(cos_sim_enc_list)

    # M01_kablab
    cos_sim_emb_list = []
    cos_sim_enc_list = []
    for emb, emb_shuffle, enc, enc_shuffle in zip(spk_emb_m_list, spk_emb_m_list_shuffle, enc_output_m_list, enc_output_m_list_shuffle):
        # spk_emb
        cos_sim_emb = np.dot(emb, emb_shuffle) / (np.linalg.norm(emb) * np.linalg.norm(emb_shuffle))
        cos_sim_emb_list.append(cos_sim_emb)        

        # enc_output
        min_cost, wp = librosa.sequence.dtw(enc.T, enc_shuffle.T)
        cos_sim_enc_frame = []
        for i in range(wp.shape[0]):
            frame = enc[wp[i, 0]]
            frame_shuffle = enc_shuffle[wp[i, 1]]
            cos_sim_enc = np.dot(frame, frame_shuffle) / (np.linalg.norm(frame) * np.linalg.norm(frame_shuffle))
            cos_sim_enc_frame.append(cos_sim_enc)
        cos_sim_enc = sum(cos_sim_enc_frame) / len(cos_sim_enc_frame)
        cos_sim_enc_list.append(cos_sim_enc)
    
    dem_emb_M01 = sum(cos_sim_emb_list) / len(cos_sim_emb_list)
    dem_enc_M01 = sum(cos_sim_enc_list) / len(cos_sim_enc_list)
    return dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01


def calc_dem_during_train(cfg, vcnet, device):
    print("calc dem")
    test_data_root = Path(cfg.test.lip_pre_loaded_path).expanduser()
    mean_std_path = Path(cfg.train.lip_mean_std_path).expanduser()
    cfg.test.speaker = ["F01_kablab"]
    test_loader_F01, test_dataset_F01 = make_test_loader(cfg, test_data_root, mean_std_path)
    test_loader_F01 = iter(test_loader_F01)

    cfg.test.speaker = ["M01_kablab"]
    test_loader_M01, test_dataset_M01 = make_test_loader(cfg, test_data_root, mean_std_path)
    test_loader_M01 = iter(test_loader_M01)
    dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01 = calc_dem(vcnet, test_loader_F01, test_loader_M01, cfg, device)
    wandb.log({"dem_emb_both": dem_emb_both})
    wandb.log({"dem_enc_both": dem_enc_both})
    wandb.log({"dem_emb_F01": dem_emb_F01})
    wandb.log({"dem_enc_F01": dem_enc_F01})
    wandb.log({"dem_emb_M01": dem_emb_M01})
    wandb.log({"dem_enc_M01": dem_enc_M01})
    return dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01


def save_dem(dem_emb_both_list, dem_enc_both_list, dem_emb_F01_list, dem_enc_F01_list, dem_emb_M01_list, dem_enc_M01_list, save_path, filename_emb, filename_enc):
    dem_emb_save_path = save_path / f"{filename_emb}.png"
    plt.figure()
    plt.plot(np.arange(len(dem_emb_both_list)), dem_emb_both_list)
    plt.plot(np.arange(len(dem_emb_F01_list)), dem_emb_F01_list)
    plt.plot(np.arange(len(dem_emb_M01_list)), dem_emb_M01_list)
    plt.xlabel("epoch")
    plt.ylabel("cosine similarity")
    plt.legend(["both", "F01", "M01"])
    plt.grid()
    plt.savefig(str(dem_emb_save_path))
    plt.close()

    dem_enc_save_path = save_path / f"{filename_enc}.png"
    plt.figure()
    plt.plot(np.arange(len(dem_enc_both_list)), dem_enc_both_list)
    plt.plot(np.arange(len(dem_enc_F01_list)), dem_enc_F01_list)
    plt.plot(np.arange(len(dem_enc_M01_list)), dem_enc_M01_list)
    plt.xlabel("epoch")
    plt.ylabel("cosine similarity")
    plt.legend(["both", "F01", "M01"])
    plt.grid()
    plt.savefig(str(dem_enc_save_path))
    plt.close()

    wandb.log({f"loss {filename_emb}": wandb.plot.line_series(
        xs=np.arange(len(dem_emb_both_list)), 
        ys=[dem_emb_both_list, dem_emb_F01_list, dem_emb_M01_list],
        keys=["both", "F01", "M01"],
        title=f"{filename_emb}",
        xname="epoch",
    )})
    wandb.log({f"loss {filename_enc}": wandb.plot.line_series(
        xs=np.arange(len(dem_emb_both_list)), 
        ys=[dem_enc_both_list, dem_enc_F01_list, dem_enc_M01_list],
        keys=["both", "F01", "M01"],
        title=f"{filename_enc}",
        xname="epoch",
    )})


@hydra.main(version_base=None, config_name="config", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 4

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
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, _, _ = make_train_val_loader(cfg, data_root, mean_std_path)

    # 損失関数
    loss_f = MaskedLoss()
    train_loss_list = []
    train_feat_add_loss_list = []
    train_mse_loss_list = []
    train_mi_loss_list = []
    train_estimater_loss_list = []
    train_classifier_loss_list = []
    train_feat_add_predicter_loss_list = []
    val_loss_list = []
    val_feat_add_loss_list = []
    val_mse_loss_list = []
    val_mi_loss_list = []
    val_estimater_loss_list = []
    val_classifier_loss_list = []
    val_feat_add_predicter_loss_list = []

    dem_emb_both_list = []
    dem_enc_both_list = []
    dem_emb_F01_list = []
    dem_enc_F01_list = []
    dem_emb_M01_list = []
    dem_enc_M01_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_audio"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        vcnet, lip_enc = make_model(cfg, device)
        mi_estimater = MyCLUBSample(
            x_dim=cfg.model.spk_emb_dim, 
            y_dim=cfg.model.ae_emb_dim,
            hidden_size=cfg.model.mi_hidden_channels
        ).to(device)
        feat_add_predicter = FeadAddPredicter(
            in_channels=cfg.model.tc_inner_channels, 
            out_channels=cfg.model.tc_feat_add_channels, 
            kernel_size=3, 
            n_layers=cfg.model.tc_feat_add_layers, 
            dropout=cfg.train.dec_dropout,
        ).to(device)
        
        # optimizer
        optimizer = torch.optim.Adam(
            params=list(vcnet.parameters()) + list(feat_add_predicter.parameters()),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_mi = torch.optim.Adam(
            params=mi_estimater.parameters(),
            lr=cfg.train.lr_mi,
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay_mi,
        )
        
        # scheduler
        scheduler = CosineLRScheduler(
            optimizer, 
            t_initial=cfg.train.max_epoch, 
            lr_min=cfg.train.warmup_lr_min, 
            warmup_t=cfg.train.warmup_t, 
            warmup_lr_init=cfg.train.warmup_lr_init, 
            warmup_prefix=True,
        )

        last_epoch = 0

        if cfg.train.check_point_start:
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            checkpoint = torch.load(checkpoint_path)
            vcnet.load_state_dict(checkpoint["vcnet"])
            mi_estimater.load_state_dict(checkpoint["mi_estimater"])
            feat_add_predicter.load_state_dict(checkpoint["feat_add_predicter"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            optimizer_mi.load_state_dict(checkpoint["optimizer_mi"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(vcnet, **cfg.wandb_conf.watch)
        wandb.watch(mi_estimater, **cfg.wandb_conf.watch)
        wandb.watch(feat_add_predicter, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
            # print(f"learning_rate = {scheduler.get_epoch_values(current_epoch)}")

            # train
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss, epoch_feat_add_predicter_loss = train_one_epoch(
                vcnet=vcnet,
                mi_estimater=mi_estimater,
                feat_add_predicter=feat_add_predicter,
                train_loader=train_loader,
                optimizer=optimizer,
                optimizer_mi=optimizer_mi,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            train_loss_list.append(epoch_loss)
            train_feat_add_loss_list.append(epoch_loss_feat_add)
            train_mse_loss_list.append(epoch_mse_loss)
            train_mi_loss_list.append(epoch_mi_loss)
            train_estimater_loss_list.append(epoch_estimater_loss)
            train_classifier_loss_list.append(epoch_classifier_loss)
            train_feat_add_predicter_loss_list.append(epoch_feat_add_predicter_loss)

            # validation
            epoch_loss, epoch_loss_feat_add, epoch_mse_loss, epoch_mi_loss, epoch_estimater_loss, epoch_classifier_loss, epoch_feat_add_predicter_loss = val_one_epoch(
                vcnet=vcnet,
                mi_estimater=mi_estimater,
                feat_add_predicter=feat_add_predicter,
                val_loader=val_loader,
                loss_f=loss_f,
                device=device,
                cfg=cfg,
                ckpt_time=ckpt_time,
            )
            val_loss_list.append(epoch_loss)
            val_feat_add_loss_list.append(epoch_loss_feat_add)
            val_mse_loss_list.append(epoch_mse_loss)
            val_mi_loss_list.append(epoch_mi_loss)
            val_estimater_loss_list.append(epoch_estimater_loss)
            val_classifier_loss_list.append(epoch_classifier_loss)
            val_feat_add_predicter_loss_list.append(epoch_feat_add_predicter_loss)

            scheduler.step(current_epoch)

            # checkpoint
            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    vcnet=vcnet,
                    mi_estimater=mi_estimater,
                    feat_add_predicter=feat_add_predicter,
                    optimizer=optimizer,
                    optimizer_mi=optimizer_mi,
                    scheduler=scheduler,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )

            # dem
            dem_emb_both, dem_enc_both, dem_emb_F01, dem_enc_F01, dem_emb_M01, dem_enc_M01 = calc_dem_during_train(cfg, vcnet, device)
            dem_emb_both_list.append(dem_emb_both)
            dem_enc_both_list.append(dem_enc_both)
            dem_emb_F01_list.append(dem_emb_F01)
            dem_enc_F01_list.append(dem_enc_F01)
            dem_emb_M01_list.append(dem_emb_M01)
            dem_enc_M01_list.append(dem_enc_M01)
            save_dem(dem_emb_both_list, dem_enc_both_list, dem_emb_F01_list, dem_enc_F01_list, dem_emb_M01_list, dem_enc_M01_list, save_path, "dem_emb", "dem_enc")

            # save loss
            save_loss(train_loss_list, val_loss_list, save_path, "loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "loss_feat_add")
            save_loss(train_mse_loss_list, val_mse_loss_list, save_path, "mse_loss")
            save_loss(train_mi_loss_list, val_mi_loss_list, save_path, "mi_loss")
            save_loss(train_estimater_loss_list, val_estimater_loss_list, save_path, "estimater_loss")
            save_loss(train_classifier_loss_list, val_classifier_loss_list, save_path, "classifier_loss")
            save_loss(train_feat_add_predicter_loss_list, val_feat_add_predicter_loss_list, save_path, "feat_add_predicter_loss")

        # save model parameter
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(vcnet.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)


if __name__ == "__main__":
    main()