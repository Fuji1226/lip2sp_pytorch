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
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from timm.scheduler import CosineLRScheduler

from utils import make_train_val_loader, save_loss, get_path_train, check_feat_add
from train_vq_audio import make_model
from loss import MaskedLoss
from model.nar_decoder import FeadAddPredicter
from model.classifier import SpeakerClassifier

# wandbへのログイン
wandb.login(key="090cd032aea4c94dd3375f1dc7823acc30e6abef")

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(777)
torch.manual_seed(777)
torch.cuda.manual_seed_all(777)
random.seed(777)


def save_checkpoint(classifier, feat_add_predicter, optimizer_classifier, optimizer_feat_add, epoch, ckpt_path):
    torch.save({
        'classifier': classifier.state_dict(),
        'feat_add_predicter': feat_add_predicter.state_dict(),
        'optimizer_classifier': optimizer_classifier.state_dict(),
        'optimizer_feat_add': optimizer_feat_add.state_dict(),
        "random": random.getstate(),
        "np_random": np.random.get_state(), 
        "torch": torch.get_rng_state(),
        "torch_random": torch.random.get_rng_state(),
        'cuda_random' : torch.cuda.get_rng_state(),
        'epoch': epoch
    }, ckpt_path)


def check_classifier_accuracy(target, pred, cfg, filename, epoch, current_time, ckpt_time):
    pred = torch.argmax(pred, dim=1)

    save_path = Path("~/lip2sp_pytorch/data_check").expanduser()
    if ckpt_time is not None:
        save_path = save_path / cfg.train.name / ckpt_time
    else:
        save_path = save_path / cfg.train.name / current_time
    os.makedirs(save_path, exist_ok=True)

    acc = torch.sum(target == pred) / target.shape[0]

    with open(str(save_path / f"{filename}.txt"), "a") as f:
        f.write(f"\n--- epoch {epoch} ---\n")
        f.write("answer\n")
        f.write(f"{target}\n")
        f.write("\npredict\n")
        f.write(f"{pred}\n")
        f.write(f"accuracy = {acc}\n")


def train_one_epoch(vcnet, feat_add_predicter, classifier, train_loader, optimizer_feat_add, optimizer_classifier, loss_f, cfg, device, epoch, ckpt_time):
    epoch_loss_classifier = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start") 
    vcnet.eval()
    feat_add_predicter.train()
    classifier.train()

    for batch in train_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            output, _, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)

        speaker_pred = classifier(quantize)
        feat_add_pred = feat_add_predicter(out_upsample)

        loss_classifier = F.cross_entropy(speaker_pred, speaker)
        loss_classifier.backward()
        clip_grad_norm_(classifier.parameters(), cfg.train.max_norm)
        optimizer_classifier.step()
        optimizer_classifier.zero_grad()
        epoch_loss_classifier += loss_classifier.item()
        wandb.log({"train_classifier_loss": loss_classifier})

        loss_feat_add = loss_f.mse_loss(feat_add_pred, feat_add, data_len, max_len=feat_add.shape[-1])
        loss_feat_add.backward()
        clip_grad_norm_(feat_add_predicter.parameters(), cfg.train.max_norm)
        optimizer_feat_add.step()
        optimizer_feat_add.zero_grad()
        epoch_loss_feat_add += loss_feat_add.item()
        wandb.log({"train_feat_add_predicter_loss": loss_feat_add})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_feat_add(feat_add[0], feat_add_pred[0], cfg, "feat_add_train", current_time, ckpt_time)
                check_classifier_accuracy(speaker, speaker_pred, cfg, "speaker_train", epoch, current_time, ckpt_time)
                break
                
        if iter_cnt % (all_iter - 1) == 0:
            check_feat_add(feat_add[0], feat_add_pred[0], cfg, "feat_add_train", current_time, ckpt_time)
            check_classifier_accuracy(speaker, speaker_pred, cfg, "speaker_train", epoch, current_time, ckpt_time)

    epoch_loss_classifier /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss_classifier, epoch_loss_feat_add


def val_one_epoch(vcnet, feat_add_predicter, classifier, val_loader, loss_f, cfg, device, epoch, ckpt_time):
    epoch_loss_classifier = 0
    epoch_loss_feat_add = 0
    iter_cnt = 0
    all_iter = len(val_loader)
    print("iter start") 
    vcnet.eval()
    feat_add_predicter.eval()
    classifier.eval()

    for batch in val_loader:
        print(f'iter {iter_cnt}/{all_iter}')
        lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len, speaker = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device), speaker.to(device)

        with torch.no_grad():
            output, _, phoneme, spk_emb, quantize, embed_idx, vq_loss, enc_output, idx_pred, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature, data_len=data_len)
            speaker_pred = classifier(quantize)
            feat_add_pred = feat_add_predicter(out_upsample)

        loss_classifier = F.cross_entropy(speaker_pred, speaker)
        loss_feat_add = loss_f.mse_loss(feat_add_pred, feat_add, data_len, max_len=feat_add.shape[-1])

        epoch_loss_classifier += loss_classifier.item()
        epoch_loss_feat_add += loss_feat_add.item()
        wandb.log({"train_classifier_loss": loss_classifier})
        wandb.log({"train_feat_add_predicter_loss": loss_feat_add})

        iter_cnt += 1
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                check_feat_add(feat_add[0], feat_add_pred[0], cfg, "feat_add_val", current_time, ckpt_time)
                check_classifier_accuracy(speaker, speaker_pred, cfg, "speaker_val", epoch, current_time, ckpt_time)
                break
                
        if iter_cnt % (all_iter - 1) == 0:
            check_feat_add(feat_add[0], feat_add_pred[0], cfg, "feat_add_val", current_time, ckpt_time)
            check_classifier_accuracy(speaker, speaker_pred, cfg, "speaker_val", epoch, current_time, ckpt_time)

    epoch_loss_classifier /= iter_cnt
    epoch_loss_feat_add /= iter_cnt
    return epoch_loss_classifier, epoch_loss_feat_add


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

    loss_f = MaskedLoss()
    train_classifier_loss_list = []
    train_feat_add_loss_list = []
    val_classifier_loss_list = []
    val_feat_add_loss_list = []

    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}_check"
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        vcnet, lip_enc = make_model(cfg, device)
        model_path = Path(cfg.train.model_path).expanduser()

        if model_path.suffix == ".ckpt":
            try:
                vcnet.load_state_dict(torch.load(str(model_path))['vcnet'])
            except:
                vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu'))['vcnet'])
        elif model_path.suffix == ".pth":
            try:
                vcnet.load_state_dict(torch.load(str(model_path)))
            except:
                vcnet.load_state_dict(torch.load(str(model_path), map_location=torch.device('cpu')))

        feat_add_predicter = FeadAddPredicter(
            in_channels=cfg.model.tc_inner_channels, 
            out_channels=cfg.model.tc_feat_add_channels, 
            kernel_size=3, 
            n_layers=cfg.model.tc_feat_add_layers, 
            dropout=cfg.train.dec_dropout,
        ).to(device)
        classifier = SpeakerClassifier(cfg.model.vq_emb_dim, 512, n_layers=2, bidirectional=True, n_speaker=len(cfg.train.speaker)).to(device)

        optimizer_feat_add = torch.optim.Adam(
            params=feat_add_predicter.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )
        optimizer_classifier = torch.optim.Adam(
            params=classifier.parameters(),
            lr=cfg.train.lr, 
            betas=(cfg.train.beta_1, cfg.train.beta_2),
            weight_decay=cfg.train.weight_decay,    
        )

        last_epoch = 0
        if cfg.train.check_point_start:
            checkpoint_path = Path(cfg.train.start_ckpt_path).expanduser()
            checkpoint = torch.load(checkpoint_path)
            classifier.load_state_dict(checkpoint["classifier"])
            feat_add_predicter.load_state_dict(checkpoint["feat_add_predicter"])
            optimizer_classifier.load_state_dict(checkpoint["optimizer_classifier"])
            optimizer_feat_add.load_state_dict(checkpoint["optimizer_feat_add"])
            random.setstate(checkpoint["random"])
            np.random.set_state(checkpoint["np_random"])
            torch.set_rng_state(checkpoint["torch"])
            torch.random.set_rng_state(checkpoint["torch_random"])
            torch.cuda.set_rng_state(checkpoint["cuda_random"])
            last_epoch = checkpoint["epoch"]

        wandb.watch(feat_add_predicter, **cfg.wandb_conf.watch)
        wandb.watch(classifier, **cfg.wandb_conf.watch)

        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")

            epoch_loss_classifier, epoch_loss_feat_add = train_one_epoch(
                vcnet=vcnet,
                feat_add_predicter=feat_add_predicter,
                classifier=classifier,
                train_loader=train_loader,
                optimizer_feat_add=optimizer_feat_add,
                optimizer_classifier=optimizer_classifier,
                loss_f=loss_f,
                cfg=cfg,
                device=device,
                epoch=current_epoch,
                ckpt_time=ckpt_time,
            )
            train_classifier_loss_list.append(epoch_loss_classifier)
            train_feat_add_loss_list.append(epoch_loss_feat_add)

            epoch_loss_classifier, epoch_loss_feat_add = val_one_epoch(
                vcnet=vcnet,
                feat_add_predicter=feat_add_predicter,
                classifier=classifier,
                val_loader=val_loader,
                loss_f=loss_f,
                cfg=cfg,
                device=device,
                epoch=current_epoch,
                ckpt_time=ckpt_time,
            )
            val_classifier_loss_list.append(epoch_loss_classifier)
            val_feat_add_loss_list.append(epoch_loss_feat_add)

            if current_epoch % cfg.train.ckpt_step == 0:
                save_checkpoint(
                    classifier=classifier,
                    feat_add_predicter=feat_add_predicter,
                    optimizer_classifier=optimizer_classifier,
                    optimizer_feat_add=optimizer_feat_add,
                    epoch=current_epoch,
                    ckpt_path=str(ckpt_path / f"{cfg.model.name}_{current_epoch}.ckpt"),
                )
            
            save_loss(train_classifier_loss_list, val_classifier_loss_list, save_path, "classifier_loss")
            save_loss(train_feat_add_loss_list, val_feat_add_loss_list, save_path, "feat_add_predicter_loss")

        classifier_save_path = save_path / f"classifier_{cfg.model.name}.pth"
        torch.save(classifier.state_dict(), str(classifier_save_path))
        feat_add_predicter_save_path = save_path / f"feat_add_predicter_{cfg.model.name}.pth"
        torch.save(feat_add_predicter.state_dict(), str(feat_add_predicter_save_path))


if __name__ == "__main__":
    main()