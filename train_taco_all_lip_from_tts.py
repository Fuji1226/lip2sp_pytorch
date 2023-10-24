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
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.autograd import detect_anomaly
from synthesis import generate_for_train_check_taco_dict

from utils import  make_train_val_loader_stop_token_all, get_path_train, save_loss, check_feat_add, check_mel_default, make_test_loader, check_att, make_test_loader_dict
from model.model_trans_taco_lipread import Lip2SP
from loss import MaskedLoss

from utils import make_train_val_loader_final, make_test_loader_final

import psutil

from prob_list import *
from transformers import get_cosine_schedule_with_warmup
from data_process.phoneme_encode import IGNORE_INDEX, SOS_INDEX, EOS_INDEX
from util_from_tts import *

import gc

# wandbへのログイン
wandb.login()

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
 
def make_pad_mask_stop_token(lengths, max_len):
    device = lengths.device

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if max_len is None:
        max_len = int(max(lengths))

    seq_range = torch.arange(0, max_len, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, max_len)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    mask = mask.to(device=device)  # マスクをデバイスに送る

    return mask

def make_model(cfg, device):
    model = Lip2SP(
        in_channels=cfg.model.in_channels,
        out_channels=cfg.model.out_channels,
        res_layers=cfg.model.res_layers,
        res_inner_channels=cfg.model.res_inner_channels,
        norm_type=cfg.model.norm_type_lip,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_head=cfg.model.n_head,
        dec_n_layers=cfg.model.dec_n_layers,
        dec_d_model=cfg.model.dec_d_model,
        conformer_conv_kernel_size=cfg.model.conformer_conv_kernel_size,
        glu_inner_channels=cfg.model.glu_inner_channels,
        glu_layers=cfg.model.glu_layers,
        glu_kernel_size=cfg.model.glu_kernel_size,
        feat_add_channels=cfg.model.tc_feat_add_channels,
        feat_add_layers=cfg.model.tc_feat_add_layers,
        n_speaker=len(cfg.train.speaker),
        spk_emb_dim=cfg.model.spk_emb_dim,
        pre_inner_channels=cfg.model.pre_inner_channels,
        post_inner_channels=cfg.model.post_inner_channels,
        post_n_layers=cfg.model.post_n_layers,
        n_position=cfg.model.length * 5,
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


def train_one_epoch(model, train_loader, optimizer, loss_f, device, cfg, training_method, mixing_prob, epoch, ckpt_time, scheduler):
    
    sum_loss = {}
    sum_loss['epoch_output_loss'] = 0
    sum_loss['epoch_dec_output_loss'] = 0
    sum_loss['epoch_stop_token_loss'] = 0
    sum_loss['epoch_ctc_loss'] = 0
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(train_loader)
    print("iter start")
    optimizer.zero_grad()
    model.train()

    for batch in train_loader:
        iter_cnt += 1
        print(f'iter {iter_cnt}/{all_iter}')

        lip = batch['lip'].to(device)
        feature = batch['feature'].to(device)
        data_len = batch['data_len'].to(device)
        target_stop_token = batch['stop_tokens'].to(device)
        phoneme_index_output = batch['text'].to(device)
        text_len = batch['text_len'].to(device)
        
        # output : postnet後の出力
        # dec_output : postnet前の出力
        all_output = model(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob, use_stop_token=True)
    
        output = all_output['output']
        dec_output = all_output['dec_output']
        logit = all_output['logit']
        ctc_output = all_output['ctc_output']
        
        B, C, T = output.shape
    
        output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T) 
        dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T) 
        
        logit_mask = 1.0 - make_pad_mask_stop_token(data_len, feature.shape[-1]).to(torch.float32)
        logit_mask = logit_mask.to(torch.bool)
        logit = torch.masked_select(logit, logit_mask)
        stop_token = torch.masked_select(target_stop_token, logit_mask)
        stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)
        
        ctc_output = F.log_softmax(ctc_output, dim=-1).permute(1, 0, 2)     # (T, B, C)
        input_lens = torch.full((lip.shape[0],), fill_value=ctc_output.shape[0], dtype=torch.int)
        ctc_loss = F.ctc_loss(ctc_output, phoneme_index_output, input_lengths=input_lens, target_lengths=text_len, blank=IGNORE_INDEX)
        
        if cfg.train.gradient_accumulation_steps > 1:
            output_loss = output_loss / cfg.train.gradient_accumulation_steps
            dec_output_loss = dec_output_loss / cfg.train.gradient_accumulation_steps
            stop_token_loss = stop_token_loss / cfg.train.gradient_accumulation_steps
            ctc_loss = ctc_loss / cfg.train.gradient_accumulation_steps
            
        loss = output_loss + dec_output_loss + stop_token_loss + ctc_loss
        loss.backward()
        sum_loss['epoch_output_loss'] += output_loss.item()
        wandb.log({"train_output_loss": output_loss})
        sum_loss['epoch_dec_output_loss'] += dec_output_loss.item()
        wandb.log({"train_dec_output_loss": dec_output_loss})
        sum_loss['epoch_stop_token_loss'] += stop_token_loss.item()
        sum_loss['epoch_ctc_loss'] += ctc_loss.item()
        
        clip_grad_norm_(model.parameters(), cfg.train.max_norm)

        if iter_cnt % cfg.train.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            grad_cnt += 1

        
        if cfg.train.debug:
            if iter_cnt > cfg.train.debug_iter:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)
                break
        
       
        if iter_cnt % (all_iter - 1) == 0:
            if cfg.model.name == "mspec80":
                check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_train", current_time, ckpt_time)
        
        del lip, feature, data_len, output, dec_output
        gc.collect()
        torch.cuda.empty_cache()
        plt.clf()
        plt.close()
        
    sum_loss['epoch_output_loss'] /= grad_cnt
    sum_loss['epoch_dec_output_loss'] /= grad_cnt
    sum_loss['epoch_stop_token_loss'] /= grad_cnt
    sum_loss['epoch_ctc_loss'] /= grad_cnt
    return sum_loss


def calc_val_loss(model, val_loader, loss_f, device, cfg, training_method, mixing_prob, ckpt_time):
    sum_loss = {}
    sum_loss['epoch_output_loss'] = 0
    sum_loss['epoch_dec_output_loss'] = 0
    sum_loss['epoch_stop_token_loss'] = 0
    sum_loss['epoch_ctc_loss'] = 0
    grad_cnt = 0
    
    iter_cnt = 0
    all_iter = len(val_loader)
    print("calc val loss")
    model.eval()

    with torch.no_grad():
        for batch in val_loader:
            iter_cnt += 1
            print(f'iter {iter_cnt}/{all_iter}')

            lip = batch['lip'].to(device)
            feature = batch['feature'].to(device)
            data_len = batch['data_len'].to(device)
            target_stop_token = batch['stop_tokens'].to(device)
            phoneme_index_output = batch['text'].to(device)
            text_len = batch['text_len'].to(device)
            
                    
            all_output = model(lip=lip, prev=feature, data_len=data_len, training_method=training_method, mixing_prob=mixing_prob, use_stop_token=True)            

            output = all_output['output']
            dec_output = all_output['dec_output']
            logit = all_output['logit']
            ctc_output = all_output['ctc_output']
            
            B, C, T = output.shape
            output_loss = loss_f.mse_loss(output, feature, data_len, max_len=T)
            dec_output_loss = loss_f.mse_loss(dec_output, feature, data_len, max_len=T) 

            logit_mask = 1.0 - make_pad_mask_stop_token(data_len, feature.shape[-1]).to(torch.float32)
            logit_mask = logit_mask.to(torch.bool)
            logit = torch.masked_select(logit, logit_mask)
            stop_token = torch.masked_select(target_stop_token, logit_mask)
            stop_token_loss = F.binary_cross_entropy_with_logits(logit, stop_token)
            
            ctc_output = F.log_softmax(ctc_output, dim=-1).permute(1, 0, 2)     # (T, B, C)
            input_lens = torch.full((lip.shape[0],), fill_value=ctc_output.shape[0], dtype=torch.int)
            ctc_loss = F.ctc_loss(ctc_output, phoneme_index_output, input_lengths=input_lens, target_lengths=text_len, blank=IGNORE_INDEX)
            
            
            if cfg.train.gradient_accumulation_steps > 1:
                output_loss = output_loss / cfg.train.gradient_accumulation_steps
                dec_output_loss = dec_output_loss / cfg.train.gradient_accumulation_steps
                stop_token_loss = stop_token_loss / cfg.train.gradient_accumulation_steps
                ctc_loss = ctc_loss / cfg.train.gradient_accumulation_steps
                
            loss = output_loss + dec_output_loss + stop_token_loss + ctc_loss
            
            sum_loss['epoch_output_loss'] += output_loss.item()
            sum_loss['epoch_dec_output_loss'] += dec_output_loss.item()
            sum_loss['epoch_stop_token_loss'] += stop_token_loss.item()
            sum_loss['epoch_ctc_loss'] += ctc_loss.item()

            if cfg.train.debug:
                if iter_cnt > cfg.train.debug_iter:
                    if cfg.model.name == "mspec80":
                        check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                    break
            # if iter_cnt % cfg.train.gradient_accumulation_steps == 0:
            #     break
            if iter_cnt % (all_iter - 1) == 0:
                if cfg.model.name == "mspec80":
                    check_mel_default(feature[0], output[0], dec_output[0], cfg, "mel_validation", current_time, ckpt_time)
                    
    grad_cnt = int(iter_cnt/cfg.train.gradient_accumulation_steps)
    sum_loss['epoch_output_loss'] /= grad_cnt
    sum_loss['epoch_dec_output_loss'] /= grad_cnt
    sum_loss['epoch_stop_token_loss'] /= grad_cnt
    sum_loss['epoch_ctc_loss'] /= grad_cnt
    return sum_loss


def mixing_prob_controller(mixing_prob, epoch, mixing_prob_change_step):
    """
    mixing_prob_change_stepを超えたらmixing_probを0.01ずつ下げていく
    0.1になったら維持
    """
    if epoch >= mixing_prob_change_step:
        if mixing_prob <= 0.1:
            return mixing_prob
        else:
            mixing_prob -= 0.01
            return mixing_prob
    else:
        return mixing_prob


@hydra.main(config_name="config_all_from_tts_desk", config_path="conf")
def main(cfg):
    if cfg.train.debug:
        cfg.train.batch_size = 4
        cfg.train.num_workers = 4

    if len(cfg.train.speaker) > 1:
        cfg.train.use_gc = True
    else:
        cfg.train.use_gc = False
        
    wandb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True,
    )
    print(f'tag: {cfg.tag}')
    #breakpoint()
    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    print(f"cpu_num = {os.cpu_count()}")
    print(f"gpu_num = {torch.cuda.device_count()}")
    torch.backends.cudnn.benchmark = True

    # 現在時刻を取得
    current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')
    current_time += f'_{cfg.tag}'
    # path
    data_root, mean_std_path, ckpt_path, save_path, ckpt_time = get_path_train(cfg, current_time)
    print("\n--- data directory check ---")
    print(f"data_root = {data_root}")
    print(f"mean_std_path = {mean_std_path}")
    print(f"ckpt_path = {ckpt_path}")
    print(f"save_path = {save_path}")

    # Dataloader作成
    train_loader, val_loader, train_dataset, val_dataset =  make_train_val_loader_final(cfg, data_root, mean_std_path)
    
    test_data_root = [Path(cfg.test.face_pre_loaded_path).expanduser()]
    print(f'test root: {test_data_root}')

    test_loader, test_dataset = make_test_loader_final(cfg, test_data_root, mean_std_path)
    
    # 損失関数
    loss_f = MaskedLoss()
    train_output_loss_list = []
    train_dec_output_loss_list = []
    train_stop_token_loss_list = []
    val_output_loss_list = []
    val_dec_output_loss_list = []
    val_stop_token_loss_list = []
    
    cfg.wandb_conf.setup.name = cfg.tag
    cfg.wandb_conf.setup.name = f"{cfg.wandb_conf.setup.name}_{cfg.model.name}"
    
    with wandb.init(**cfg.wandb_conf.setup, config=wandb_cfg, settings=wandb.Settings(start_method='fork')) as run:
        # model
        model = make_model(cfg, device)
        
        if cfg.from_tts.tts_name is not None:
            name = cfg.from_tts.tts_name
            tts_path = cfg.from_tts[name]
            model = load_from_tts(model, tts_path)
            
            print(f'load ckpt: {name}')

        # optimizer
        optimizer = torch.optim.Adam(
            params=model.parameters(),
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

        #lr_scheduler = optim.lr_scheduler.StepLR(optimizer, gamma=0.5, step_size=100000)
        num_warmup_steps = int(len(train_loader)/cfg.train.gradient_accumulation_steps) * 5

        num_training_steps = int(len(train_loader)/cfg.train.gradient_accumulation_steps) * 300

        scheduler = get_cosine_schedule_with_warmup(optimizer, 
            num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
        # scheduler = CosineLRScheduler(
        #     optimizer, 
        #     t_initial=cfg.train.max_epoch, 
        #     lr_min=cfg.train.lr / 10, 
        #     warmup_t=cfg.train.warmup_t, 
        #     warmup_lr_init=cfg.train.warmup_lr_init, 
        #     warmup_prefix=True,
        # )

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

        prob_list = mixing_prob_controller_test16(cfg)


        for epoch in range(cfg.train.max_epoch - last_epoch):
            current_epoch = 1 + epoch + last_epoch
            print(f"##### {current_epoch} #####")
          
            # 学習方法の変更
            if current_epoch < cfg.train.tm_change_step:
                training_method = "tf"  # teacher forcing
            else:
                training_method = "ss"  # scheduled sampling
                
            training_method = cfg.method

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

            mixing_prob = prob_list[current_epoch-1]

            print(f"training_method : {training_method}")
            print(f"mixing prob: {mixing_prob}")
            print(f"learning_rate = {scheduler.get_last_lr()[0]}")

            # training
            train_sum_loss = train_one_epoch(
                model=model, 
                train_loader=train_loader, 
                optimizer=optimizer, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg, 
                training_method=training_method,
                mixing_prob=mixing_prob,
                epoch=current_epoch,
                ckpt_time=ckpt_time,
                scheduler=scheduler
            )
            train_output_loss_list.append(train_sum_loss['epoch_output_loss'])
            train_dec_output_loss_list.append(train_sum_loss['epoch_dec_output_loss'])
            train_stop_token_loss_list.append(train_sum_loss['epoch_stop_token_loss'])

            # validation
            val_sum_loss = calc_val_loss(
                model=model, 
                val_loader=val_loader, 
                loss_f=loss_f, 
                device=device, 
                cfg=cfg,
                training_method=training_method,
                mixing_prob=mixing_prob,
                ckpt_time=ckpt_time,
            )
            val_output_loss_list.append(val_sum_loss['epoch_output_loss'])
            val_dec_output_loss_list.append(val_sum_loss['epoch_dec_output_loss'])
            val_stop_token_loss_list.append(val_sum_loss['epoch_stop_token_loss'])
        
            #scheduler.step()

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
            save_loss(train_stop_token_loss_list, val_stop_token_loss_list, save_path, "stop_token_loss")

            # epoch_output_loss_FR_train, epoch_dec_output_loss_FR_train = generate_for_FR_train_loss(
            #     cfg = cfg,
            #     model = model,
            #     train_loader = train_loader,
            #     dataset=train_dataset,
            #     device=device,
            #     save_path=save_path,
            #     epoch=epoch,
            #     loss_f=loss_f
            # )
            # train_output_loss_list_FR.append(epoch_output_loss_FR_train)
            # train_dec_output_loss_list_FR.append(epoch_dec_output_loss_FR_train)
            # val_output_loss_list_FR.append(epoch_output_loss_FR_val)
            # val_dec_output_loss_list_FR.append(epoch_dec_output_loss_FR_val)  
            # save_loss(train_output_loss_list_FR, val_output_loss_list_FR, save_path, "FR_output_loss")
            # save_loss(train_dec_output_loss_list_FR, val_dec_output_loss_list_FR, save_path, "FR_dec_output_loss")
            
            generate_for_train_check_taco_dict(
                cfg = cfg,
                model = model,
                test_loader = test_loader,
                dataset=test_dataset,
                device=device,
                save_path=save_path,
                epoch=epoch,
                mixing_prob=mixing_prob
            )
            
            if epoch > 100:
                model_grad_ok(model)
            
            mem = psutil.virtual_memory() 
            print(f'cpu usage: {mem.percent}')
            plt.clf()
            plt.close()

        # モデルの保存
        model_save_path = save_path / f"model_{cfg.model.name}.pth"
        torch.save(model.state_dict(), str(model_save_path))
        artifact_model = wandb.Artifact('model', type='model')
        artifact_model.add_file(str(model_save_path))
        wandb.log_artifact(artifact_model)
            
    wandb.finish()


if __name__=='__main__':
    main()