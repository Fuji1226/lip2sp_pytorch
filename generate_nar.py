import hydra
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm
import torch

from data_check import save_data
from train_nar import make_model
from utils import (
    make_test_loader,
    get_path_test,
    load_pretrained_model,
    gen_data_separate,
    gen_data_concat,
    fix_random_seed,
    select_checkpoint,
    delete_unnecessary_checkpoint,
)
from calc_accuracy import calc_accuracy, calc_mean

current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

def generate(
    cfg,
    model,
    test_loader,
    dataset,
    device,
    save_path,
):
    model.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    begin_time=time.time()

    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, text, stop_token, spk_emb, feature_len, lip_len, text_len, speaker, speaker_idx, filename = batch
        lip = lip.to(device)
        feature = feature.to(device)
        lip_len = lip_len.to(device)
        feature_len = feature_len.to(device)
        spk_emb = spk_emb.to(device)

        lip_sep = gen_data_separate(lip, int(cfg.model.input_lip_sec * cfg.model.fps), cfg.model.fps)
        lip_len = lip_len.expand(lip_sep.shape[0])
        spk_emb = spk_emb.expand(lip_sep.shape[0], -1)

        with torch.no_grad():
            output, classifier_out, fmaps = model(lip_sep, lip_len, spk_emb)

        output = gen_data_concat(
            output, 
            int(cfg.model.fps * cfg.model.reduction_factor), 
            int((lip_len[0] % cfg.model.fps) * cfg.model.reduction_factor)
        )

        _save_path = save_path / "griffinlim" / speaker[0] / filename[0]
        _save_path.mkdir(parents=True, exist_ok=True)
        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )
    finish_time=time.time()
    dif_time=finish_time-begin_time
    print("生成時間",dif_time,"[s]")

@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    fix_random_seed(cfg.train.random_seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")
    
    model_path = select_checkpoint(cfg)
    model = make_model(cfg, device)
    model = load_pretrained_model(model_path, model, "model")
    cfg.train.face_or_lip = model_path.parents[2].name
    cfg.test.face_or_lip = model_path.parents[2].name    

    data_root_list, save_path_list, train_data_root = get_path_test(cfg, model_path)
    
    for data_root, save_path in zip(data_root_list, save_path_list):
        test_loader, test_dataset = make_test_loader(cfg, data_root, train_data_root)
        generate(
            cfg=cfg,
            model=model,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
        )

    for data_root, save_path in zip(data_root_list, save_path_list):
        for speaker in cfg.test.speaker:
            save_path_spk = save_path / "griffinlim" / speaker
            calc_accuracy(save_path_spk, save_path.parents[0], cfg, "accuracy_griffinlim")
        calc_mean(save_path.parents[0] / 'accuracy_griffinlim.txt')

    delete_unnecessary_checkpoint(
        result_dir=save_path.parents[3],
        checkpoint_dir=model_path.parents[1],
    )
        

if __name__ == "__main__":
    main()