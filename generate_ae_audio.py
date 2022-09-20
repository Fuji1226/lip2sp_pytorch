import hydra

from pathlib import Path
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import random
import time
from tqdm import tqdm

import torch

from data_check import save_data
from train_ae_audio import make_model
from utils import make_test_loader, get_path_test_vc
from calc_accuracy import calc_accuracy, calc_accuracy_vc

# 現在時刻を取得
current_time = datetime.now().strftime('%Y:%m:%d_%H-%M-%S')

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)


def generate(cfg, vcnet, test_loader, dataset, device, save_path, ref_loader):
    vcnet.eval()

    lip_mean = dataset.lip_mean.to(device)
    lip_std = dataset.lip_std.to(device)
    feat_mean = dataset.feat_mean.to(device)
    feat_std = dataset.feat_std.to(device)
    feat_add_mean = dataset.feat_add_mean.to(device)
    feat_add_std = dataset.feat_add_std.to(device)

    process_times = []

    iter_cnt = 0
    for batch in tqdm(test_loader, total=len(test_loader)):
        wav, lip, feature, feat_add, upsample, data_len, speaker, label = batch
        feat_add = feat_add[:, 0, :].unsqueeze(1)
        lip, feature, feat_add, data_len = lip.to(device), feature.to(device), feat_add.to(device), data_len.to(device)

        # speaker embeddingのために使用する音響特徴量
        _, _, feature_ref, _, _, _, _, _ = ref_loader.next()
        feature_ref = feature_ref.to(device)

        # print(f"feature = {feature.shape}, feature_ref = {feature_ref.shape}")

        start_time = time.time()

        with torch.no_grad():
            output, feat_add_out, phoneme, spk_emb, enc_output, spk_class, out_upsample = vcnet(feature=feature, feature_ref=feature_ref)

        end_time = time.time()
        process_time = end_time - start_time
        process_times.append(process_time)

        _save_path = save_path / label[0]
        os.makedirs(_save_path, exist_ok=True)

        save_data(
            cfg=cfg,
            save_path=_save_path,
            wav=wav,
            lip=lip,
            feature=feature,
            feat_add=feat_add,
            output=output,
            lip_mean=lip_mean,
            lip_std=lip_std,
            feat_mean=feat_mean,
            feat_std=feat_std,
        )

        iter_cnt += 1
        if iter_cnt == 53:
            break

    return process_times


def generate_each_speaker(cfg, model_path, vcnet, device, speaker):
    print(f"generate {speaker}")
    print("reference feature : F01_kablab")
    data_root_list, mean_std_path, save_path_list = get_path_test_vc(cfg, model_path, speaker, "F01_kablab")
    ref_data_root = Path("~/dataset/lip/np_files/lip_cropped/train").expanduser()
    ref_mean_std_path = Path("~/dataset/lip/np_files/lip_cropped/mean_std").expanduser()
    cfg.test.speaker = ["F01_kablab"]
    ref_loader, ref_dataset = make_test_loader(cfg, ref_data_root, ref_mean_std_path)
    ref_loader = iter(ref_loader)

    # 同じ発話内容のものを使いたくないので適当に取り出しておく
    for _ in range(100):
        _ = ref_loader.next()

    for data_root, save_path in zip(data_root_list, save_path_list):
        cfg.test.speaker = [speaker]
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        print("--- generate ---")
        process_times = generate(
            cfg=cfg,
            vcnet=vcnet,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
            ref_loader=ref_loader,
        )

        if speaker == "F01_kablab":
            print("--- calc accuracy ---")
            calc_accuracy(save_path, save_path.parents[0], cfg, process_times)

    print("reference feature : M01_kablab")
    data_root_list, mean_std_path, save_path_list = get_path_test_vc(cfg, model_path, speaker, "M01_kablab")
    cfg.test.speaker = ["M01_kablab"]
    ref_loader, ref_dataset = make_test_loader(cfg, ref_data_root, ref_mean_std_path)
    ref_loader = iter(ref_loader)

    # 同じ発話内容のものを使いたくないので適当に取り出しておく
    for _ in range(100):
        _ = ref_loader.next()

    for data_root, save_path in zip(data_root_list, save_path_list):
        cfg.test.speaker = [speaker]
        test_loader, test_dataset = make_test_loader(cfg, data_root, mean_std_path)

        print("--- generate ---")
        process_times = generate(
            cfg=cfg,
            vcnet=vcnet,
            test_loader=test_loader,
            dataset=test_dataset,
            device=device,
            save_path=save_path,
            ref_loader=ref_loader,
        )

        if speaker == "M01_kablab":
            print("--- calc accuracy ---")
            calc_accuracy(save_path, save_path.parents[0], cfg, process_times)
        

def calc_accuracy_each_speaker(cfg, save_path_list_same, save_path_list_mix):
    for p_s, p_m in zip(save_path_list_same, save_path_list_mix):
        calc_accuracy_vc(cfg, p_s, p_m)


@hydra.main(config_name="config", config_path="conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    vcnet, lip_enc = make_model(cfg, device)
    model_path = Path("/home/usr4/r70264c/lip2sp_pytorch/check_point/ae/lip/2022:09:20_00-47-02/mspec80_140.ckpt")
    
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

    generate_each_speaker(cfg, model_path, vcnet, device, "F01_kablab")
    generate_each_speaker(cfg, model_path, vcnet, device, "M01_kablab")

    _, _, save_path_list_same_F01 = get_path_test_vc(cfg, model_path, "F01_kablab", "F01_kablab")
    _, _, save_path_list_mix_F01 = get_path_test_vc(cfg, model_path, "F01_kablab", "M01_kablab")
    _, _, save_path_list_same_M01 = get_path_test_vc(cfg, model_path, "M01_kablab", "M01_kablab")
    _, _, save_path_list_mix_M01 = get_path_test_vc(cfg, model_path, "M01_kablab", "F01_kablab")

    print("calc accuracy F01_kablab")
    calc_accuracy_each_speaker(cfg, save_path_list_same_F01, save_path_list_mix_F01)
    print("calc accuracy M01_kablab")
    calc_accuracy_each_speaker(cfg, save_path_list_same_M01, save_path_list_mix_M01)


if __name__ == "__main__":
    main()