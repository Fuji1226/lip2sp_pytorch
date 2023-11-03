from pathlib import Path
import hydra
import torch
import sys
import torchvision
from torchvision import transforms as T
import numpy as np
from tqdm import tqdm
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

from train_nar_with_ex_avhubert_raw import make_model
from utils import load_pretrained_model


def random_crop(lip, center, cfg):
    """
    ランダムクロップ
    lip : (T, C, H, W)
    center : 中心を切り取るかどうか
    """
    if center:
        top = left = (cfg.model.imsize - cfg.model.imsize_cropped) // 2
    else:
        top = torch.randint(0, cfg.model.imsize - cfg.model.imsize_cropped, (1,))
        left = torch.randint(0, cfg.model.imsize - cfg.model.imsize_cropped, (1,))
    height = width = cfg.model.imsize_cropped
    lip = T.functional.crop(lip, top, left, height, width)
    return lip


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = make_model(cfg, device)
    # model_path = Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert/2023:10:12_19-34-33/29.ckpt').expanduser()
    # model = load_pretrained_model(model_path, model, "model")
    model.eval()
    save_dir_name = 'avhubert_feature_en'

    data_dir = Path('/home/minami/dataset/lip/avhubert_preprocess_fps25/F01_kablab')
    data_path_list = list(data_dir.glob('*.mp4'))
    lip_mean = np.array([cfg.model.avhubert_lip_mean])
    lip_std = np.array([cfg.model.avhubert_lip_std])
    lip_mean = torch.from_numpy(lip_mean)
    lip_std = torch.from_numpy(lip_std)
    lip_mean = lip_mean.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)   # (C, 1, 1, 1)
    lip_std = lip_std.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)     # (C, 1, 1, 1)

    for video_path in tqdm(data_path_list):
        lip, _, _ = torchvision.io.read_video(str(video_path), pts_unit="sec", output_format='TCHW')    # (T, C, H, W)
        lip = torchvision.transforms.functional.rgb_to_grayscale(lip)
        lip = lip.numpy()
        lip = torch.from_numpy(lip).permute(1, 2, 3, 0)     # (C, H, W, T)
        lip = lip.permute(3, 0, 1, 2)   # (T, C, H, W)
        lip = random_crop(lip, center=True, cfg=cfg)
        lip = lip.permute(1, 2, 3, 0)   # (C, H, W, T)

        lip = lip / 255.0
        lip = (lip - lip_mean) / lip_std
        lip = lip.to(torch.float32)

        lip = lip.unsqueeze(0)      # (B, C, H, W, T)
        spk_emb = torch.rand(lip.shape[0], 256).to(torch.float32)
        lip_len = torch.tensor([lip.shape[-1]])
        lip = lip.to(device)
        spk_emb = spk_emb.to(device)
        lip_len = lip_len.to(device)

        with torch.no_grad():
            output, classifier_out, avhubert_feature = model(
                lip=lip,
                audio=None,
                lip_len=lip_len,
                spk_emb=spk_emb,
            )
        avhubert_feature = avhubert_feature.cpu().numpy().squeeze()   # (T, C)
        save_path = Path(str(video_path).replace('avhubert_preprocess_fps25', f'{save_dir_name}').replace('.mp4', '.npy'))
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        np.save(save_path, avhubert_feature)



if __name__ == '__main__':
    main()