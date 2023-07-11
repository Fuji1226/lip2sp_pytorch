from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as T
import hydra


def random_crop(cfg, lip, center):
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


@hydra.main(config_name="config", config_path="../../conf")
def main(cfg):
    data_dir = Path("~/dataset/lip/np_files/lrs2/train").expanduser()
    spk_dir_list = list(data_dir.glob("*/mspec80"))
    save_dir = Path("~/lip2sp_pytorch/data_process/lrs2/npz_check").expanduser()

    for spk_dir in tqdm(spk_dir_list):
        data_path_list = list(spk_dir.glob("*.npz"))
        for data_path in data_path_list:
            npz_key = np.load(str(data_path))
            wav = npz_key["wav"]
            feature = npz_key["feature"]
            lip = npz_key["lip"]    # (C, H, W, T)
            lip = np.transpose(lip, (3, 0, 1, 2))   # (T, C, H, W)
            lip = torch.from_numpy(lip)
            lip = random_crop(cfg, lip, center=True)
            lip = lip.numpy()
            lip = np.transpose(lip, (1, 2, 3, 0))
            lip = np.repeat(lip, 3, axis=0)
            
            speaker_name = data_path.parents[1].name
            filename = data_path.stem
            save_dir_spk = save_dir / speaker_name
            save_dir_spk.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            out = cv2.VideoWriter(str(save_dir_spk / f"{filename}.mp4"), int(fourcc), 25, (lip.shape[1], lip.shape[2]))
            
            for t in range(lip.shape[-1]):
                frame = lip[..., t]
                frame = frame.transpose(1, 2, 0)
                out.write(frame)
            
            out.release()
            break
    
    
if __name__ == "__main__":
    main()