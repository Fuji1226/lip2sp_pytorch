import os
import sys
from pathlib import Path
sys.path.append(str(Path("~/lip2sp_pytorch_all/lip2sp_920_re/data_process").expanduser()))

import numpy as np
import torch
import hydra
from tqdm import tqdm

try:
    from .transform_no_chainer import load_data_for_npz
except:
    from transform_no_chainer import load_data_for_npz

# speakerのみ変更してください
speaker = "F01_kabulab"
#LIP_PATH = Path(f"~/dataset/lip/cropped/{speaker}").expanduser()

LIP_PATH = Path("/home/usr1/q70261a/dataset/lip/tmp_add/F01_kablab_20220930").expanduser()
LIP_TRAIN_DATA_PATH = Path(f"~/dataset/lip/np_files_96add/lip_cropped/train").expanduser()
LIP_TRAIN_MEAN_STD_SAVE_PATH = Path(f"~/dataset/lip/np_files_96add/lip_cropped/mean_std").expanduser()
LIP_TEST_DATA_PATH = Path(f"~/dataset/lip/np_files_96add/lip_cropped/test").expanduser()
LIP_TEST_MEAN_STD_SAVE_PATH = Path(f"~/dataset/lip/np_files_96add/lip_cropped/mean_std").expanduser()


def get_dataset_lip(data_root):    
    """
    mp4, wavまでのパス取得
    mp4とwavが同じディレクトリに入っている状態を想定
    """
    train_items = []
    test_items = []
    
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".wav"):
                # ATRのjセットをテストデータにするので，ここで分けます
                if '_j' in Path(file).stem:
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}.mp4")
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            test_items.append([video_path, audio_path])
                else:
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}.mp4")
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            train_items.append([video_path, audio_path])
    return train_items, test_items


def get_dataset_face(data_root):    
    """
    mp4, wavまでのパス取得
    mp4とwavが同じディレクトリに入っている状態を想定
    """
    train_items = []
    test_items = []
    
    for curdir, dir, files in os.walk(data_root):
        for file in files:
            if file.endswith(".wav"):
                if '_norm.wav' in Path(file).stem:
                    continue

                else:
                    if '_j' in Path(file).stem:
                        audio_path = os.path.join(curdir, file)
                        video_path = os.path.join(curdir, f"{Path(file).stem}.mp4")
                        if os.path.isfile(video_path) and os.path.isfile(audio_path):
                                test_items.append([video_path, audio_path])
                    else:
                        audio_path = os.path.join(curdir, file)
                        video_path = os.path.join(curdir, f"{Path(file).stem}.mp4")
                        if os.path.isfile(video_path) and os.path.isfile(audio_path):
                                train_items.append([video_path, audio_path])
    return train_items, test_items


def save_data_train(items, len, cfg, data_save_path, mean_std_save_path, device, time_only):
    """
    データ，平均，標準偏差の保存
    話者ごとに行うことを想定してます
    """
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0

    print("save train data")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path = items[i]
            video_path, audio_path = Path(video_path), Path(audio_path)

            # 話者ラベル(F01_kablabとかです)
            speaker = audio_path.parents[0].name

            wav, (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
                video_path=video_path,
                audio_path=audio_path,
                cfg=cfg,
            )
       
            if cfg.model.name == "mspec80_trans":
                assert feature.shape[-1] == 80
            elif cfg.model.name == "mspec40":
                assert feature.shape[-1] == 40
            elif cfg.model.name == "mspec60":
                assert feature.shape[-1] == 60
            elif cfg.model.name == "world":
                assert feature.shape[-1] == 29
            elif cfg.model.name == "world_melfb":
                assert feature.shape[-1] == 32
            
            assert feat_add.shape[-1] == 2
            
            # データの保存
            os.makedirs(os.path.join(data_save_path, speaker), exist_ok=True)
            np.savez(
                f"{data_save_path}/{speaker}/{audio_path.stem}_{cfg.model.name}",
                wav=wav,
                lip=lip,
                feature=feature,
                feat_add=feat_add,
                upsample=upsample,
                data_len=data_len,
            )

            lip = torch.from_numpy(lip).to(device)
            feature = torch.from_numpy(feature).to(device)
            feat_add = torch.from_numpy(feat_add).to(device)

            if time_only:
                # 時間方向のみの平均、標準偏差を計算
                lip_mean += torch.mean(lip.float(), dim=3)
                lip_std += torch.std(lip.float(), dim=3)
            else:
                # 時間、空間方向両方の平均、標準偏差を計算
                lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
                lip_std += torch.std(lip.float(), dim=(1, 2, 3))

            feat_mean += torch.mean(feature, dim=0)
            feat_std += torch.std(feature, dim=0)
            feat_add_mean += torch.mean(feat_add, dim=0)
            feat_add_std += torch.std(feat_add, dim=0)
        except:
            print(f"error : {audio_path.stem}")

    # データ全体の平均、分散を計算 (C,) チャンネルごと
    lip_mean /= len     
    lip_std /= len      
    feat_mean /= len    
    feat_std /= len     
    feat_add_mean /= len
    feat_add_std /= len

    lip_mean = lip_mean.to('cpu').detach().numpy().copy()
    lip_std = lip_std.to('cpu').detach().numpy().copy()
    feat_mean = feat_mean.to('cpu').detach().numpy().copy()
    feat_std = feat_std.to('cpu').detach().numpy().copy()
    feat_add_mean = feat_add_mean.to('cpu').detach().numpy().copy()
    feat_add_std = feat_add_std.to('cpu').detach().numpy().copy()
    
    os.makedirs(os.path.join(mean_std_save_path, speaker), exist_ok=True)
    np.savez(
        f"{mean_std_save_path}/{speaker}/train_{cfg.model.name}",
        lip_mean=lip_mean, 
        lip_std=lip_std, 
        feat_mean=feat_mean, 
        feat_std=feat_std, 
        feat_add_mean=feat_add_mean, 
        feat_add_std=feat_add_std,
    )


def save_data_test(items, len, cfg, data_save_path, mean_std_save_path, device, time_only):
    lip_mean = 0
    lip_std = 0
    feat_mean = 0
    feat_std = 0
    feat_add_mean = 0
    feat_add_std = 0

    print("save test data")
    for i in tqdm(range(len)):
        try:
            video_path, audio_path = items[i]
            video_path, audio_path = Path(video_path), Path(audio_path)

            # 話者ラベル
            speaker = video_path.parents[0].name

            wav, (lip, feature, feat_add, upsample), data_len = load_data_for_npz(
                video_path=video_path,
                audio_path=audio_path,
                cfg=cfg,
            )

            if cfg.model.name == "mspec80":
                assert feature.shape[-1] == 80
            elif cfg.model.name == "mspec40":
                assert feature.shape[-1] == 40
            elif cfg.model.name == "mspec60":
                assert feature.shape[-1] == 60
            elif cfg.model.name == "world":
                assert feature.shape[-1] == 29
            elif cfg.model.name == "world_melfb":
                assert feature.shape[-1] == 32
            
            assert feat_add.shape[-1] == 2
            
            # データの保存
            os.makedirs(os.path.join(data_save_path, speaker), exist_ok=True)
            np.savez(
                f"{data_save_path}/{speaker}/{audio_path.stem}_{cfg.model.name}",
                wav=wav,
                lip=lip,
                feature=feature,
                feat_add=feat_add,
                upsample=upsample,
                data_len=data_len,
            )

            lip = torch.from_numpy(lip).to(device)
            feature = torch.from_numpy(feature).to(device)
            feat_add = torch.from_numpy(feat_add).to(device)

            if time_only:
                # 時間方向のみの平均、標準偏差を計算
                lip_mean += torch.mean(lip.float(), dim=3)
                lip_std += torch.std(lip.float(), dim=3)
            else:
                # 時間、空間方向両方の平均、標準偏差を計算
                lip_mean += torch.mean(lip.float(), dim=(1, 2, 3))
                lip_std += torch.std(lip.float(), dim=(1, 2, 3))
            
            feat_mean += torch.mean(feature, dim=0)
            feat_std += torch.std(feature, dim=0)
            feat_add_mean += torch.mean(feat_add, dim=0)
            feat_add_std += torch.std(feat_add, dim=0)
        except:
            print(f"error : {audio_path.stem}")

    # データ全体の平均、分散を計算 (C,) チャンネルごと
    lip_mean /= len     
    lip_std /= len      
    feat_mean /= len    
    feat_std /= len     
    feat_add_mean /= len
    feat_add_std /= len

    lip_mean = lip_mean.to('cpu').detach().numpy().copy()
    lip_std = lip_std.to('cpu').detach().numpy().copy()
    feat_mean = feat_mean.to('cpu').detach().numpy().copy()
    feat_std = feat_std.to('cpu').detach().numpy().copy()
    feat_add_mean = feat_add_mean.to('cpu').detach().numpy().copy()
    feat_add_std = feat_add_std.to('cpu').detach().numpy().copy()
    
    os.makedirs(os.path.join(mean_std_save_path, speaker), exist_ok=True)
    np.savez(
        f"{mean_std_save_path}/{speaker}/test_{cfg.model.name}",
        lip_mean=lip_mean, 
        lip_std=lip_std, 
        feat_mean=feat_mean, 
        feat_std=feat_std, 
        feat_add_mean=feat_add_mean, 
        feat_add_std=feat_add_std,
    )


def save_data(data_root, train_data_save_path, train_mean_std_save_path, test_data_save_path, test_mean_std_save_path, cfg, device, time_only=True):
    train_items, test_items = get_dataset_lip(
        data_root=data_root,
    )
    n_data_train = len(train_items)
    n_data_test = len(test_items)
    
    save_data_train(
        items=train_items,
        len=n_data_train,
        cfg=cfg,
        data_save_path=str(train_data_save_path),
        mean_std_save_path=str(train_mean_std_save_path),
        device=device,
        time_only=time_only,
    )

    save_data_test(
        items=test_items,
        len=n_data_test,
        cfg=cfg,
        data_save_path=str(test_data_save_path),
        mean_std_save_path=str(test_mean_std_save_path),
        device=device,
        time_only=time_only,
    )


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    """
    顔をやるか口唇切り取ったやつをやるかでpathを変更してください
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device = {device}")

    # 口唇切り取った動画
    print("--- lip data processing ---")
    print(f"speaker = {speaker}, mode = {cfg.model.name}")
    # save_data(
    #     data_root=LIP_PATH,
    #     train_data_save_path=cfg.train.lip_pre_loaded_path,
    #     train_mean_std_save_path=cfg.train.lip_mean_std_path,
    #     test_data_save_path=cfg.test.lip_pre_loaded_path,
    #     test_mean_std_save_path=cfg.test.lip_mean_std_path,
    #     cfg=cfg,
    #     device=device,
    #     time_only=True,
    # )
    save_data(
        data_root=LIP_PATH,
        train_data_save_path=LIP_TRAIN_DATA_PATH,
        train_mean_std_save_path=LIP_TRAIN_MEAN_STD_SAVE_PATH,
        test_data_save_path=LIP_TEST_DATA_PATH,
        test_mean_std_save_path=LIP_TEST_MEAN_STD_SAVE_PATH,
        cfg=cfg,
        device=device,
        time_only=True,
    )
    print("Done")


    # # 顔
    # print("--- face data processing ---")
    # save_data(
    #     data_root=FACE_PATH,
    #     train_data_save_path=cfg.train.face_pre_loaded_path,
    #     train_mean_std_save_path=cfg.train.face_mean_std_path,
    #     test_data_save_path=cfg.test.face_pre_loaded_path,
    #     test_mean_std_save_path=cfg.test.face_mean_std_path,
    #     cfg=cfg,
    #     device=device,
    # )
    # print("Done")


if __name__ == "__main__":
    main()