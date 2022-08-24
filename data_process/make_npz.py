"""
事前にnpzファイルを作るためのコード
F01_kablabなど話者ごとにディレクトリを分けて,その中にmp4とwavがあることを想定しています
dataset/lip/lip_cropped/F01_kablab

実行すると
平均,標準偏差が下のディレクトリに
face
    dataset/lip/np_files/face/mean_std/F01_kablab
lip
    dataset/lip/np_files/lip_cropped/mean_std/F01_kablab

データを読み込んだ結果が下の2つに保存されるはずです
face
    dataset/lip/np_files/face/train/F01_kablab
    dataset/lip/np_files/face/test/F01_kablab
lip 
    dataset/lip/np_files/lip_cropped/train/F01_kablab
    dataset/lip/np_files/lip_cropped/test/F01_kablab

このディレクトリ構造になることを前提にdataset_npz.pyを書いているので,使用する場合は揃えてほしいです!

顔でやる場合はcroppedまでのパスを通してください(FACE_PATH)
口唇動画でやる場合は先にdata_process/lip_crop_ito.pyで口唇動画を作成してください(LIP_PATH)

LIP_PATH, LIP_PATH_128128, LIP_PATH_9696のの違いは口唇部分の切り取り範囲です
    LIP_PATH : (96, 128)
    LIP_PATH_128128 : (128, 128)
    LIP_PATH_9696 : (96, 96)

実行はshellのmake_npz.shでお願いします
model=
    mspec 
    world 
    world_melfb
で，それぞれいけます

手順
1. LIP_PATH, FACE_PATHの変更
    動画と音声ファイルを保存しているディレクトリまでのパスを設定してください
    また,口唇部分をdata_process/ip_crop_ito.pyで切り抜いた後などは,wavファイルがないと思います
    その時はdata_process/copy_wav.pyを実行し,先にwavファイルをコピーしておいてください(同じディレクトリに動画と音声があることを想定して書いているので…)

2. conf/trainとconf/testのパスの変更
    ここが保存先になり,train.pyなどを実行する際に読み込む対象です
    lip_mean_std_path
    face_mean_std_path
    lip_pre_loaded_path
    face_pre_loaded_path

    追記
    現在はPath().expanduser()を使用しているので,パスを変更しないで大丈夫です
    
3. shells/make_npz.shの実行
    modelを変更すれば,それに対応した音響特徴量を計算します
    また,必要ないやつはコメントアウトしてください
"""

import os
import sys
from pathlib import Path

# 親ディレクトリからのimport用
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pathlib import Path

import numpy as np
import torch
import hydra

try:
    from .transform_no_chainer import load_data_for_npz
except:
    from transform_no_chainer import load_data_for_npz

LIP_PATH = "/home/usr4/r70264c/dataset/lip/lip_cropped"     # 変更
FACE_PATH = "/home/usr4/r70264c/dataset/lip/cropped"        # 変更
LIP_PATH_128128 = "/home/usr4/r70264c/dataset/lip/lip_cropped_128128"     # 変更
LIP_PATH_9696 = "/home/usr4/r70264c/dataset/lip/lip_cropped_9696"   # 変更


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
                    video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
                    if os.path.isfile(video_path) and os.path.isfile(audio_path):
                            test_items.append([video_path, audio_path])
                else:
                    audio_path = os.path.join(curdir, file)
                    video_path = os.path.join(curdir, f"{Path(file).stem}_crop.mp4")
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

    for i in range(len):
        video_path, audio_path = items[i]
        video_path, audio_path = Path(video_path), Path(audio_path)

        # 話者ラベル(F01_kablabとかです)
        speaker = audio_path.parents[0].name

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
            print("time only")
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

    for i in range(len):
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
            print("time only")
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


def save_data(data_root, train_data_save_path, train_mean_std_save_path, test_data_save_path, test_mean_std_save_path, cfg, device, time_only=False):
    train_items, test_items = get_dataset_lip(
        data_root=data_root,
    )
    n_data_train = len(train_items)
    n_data_test = len(test_items)
    
    save_data_train(
        items=train_items,
        len=n_data_train,
        cfg=cfg,
        data_save_path=str(Path(train_data_save_path).expanduser()),
        mean_std_save_path=str(Path(train_mean_std_save_path).expanduser()),
        device=device,
        time_only=time_only,
    )

    save_data_test(
        items=test_items,
        len=n_data_test,
        cfg=cfg,
        data_save_path=str(Path(test_data_save_path).expanduser()),
        mean_std_save_path=str(Path(test_mean_std_save_path).expanduser()),
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


    # 口唇切り取った動画(96, 96)で平均を時間方向のみ
    print("--- lip data 9696 time_only processing ---")
    save_data(
        data_root=LIP_PATH_9696,
        train_data_save_path=cfg.train.lip_pre_loaded_path_9696_time_only,
        train_mean_std_save_path=cfg.train.lip_mean_std_path_9696_time_only,
        test_data_save_path=cfg.test.lip_pre_loaded_path_9696_time_only,
        test_mean_std_save_path=cfg.test.lip_mean_std_path_9696_time_only,
        cfg=cfg,
        device=device,
        time_only=True,
    )
    print("Done")


    # # 口唇切り取った動画(96, 96)
    # print("--- lip data 9696 processing ---")
    # save_data(
    #     data_root=LIP_PATH_9696,
    #     train_data_save_path=cfg.train.lip_pre_loaded_path_9696,
    #     train_mean_std_save_path=cfg.train.lip_mean_std_path_9696,
    #     test_data_save_path=cfg.test.lip_pre_loaded_path_9696,
    #     test_mean_std_save_path=cfg.test.lip_mean_std_path_9696,
    #     cfg=cfg,
    #     device=device,
    # )
    # print("Done")


    # # 口唇切り取った動画(128, 128)
    # print("--- lip data 128128 processing ---")
    # save_data(
    #     data_root=LIP_PATH_128128,
    #     train_data_save_path=cfg.train.lip_pre_loaded_path_128128,
    #     train_mean_std_save_path=cfg.train.lip_mean_std_path_128128,
    #     test_data_save_path=cfg.test.lip_pre_loaded_path_128128,
    #     test_mean_std_save_path=cfg.test.lip_mean_std_path_128128,
    #     cfg=cfg,
    #     device=device,
    # )
    # print("Done")


    # # 口唇切り取った動画(96, 128)
    # print("--- lip data processing ---")
    # save_data(
    #     data_root=LIP_PATH,
    #     train_data_save_path=cfg.train.lip_pre_loaded_path,
    #     train_mean_std_save_path=cfg.train.lip_mean_std_path,
    #     test_data_save_path=cfg.test.lip_pre_loaded_path,
    #     test_mean_std_save_path=cfg.test.lip_mean_std_path,
    #     cfg=cfg,
    #     device=device,
    # )
    # print("Done")


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