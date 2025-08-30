from pathlib import Path
from tqdm import tqdm
import numpy as np
import hydra
from feature import wav2mel, wav2mel_avhubert

#もしかして、make_npz.pyのあとにやるやつ？

def pad(feature, data_len):
    feature_padded = np.zeros((feature.shape[0], data_len), dtype=feature.dtype)
    feature_padded[:, :feature.shape[-1]] = feature
    return feature_padded


@hydra.main(config_name="config", config_path="../conf")
def main(cfg):
    #!データ読み込み先の指定！忘れない！
    #いつものnpzファイルは、~/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/[data_split]/[speaker]/master/[.npz]
    data_dir = Path('~/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray').expanduser()
    data_path_list_kab = list(data_dir.glob('*/kab2022/master/*.npz')) #kablab2022に、lipがない
    data_path_list_katsu = list(data_dir.glob('*/F1/master/*.npz')) #kablab2022に、lipがない

    #"""
    # 追加: 最初の5つのパスを表示
    print("kabの1つのnpzファイルパス:")
    for p in data_path_list_kab[:1]:
        npz = np.load(p)
        print(p)
        print(npz.files)
        #[wav, feature]

    for p in data_path_list_katsu[:-1]:
        npz = np.load(p)
        print(p)
        print(npz.files)
        print(npz['lip'].shape, npz['wav'].shape, npz['feature'].shape, npz['feature_avhubert'].shape)
        #['wav', 'lip', 'feature', 'feat_add', 'landmark', 'upsample', 'data_len']

    breakpoint()
    #"""

    data_path_list = data_path_list_kab
    for data_path in tqdm(data_path_list):
        npz_key = np.load(str(data_path))
        lip = npz_key['lip']
        wav = npz_key['wav']
        feature = wav2mel(wav, cfg, ref_max=False)
        feature_avhubert = wav2mel_avhubert(wav, cfg)
        mul_factor = (cfg.model.sampling_rate // cfg.model.hop_length) // cfg.model.fps
        data_len = int(lip.shape[-1] * mul_factor)
        feature = feature[:, :data_len]
        feature_avhubert = feature_avhubert[:, :data_len]
        if feature.shape[-1] != data_len:
            feature = pad(feature, data_len)
        if feature_avhubert.shape[-1] != data_len:
            feature_avhubert = pad(feature_avhubert, data_len)
        assert feature.shape[-1] == data_len
        assert feature_avhubert.shape[-1] == data_len
        
        save_path = Path(str(data_path).replace('mspec80', 'mspec_avhubert'))
        save_path.parents[0].mkdir(parents=True, exist_ok=True)
        np.savez(
            str(save_path.parents[0] / save_path.stem),
            wav=wav,
            lip=lip,
            feature=feature.T,
            feature_avhubert=feature_avhubert.T,
        )


if __name__ == '__main__':
    main()