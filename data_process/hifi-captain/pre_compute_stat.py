import sys
from pathlib import Path
import hydra
import numpy as np
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

from utils import (
    get_datasets_external_data_raw,
)
from dataset.utils import (
    get_stat_load_data_raw,
    calc_mean_var_std,
)


@hydra.main(version_base=None, config_name="config", config_path="../../conf")
def main(cfg):
    train_external_data_path_list = get_datasets_external_data_raw(cfg, 'train')
    lip_mean_list, lip_var_list, lip_len_list, feat_mean_list, feat_var_list, feat_len_list = get_stat_load_data_raw(
        train_external_data_path_list, cfg
    )
    feat_mean, feat_var, feat_std = calc_mean_var_std(feat_mean_list, feat_var_list, feat_len_list)
    save_path = Path('/home/minami/dataset/hi-fi-captain/ja-JP')
    np.savez(
        str(save_path / 'feat_mean_var_std'),
        feat_mean=feat_mean,
        feat_var=feat_var,
        feat_std=feat_std,
    )


if __name__ == '__main__':
    main()