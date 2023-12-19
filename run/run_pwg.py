import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))
import subprocess


def main():
    debug = False
    wandb_conf = 'debug' if debug else 'pwg'
    # subprocess.run(
    #     [
    #         'python',
    #         '/home/minami/lip2sp_pytorch/parallelwavegan/pwg_train.py',
    #         'model=master',
    #         'train=pwg',
    #         'test=pwg',
    #         f'wandb_conf={wandb_conf}',
    #         f'train.debug={debug}',
    #         'train.jvs.use=False',
    #         'train.hifi_captain.use=True',
    #         'train.check_point_start_separate_save_dir=False',
    #         'train.start_ckpt_path_separate_save_dir=""',
    #         'model.input_lip_sec=1.0',
    #     ]
    # )
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/parallelwavegan/pwg_train.py',
            'model=master',
            'train=pwg',
            'test=pwg',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            'train.jvs.use=True',
            'train.hifi_captain.use=False',
            'train.check_point_start_separate_save_dir=True',
            'train.start_ckpt_path_separate_save_dir="/home/minami/lip2sp_pytorch/check_point/pwg/avhubert_preprocess_fps25_gray/master/2023:12:17_16-16-42/15.ckpt"',
            'model.input_lip_sec=1.0',
        ]
    )
    
    
    
if __name__ == '__main__':
    main()