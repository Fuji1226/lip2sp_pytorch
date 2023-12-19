import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import subprocess
from run.utils import (
    clean_trash,
    get_last_checkpoint_path,
    get_best_checkpoint_path,
    run_program,
)


def run_calc_accuracy(
    run_filename_train,
    run_filename_generate,
    wandb_conf,
    debug,
    module_is_fixed,
    corpus,
    speaker,
    check_point_start_separate_save_dir,
    start_ckpt_path_separate_save_dir,
    subject,
    message,
    checkpoint_dir,
    metric_for_select,
    result_dir,
    model_name,
    model_size,
    kablab_use,
    tcd_timit_use,
    ssl_feature_dropout,
    ckpt_path_avhubert,
    ckpt_path_raven,
    ckpt_path_vatlm,
    model_path,
):
    subprocess.run(
        [
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=master',
            'train=nar',
            'test=nar',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            f'train.module_is_fixed={module_is_fixed}',
            f'train.corpus={corpus}',
            f'train.speaker={speaker}',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'train.kablab.use={kablab_use}',
            f'train.tcd_timit.use={tcd_timit_use}',
            f'test.model_path={model_path}',
            f'test.metric_for_select={metric_for_select}',
            f'test.speaker={speaker}',
            f'test.debug={debug}',
            f'model.model_name={model_name}',
            f'model.avhubert_config.model_size={model_size}',
            f'model.raven_config.model_size={model_size}',
            f'model.vatlm_config.model_size={model_size}',
            f'model.ssl_feature_dropout={ssl_feature_dropout}',
            f'model.ckpt_path_avhubert={ckpt_path_avhubert}',
            f'model.ckpt_path_raven={ckpt_path_raven}',
            f'model.ckpt_path_vatlm={ckpt_path_vatlm}',
        ]
    )


def main():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    condition_list= [
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_size': 'large',
            'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_22-13-20/40.ckpt',
            'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_06-44-11/50.ckpt',
            'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_16-08-25/47.ckpt',
        },
    ]
    model_name_list = [
        'ensemble_avhubert_vatlm',
        'ensemble_avhubert_raven',
        'ensemble_raven_vatlm',
    ]
    ssl_feature_dropout_list = [0,]
    
    for condition in condition_list:
        for model_name in model_name_list:
            if model_name == 'ensemble_avhubert_vatlm':
                model_path = '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_06-02-04/42.ckpt'
            elif model_name == 'ensemble_avhubert_raven':
                model_path = '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_14-05-12/46.ckpt'
            elif model_name == 'ensemble_raven_vatlm':
                model_path = '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_22-27-49/48.ckpt'
                
            for ssl_feature_dropout in ssl_feature_dropout_list:
                run_calc_accuracy(
                    run_filename_train='train_nar_ssl.py',
                    run_filename_generate='generate_nar_ssl.py',
                    wandb_conf=wandb_conf,
                    debug=debug,
                    module_is_fixed='',
                    corpus=condition['corpus'],
                    speaker=condition['speaker'],
                    check_point_start_separate_save_dir=False,
                    start_ckpt_path_separate_save_dir='',
                    subject=subject,
                    message='',
                    checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master').expanduser(),
                    result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master').expanduser(),
                    metric_for_select='val_loss_list',
                    model_name=model_name,
                    model_size=condition['model_size'],
                    kablab_use=condition['kablab_use'],
                    tcd_timit_use=condition['tcd_timit_use'],
                    ssl_feature_dropout=ssl_feature_dropout,
                    ckpt_path_avhubert=condition['ckpt_path_avhubert'],
                    ckpt_path_raven=condition['ckpt_path_raven'],
                    ckpt_path_vatlm=condition['ckpt_path_vatlm'],
                    model_path=model_path,
                )
                
                
if __name__ == '__main__':
    main()