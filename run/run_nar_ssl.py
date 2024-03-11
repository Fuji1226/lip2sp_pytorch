import sys
from pathlib import Path

sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import subprocess

from run.utils import (
    clean_trash,
    get_best_checkpoint_path,
    get_last_checkpoint_path,
)


def run_nar(
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
        load_pretrained_weight,
        kablab_use,
        tcd_timit_use,
        ssl_feature_dropout,
        ckpt_path_avhubert,
        ckpt_path_raven,
        ckpt_path_vatlm,
):
    subprocess.run(
        [
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
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
            f'model.model_name={model_name}',
            f'model.avhubert_config.model_size={model_size}',
            f'model.avhubert_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.raven_config.model_size={model_size}',
            f'model.raven_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.vatlm_config.model_size={model_size}',
            f'model.vatlm_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.ssl_feature_dropout={ssl_feature_dropout}',
            f'model.ckpt_path_avhubert={ckpt_path_avhubert}',
            f'model.ckpt_path_raven={ckpt_path_raven}',
            f'model.ckpt_path_vatlm={ckpt_path_vatlm}',
        ]
    )
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
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.speaker={speaker}',
            f'test.debug={debug}',
            f'model.model_name={model_name}',
            f'model.avhubert_config.model_size={model_size}',
            f'model.avhubert_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.raven_config.model_size={model_size}',
            f'model.raven_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.vatlm_config.model_size={model_size}',
            f'model.vatlm_config.load_pretrained_weight={load_pretrained_weight}',
            f'model.ssl_feature_dropout={ssl_feature_dropout}',
            f'model.ckpt_path_avhubert={ckpt_path_avhubert}',
            f'model.ckpt_path_raven={ckpt_path_raven}',
            f'model.ckpt_path_vatlm={ckpt_path_vatlm}',
        ]
    )
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    clean_trash()
    return checkpoint_path_best


def experiments():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    data_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
        },
        # {
        #     'corpus': [],
        #     'speaker': ['spk1', 'spk2', 'spk3'],
        #     'kablab_use': False,
        #     'tcd_timit_use': True,
        # },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
        },
    ]
    model_condition_list = [
        {
            'model_name': 'avhubert',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'load_pretrained_weight': False,
        },
        {
            'model_name': 'raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'load_pretrained_weight': False,
        },
        {
            'model_name': 'vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'load_pretrained_weight': False,
        },
        # {
        #     'model_name': 'avhubert',
        #     'model_size': 'base',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'raven',
        #     'model_size': 'base',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'vatlm',
        #     'model_size': 'base',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'avhubert',
        #     'model_size': 'large',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'raven',
        #     'model_size': 'large',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'vatlm',
        #     'model_size': 'large',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
        # {
        #     'model_name': 'lightweight',
        #     'model_size': 'base',
        #     'ssl_feature_dropout': 0,
        #     'load_pretrained_weight': True,
        # },
    ]

    for data in data_list:
        for model_condition in model_condition_list:
            run_nar(
                run_filename_train='train_nar_ssl.py',
                run_filename_generate='generate_nar_ssl.py',
                wandb_conf=wandb_conf,
                debug=debug,
                module_is_fixed='',
                corpus=data['corpus'],
                speaker=data['speaker'],
                check_point_start_separate_save_dir=False,
                start_ckpt_path_separate_save_dir='',
                subject=subject,
                message='',
                checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master').expanduser(),
                result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master').expanduser(),
                metric_for_select='val_loss_list',
                model_name=model_condition['model_name'],
                model_size=model_condition['model_size'],
                load_pretrained_weight=model_condition['load_pretrained_weight'],
                kablab_use=data['kablab_use'],
                tcd_timit_use=data['tcd_timit_use'],
                ssl_feature_dropout=model_condition['ssl_feature_dropout'],
                ckpt_path_avhubert='',
                ckpt_path_raven='',
                ckpt_path_vatlm='',
            )


def experiments_ensemble():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    condition_list= [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_size': 'base',
            'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_03-47-15/50.ckpt',
            'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_05-08-07/46.ckpt',
            'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_06-28-58/48.ckpt',
        },
        # {
        #     'corpus': ['ATR', 'BASIC5000'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'kablab_use': True,
        #     'tcd_timit_use': False,
        #     'model_size': 'base',
        #     'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_01-37-46/49.ckpt',
        #     'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_08-22-13/49.ckpt',
        #     'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_15-26-19/50.ckpt',
        # },
        # {
        #     'corpus': [],
        #     'speaker': ['spk1', 'spk2', 'spk3'],
        #     'kablab_use': False,
        #     'tcd_timit_use': True,
        #     'model_size': 'base',
        #     'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_22-48-50/46.ckpt',
        #     'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:04_00-17-28/49.ckpt',
        #     'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:04_01-46-18/48.ckpt',
        # }, 
        # {
        #     'corpus': ['ATR'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'kablab_use': True,
        #     'tcd_timit_use': False,
        #     'model_size': 'large',
        #     'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_07-49-05/49.ckpt',
        #     'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_09-24-17/44.ckpt',
        #     'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_11-07-13/41.ckpt',
        # },
        # {
        #     'corpus': ['ATR', 'BASIC5000'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'kablab_use': True,
        #     'tcd_timit_use': False,
        #     'model_size': 'large',
        #     'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_22-13-20/40.ckpt',
        #     'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_06-44-11/50.ckpt',
        #     'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_16-08-25/47.ckpt',
        # },
        # {
        #     'corpus': [],
        #     'speaker': ['spk1', 'spk2', 'spk3'],
        #     'kablab_use': False,
        #     'tcd_timit_use': True,
        #     'model_size': 'large',
        #     'ckpt_path_avhubert': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:04_03-15-20/45.ckpt',
        #     'ckpt_path_raven': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:07_16-38-11/47.ckpt',
        #     'ckpt_path_vatlm': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:07_18-33-38/41.ckpt',
        # },
    ]
    model_name_list = [
        'ensemble',
        'ensemble_avhubert_vatlm',
        'ensemble_avhubert_raven',
        'ensemble_raven_vatlm',
    ]
    ssl_feature_dropout_list = [0,]

    for condition in condition_list:
        for model_name in model_name_list:
            for ssl_feature_dropout in ssl_feature_dropout_list:
                run_nar(
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
                )



def main():
    # experiments_ensemble()
    experiments()


if __name__ == '__main__':
    main()