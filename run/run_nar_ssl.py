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
        kablab_use,
        tcd_timit_use,
        ssl_feature_dropout,
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
            f'model.raven_config.model_size={model_size}',
            f'model.vatlm_config.model_size={model_size}',
            f'model.ssl_feature_dropout={ssl_feature_dropout}',
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
            f'model.raven_config.model_size={model_size}',
            f'model.vatlm_config.model_size={model_size}',
            f'model.ssl_feature_dropout={ssl_feature_dropout}',
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
        {
            'corpus': [],
            'speaker': ['spk1', 'spk2', 'spk3'],
            'kablab_use': False,
            'tcd_timit_use': True,
        },
        # {
        #     'corpus': ['ATR', 'BASIC5000'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'tcd_timit_use': False,
        # },
    ]
    model_condition_list = [
        {
            'model_name': 'avhubert',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'avhubert',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'raven',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'lightweight',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
        },
        {
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0.1,
        },
        {
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0.2,
        },
        {
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0.3,
        },
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
                kablab_use=data['kablab_use'],
                tcd_timit_use=data['tcd_timit_use'],
                ssl_feature_dropout=model_condition['ssl_feature_dropout'],
            )


def main():
    experiments()


if __name__ == '__main__':
    main()