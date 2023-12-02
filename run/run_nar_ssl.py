import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

from run.utils import (
    clean_trash,
    get_last_checkpoint_path,
    get_best_checkpoint_path,
    get_result,
    send_email,
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
        tcd_timit_use,
        which_optim,
        weight_decay,
):
    run_program(
        script=[
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
            f'train.tcd_timit.use={tcd_timit_use}',
            f'train.which_optim={which_optim}',
            f'train.weight_decay={weight_decay}',
            f'model.model_name={model_name}',
            f'model.avhubert_config.model_size={model_size}',
            f'model.raven_config.model_size={model_size}',
            f'model.vatlm_config.model_size={model_size}',
        ],
        subject=subject,
        body=f'finish {run_filename_train}. {message}'
    )
    run_program(
        script=[
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
            f'train.tcd_timit.use={tcd_timit_use}',
            f'train.which_optim={which_optim}',
            f'train.weight_decay={weight_decay}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.speaker={speaker}',
            f'test.debug={debug}',
            f'model.model_name={model_name}',
            f'model.avhubert_config.model_size={model_size}',
            f'model.raven_config.model_size={model_size}',
            f'model.vatlm_config.model_size={model_size}',
        ],
        subject=subject,
        body=f'finish {run_filename_generate}. {message}'
    )
    send_email(
        subject=subject,
        body='result: griffin_lim\n\n' + get_result(result_dir, 'accuracy_griffinlim.txt')
    )
    send_email(
        subject=subject,
        body='result: pwg\n\n' + get_result(result_dir, 'accuracy_pwg.txt')
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
        # {
        #     'corpus': ['ATR'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'tcd_timit_use': False,
        # },
        {
            'corpus': [],
            'speaker': ['spk1', 'spk2', 'spk3'],
            'tcd_timit_use': True,
        },
        # {
        #     'corpus': ['ATR', 'BASIC5000'],
        #     'speaker': ["F01_kablab", "M01_kablab"],
        #     'tcd_timit_use': False,
        # },
    ]
    model_condition_list = [
        # {
        #     'model_name': 'avhubert',
        #     'model_size': 'base',
        # },
        # {
        #     'model_name': 'raven',
        #     'model_size': 'base',
        # },
        {
            'model_name': 'vatlm',
            'model_size': 'base',
        },
        # {
        #     'model_name': 'avhubert',
        #     'model_size': 'large',
        # },
        # {
        #     'model_name': 'raven',
        #     'model_size': 'large',
        # },
        {
            'model_name': 'vatlm',
            'model_size': 'large',
        },
        # {
        #     'model_name': 'lightweight',
        #     'model_size': 'base',
        # },
    ]
    optimizer_param_list = [
        {
            'which_optim': 'adam',
            'weight_decay': 1.0e-3,
        },
        {
            'which_optim': 'adam',
            'weight_decay': 1.0e-4,
        },
        {
            'which_optim': 'adam',
            'weight_decay': 1.0e-5,
        },
        {
            'which_optim': 'adam',
            'weight_decay': 1.0e-6,
        },
        {
            'which_optim': 'adam',
            'weight_decay': 0,
        },
        {
            'which_optim': 'adamw',
            'weight_decay': 1.0e-3,
        },
        {
            'which_optim': 'adamw',
            'weight_decay': 1.0e-4,
        },
        {
            'which_optim': 'adamw',
            'weight_decay': 1.0e-5,
        },
        {
            'which_optim': 'adamw',
            'weight_decay': 1.0e-6,
        },
        {
            'which_optim': 'adamw',
            'weight_decay': 0,
        },
    ]

    for data in data_list:
        for model_condition in model_condition_list:
            for optimizer_param in optimizer_param_list:
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
                    message='test',
                    checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master').expanduser(),
                    result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master').expanduser(),
                    metric_for_select='val_loss_list',
                    model_name=model_condition['model_name'],
                    model_size=model_condition['model_size'],
                    tcd_timit_use=data['tcd_timit_use'],
                    which_optim=optimizer_param['which_optim'],
                    weight_decay=optimizer_param['weight_decay'],
                )


def experiments_ensembles():
    debug = True
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    condition_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'base',
            'ckpt_path_avhubert': '~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-16-55/2.ckpt',
            'ckpt_path_raven': '~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-18-17/2.ckpt',
            'ckpt_path_vatlm': '~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:11:25_19-19-35/2.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'large',
            'ckpt_path_avhubert': '',
            'ckpt_path_raven': '',
            'ckpt_path_vatlm': '',
        },
        {
            'corpus': [],
            'speaker': ['spk1', 'spk2', 'spk3'],
            'tcd_timit_use': True,
            'model_name': 'ensemble',
            'model_size': 'base',
            'ckpt_path_avhubert': '',
            'ckpt_path_raven': '',
            'ckpt_path_vatlm': '',
        },
        {
            'corpus': [],
            'speaker': ['spk1', 'spk2', 'spk3'],
            'tcd_timit_use': True,
            'model_name': 'ensemble',
            'model_size': 'large',
            'ckpt_path_avhubert': '',
            'ckpt_path_raven': '',
            'ckpt_path_vatlm': '',
        },
    ]

    for condition in condition_list:
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
            message='test',
            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master').expanduser(),
            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master').expanduser(),
            metric_for_select='val_loss_list',
            model_name=condition['model_name'],
            model_size=condition['model_size'],
            tcd_timit_use=condition['tcd_timit_use'],
        )


def main():
    experiments()
    # experiments_ensembles()


if __name__ == '__main__':
    main()