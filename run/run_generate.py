import sys
from pathlib import Path
sys.path.append(str(Path('~/lip2sp_pytorch').expanduser()))

import subprocess


def run_generate(
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
    condition_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'avhubert',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_03-47-15/50.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_05-08-07/46.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_06-28-58/48.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'avhubert',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_07-49-05/49.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'raven',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_09-24-17/44.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_11-07-13/41.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'lightweight',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:03_12-42-34/47.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:06_22-38-46/48.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_16-38-06/43.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_17-59-59/40.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_raven_vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_19-20-26/45.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:08_12-44-50/47.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_20-41-03/36.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_raven',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_22-17-48/48.ckpt',
        },
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_raven_vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:10_23-57-41/47.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'avhubert',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_01-37-46/49.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_08-22-13/49.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_15-26-19/50.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'avhubert',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:11_22-13-20/40.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'raven',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_06-44-11/50.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:12_16-08-25/47.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'lightweight',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:13_00-40-17/48.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:13_15-32-33/50.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:13_23-21-58/42.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_raven',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:14_05-55-31/46.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_raven_vatlm',
            'model_size': 'base',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:14_12-38-25/47.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:14_19-18-11/47.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_06-02-04/42.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_avhubert_raven',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_14-05-12/46.ckpt',
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
            'kablab_use': True,
            'tcd_timit_use': False,
            'model_name': 'ensemble_raven_vatlm',
            'model_size': 'large',
            'ssl_feature_dropout': 0,
            'model_path': '/home/minami/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master/2023:12:15_22-27-49/48.ckpt',
        },
    ]
    for condition in condition_list:
        run_generate(
            run_filename_train='train_nar_ssl.py',
                run_filename_generate='generate_nar_ssl.py',
                wandb_conf='nar',
                debug=False,
                module_is_fixed='',
                corpus=condition['corpus'],
                speaker=condition['speaker'],
                check_point_start_separate_save_dir=False,
                start_ckpt_path_separate_save_dir='',
                subject='generate',
                message='',
                checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/master').expanduser(),
                result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master').expanduser(),
                metric_for_select='val_loss_list',
                model_name=condition['model_name'],
                model_size=condition['model_size'],
                kablab_use=condition['kablab_use'],
                tcd_timit_use=condition['tcd_timit_use'],
                ssl_feature_dropout=condition['ssl_feature_dropout'],
                ckpt_path_avhubert='',
                ckpt_path_raven='',
                ckpt_path_vatlm='',
                model_path=condition['model_path'],
        )
    
    
if __name__ == '__main__':
    main()