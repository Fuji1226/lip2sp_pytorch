from pathlib import Path
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import torch
import numpy as np


DEBUG = False
WANDB_CONF = 'debug' if DEBUG else 'audio_ae'
WANDB_CONF_NAR = 'debug' if DEBUG else 'nar'
SUBJECT = 'プログラム経過'


def get_last_checkpoint_path(checkpoint_dir):
    checkpoint_dir_list = list(checkpoint_dir.glob('*'))
    checkpoint_dir_list = sorted(checkpoint_dir_list, reverse=False)
    required_checkpoint_dir = checkpoint_dir_list[-1]
    checkpoint_path_list = list(required_checkpoint_dir.glob('*'))
    checkpoint_path_list = sorted(
        checkpoint_path_list, 
        reverse=False, 
        key=lambda x: int(x.stem.split('_')[-1])
    )
    checkpoint_path = checkpoint_path_list[-1]
    return checkpoint_path


def get_best_checkpoint_path(checkpoint_path_last, metric_for_select):
    checkpoint_dict_last = torch.load(str(checkpoint_path_last))
    best_checkpoint = np.argmin(checkpoint_dict_last[metric_for_select]) + 1
    filename_prev = checkpoint_path_last.stem
    filename_new = filename_prev.split('_')
    filename_new[-1] = str(best_checkpoint)
    filename_new = '_'.join(filename_new)
    checkpoint_path = Path(str(checkpoint_path_last).replace(filename_prev, filename_new))
    return checkpoint_path


def get_result(result_dir):
    result_dir_list = list(result_dir.glob('*'))
    result_dir_list = sorted(result_dir_list, reverse=False)
    required_result = result_dir_list[-1]
    result_dir = list(required_result.glob('**/test_data'))[0]

    result_audio_path = result_dir / 'accuracy_griffinlim_audio.txt'
    result_lip_path = result_dir / 'accuracy_griffinlim_lip.txt'
    content_dict = {}

    if result_audio_path.exists() and result_lip_path.exists():
        with open(str(result_audio_path), 'r') as f:
            content_dict['audio'] = f.read()
        with open(str(result_lip_path), 'r') as f:
            content_dict['lip'] = f.read()
        content_dict['audio'] = 'result audio\n' + content_dict['audio']
        content_dict['lip'] = 'result lip\n' + content_dict['lip']
        content = '\n\n'.join(content_dict.values())
    elif result_audio_path.exists():
        with open(str(result_audio_path), 'r') as f:
            content = f.read()
    elif result_lip_path.exists():
        with open(str(result_lip_path), 'r') as f:
            content = f.read()

    return content


def get_result_nar(result_dir):
    result_dir_list = list(result_dir.glob('*'))
    result_dir_list = sorted(result_dir_list, reverse=False)
    required_result = result_dir_list[-1]
    result_dir = list(required_result.glob('**/test_data'))[0]
    result_path = result_dir / 'accuracy_griffinlim.txt'
    with open(str(result_path), 'r') as f:
        content = f.read()
    return content


def send_email(subject, body):
    from_email = "tmnm13009@gmail.com"
    password = "igumvwzbztowbigt"  # googleアカウントでアプリパスワードを取得し、それを利用する
    to_email = "tmnm13009@gmail.com"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        print("メールが送信されました。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def run_program(script, subject, body):
    subprocess.run(script)
    send_email(subject, body)


def run_nar(
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate,
    metric_for_select,
    use_spatial_aug,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=nar',
            'test=nar',
            f'wandb_conf={WANDB_CONF_NAR}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=[]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.995',
            'train.max_epoch=200',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            'train.use_segment_masking=True',
            'model.which_res="default_remake"',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=nar',
            'test=nar',
            f'wandb_conf={WANDB_CONF_NAR}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=[]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.995',
            'train.max_epoch=200',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            'train.use_segment_masking=True',
            'model.which_res="default_remake"',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate}'
    )
    send_email(subject=SUBJECT, body=get_result_nar(result_dir))


def run_audio_ae(
    ae_emb_dim, 
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate, 
    metric_for_select,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["lip_encoder"]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.92',
            'train.max_epoch=50',
            f'train.pretrained_model_path=""',
            'train.load_pretrained_model=False',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train} ae_emb_dim={ae_emb_dim}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["lip_encoder"]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.92',
            'train.max_epoch=50',
            f'train.pretrained_model_path=""',
            'train.load_pretrained_model=False',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
            'test.save_pred_audio=True',
            'test.save_pred_lip=False',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate} ae_emb_dim={ae_emb_dim}'
    )
    send_email(subject=SUBJECT, body=get_result(result_dir))
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    return checkpoint_path_best


def run_audio_ae_lip(
    ae_emb_dim, 
    pretrained_model_path, 
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate, 
    metric_for_select,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["audio_encoder", "audio_decoder"]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.apply_upsampling=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.995',
            'train.max_epoch=200',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train} ae_emb_dim={ae_emb_dim}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["audio_encoder", "audio_decoder"]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.apply_upsampling=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.995',
            'train.max_epoch=200',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
            'test.save_pred_audio=False',
            'test.save_pred_lip=True',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate} ae_emb_dim={ae_emb_dim}'
    )
    send_email(subject=SUBJECT, body=get_result(result_dir))
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    return checkpoint_path_best


def run_audio_ae_lip_dec(
    ae_emb_dim, 
    pretrained_model_path, 
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate, 
    metric_for_select,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["audio_encoder"]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.apply_upsampling=False',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train} ae_emb_dim={ae_emb_dim}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["audio_encoder"]',
            'train.use_jsut_corpus=False',
            'train.use_jvs_corpus=False',
            'train.apply_upsampling=False',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
            'test.save_pred_audio=False',
            'test.save_pred_lip=True',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate} ae_emb_dim={ae_emb_dim}'
    )
    send_email(subject=SUBJECT, body=get_result(result_dir))


def run_audio_ae_adv(
    ae_emb_dim, 
    pretrained_model_path, 
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate, 
    metric_for_select,
    bce_loss_weight,
    use_segment_masking,
    which_domain_classifier,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=[]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=True',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'train.bce_loss_weight={bce_loss_weight}',
            f'train.use_segment_masking={use_segment_masking}',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            'model.domain_classifier_hidden_channels=128',
            f'model.which_domain_classifier={which_domain_classifier}',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train} ae_emb_dim={ae_emb_dim}, bce_loss_weight={bce_loss_weight}, use_segment_masking={use_segment_masking}, which_domain_classifier={which_domain_classifier}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=[]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=True',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'train.bce_loss_weight={bce_loss_weight}',
            f'train.use_segment_masking={use_segment_masking}',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            'model.domain_classifier_hidden_channels=128',
            f'model.which_domain_classifier={which_domain_classifier}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
            'test.save_pred_audio=False',
            'test.save_pred_lip=True',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate} ae_emb_dim={ae_emb_dim}, bce_loss_weight={bce_loss_weight}, use_segment_masking={use_segment_masking}, which_domain_classifier={which_domain_classifier}'
    )
    send_email(subject=SUBJECT, body=get_result(result_dir))


def run_audio_ae_cycle(
    ae_emb_dim, 
    pretrained_model_path, 
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate, 
    metric_for_select,
    which_domain_classifier,
    which_feature_converter,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["lip_encoder", "audio_encoder", "audio_decoder"]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=True',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            'model.domain_classifier_hidden_channels=128',
            'model.converter_hidden_channels=128',
            f'model.which_domain_classifier={which_domain_classifier}',
            f'model.which_feature_converter={which_feature_converter}',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_train} ae_emb_dim={ae_emb_dim}, which_domain_classifier={which_domain_classifier}, which_feature_converter={which_feature_converter}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            f'wandb_conf={WANDB_CONF}',
            f'train.debug={DEBUG}',
            'train.module_is_fixed=["lip_encoder", "audio_encoder", "audio_decoder"]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=True',
            'train.lr=1.0e-4',
            'train.lr_decay_exp=0.9',
            'train.max_epoch=30',
            f'train.pretrained_model_path={pretrained_model_path}',
            'train.load_pretrained_model=True',
            f'model.ae_emb_dim={ae_emb_dim}',
            'model.audio_enc_which_encoder=""',
            'model.audio_enc_hidden_channels=64',
            'model.domain_classifier_hidden_channels=128',
            'model.converter_hidden_channels=128',
            f'model.which_domain_classifier={which_domain_classifier}',
            f'model.which_feature_converter={which_feature_converter}',
            f'model.which_domain_classifier={which_domain_classifier}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={DEBUG}',
            'test.save_pred_audio=False',
            'test.save_pred_lip=True',
        ],
        subject=SUBJECT,
        body=f'finish {run_filename_generate} ae_emb_dim={ae_emb_dim}, which_domain_classifier={which_domain_classifier}, which_feature_converter={which_feature_converter}'
    )
    send_email(subject=SUBJECT, body=get_result(result_dir))


def main():
    # run_nar(
    #     checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/face_cropped_max_size_fps25').expanduser(),
    #     result_dir=Path('~/lip2sp_pytorch/result/nar/generate/face_cropped_max_size_fps25').expanduser(),
    #     run_filename_train='train_nar_amp.py',
    #     run_filename_generate='generate_nar.py',
    #     metric_for_select='val_loss_list',
    #     use_spatial_aug=False,
    # )

    for ae_emb_dim in [8, 16, 32]:
        if ae_emb_dim == 8:
            checkpoint_path_best_ae_lip = Path('/home/minami/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25/2023:08:24_10-50-37/mspec80_176.ckpt')
        elif ae_emb_dim == 16:
            checkpoint_path_best_ae_lip = Path('/home/minami/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25/2023:08:25_01-20-53/mspec80_117.ckpt')
        elif ae_emb_dim == 32:
            checkpoint_path_best_ae_lip = Path('/home/minami/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25/2023:08:25_16-10-16/mspec80_182.ckpt')
        
        # checkpoint_path_best_ae = run_audio_ae(
        #     ae_emb_dim=ae_emb_dim,
        #     checkpoint_dir=Path('~/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25').expanduser(),
        #     result_dir=Path('~/lip2sp_pytorch/result/audio_ae/generate/face_cropped_max_size_fps25').expanduser(),
        #     run_filename_train='train_audio_ae.py',
        #     run_filename_generate='generate_audio_ae_lip.py',
        #     metric_for_select='val_loss_list',
        # )

        # checkpoint_path_best_ae_lip = run_audio_ae_lip(
        #     ae_emb_dim=ae_emb_dim,
        #     pretrained_model_path=str(checkpoint_path_best_ae),
        #     checkpoint_dir=Path('~/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25').expanduser(),
        #     result_dir=Path('~/lip2sp_pytorch/result/audio_ae/generate/face_cropped_max_size_fps25').expanduser(),
        #     run_filename_train='train_audio_ae_lip.py',
        #     run_filename_generate='generate_audio_ae_lip.py',
        #     metric_for_select='val_mse_loss_mel_lip_list',
        # )

        # run_audio_ae_lip_dec(
        #     ae_emb_dim=ae_emb_dim,
        #     pretrained_model_path=str(checkpoint_path_best_ae_lip),
        #     checkpoint_dir=Path('~/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25').expanduser(),
        #     result_dir=Path('~/lip2sp_pytorch/result/audio_ae/generate/face_cropped_max_size_fps25').expanduser(),
        #     run_filename_train='train_audio_ae_lip_dec.py',
        #     run_filename_generate='generate_audio_ae_lip.py',
        #     metric_for_select='val_loss_list',
        # )

        for which_domain_classifier in ['convrnn']:
            for bce_loss_weight in [0, 0.1, 0.3, 0.5]:
                run_audio_ae_adv(
                    ae_emb_dim=ae_emb_dim,
                    pretrained_model_path=str(checkpoint_path_best_ae_lip),
                    checkpoint_dir=Path('~/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25').expanduser(),
                    result_dir=Path('~/lip2sp_pytorch/result/audio_ae/generate/face_cropped_max_size_fps25').expanduser(),
                    run_filename_train='train_audio_ae_adv_fix.py',
                    run_filename_generate='generate_audio_ae_lip.py',
                    metric_for_select='val_mse_loss_mel_lip_list',
                    bce_loss_weight=bce_loss_weight,
                    use_segment_masking=True,
                    which_domain_classifier=which_domain_classifier,
                )
                if DEBUG:
                    break

        # for which_domain_classifier in ['linear', 'convrnn']:
        #     for which_feature_converter in ['linear', 'convrnn']:
        #         run_audio_ae_cycle(
        #             ae_emb_dim=ae_emb_dim,
        #             pretrained_model_path=str(checkpoint_path_best_ae_lip),
        #             checkpoint_dir=Path('~/lip2sp_pytorch/check_point/audio_ae/face_cropped_max_size_fps25').expanduser(),
        #             result_dir=Path('~/lip2sp_pytorch/result/audio_ae/generate/face_cropped_max_size_fps25').expanduser(),
        #             run_filename_train='train_audio_ae_cycle.py',
        #             run_filename_generate='generate_audio_ae_cycle.py',
        #             metric_for_select='val_mse_loss_audio_from_lip_mel_list',
        #             which_domain_classifier=which_domain_classifier,
        #             which_feature_converter=which_feature_converter,
        #         )

    
if __name__ == '__main__':
    main()