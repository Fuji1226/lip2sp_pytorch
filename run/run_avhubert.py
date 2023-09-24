from pathlib import Path
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import torch
import numpy as np


def clean_trash():
    subprocess.run(['trash-empty'])


def get_last_checkpoint_path(checkpoint_dir):
    checkpoint_dir_list = list(checkpoint_dir.glob('*'))
    checkpoint_dir_list = sorted(checkpoint_dir_list, reverse=False)
    required_checkpoint_dir = checkpoint_dir_list[-1]
    checkpoint_path_list = list(required_checkpoint_dir.glob('*'))
    checkpoint_path_list = sorted(
        checkpoint_path_list, 
        reverse=False, 
        key=lambda x: int(x.stem)
    )
    checkpoint_path = checkpoint_path_list[-1]
    return checkpoint_path


def get_best_checkpoint_path(checkpoint_path_last, metric_for_select):
    checkpoint_dict_last = torch.load(str(checkpoint_path_last))
    best_checkpoint = np.argmin(checkpoint_dict_last[metric_for_select]) + 1
    filename_prev = checkpoint_path_last.stem + checkpoint_path_last.suffix
    filename_new = str(best_checkpoint) + checkpoint_path_last.suffix
    checkpoint_path = Path(str(checkpoint_path_last).replace(filename_prev, filename_new))
    return checkpoint_path


def get_result(result_dir):
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
    use_time_masking,
    module_is_fixed,
    use_jsut_corpus,
    use_jvs_corpus,
    lr,
    # beta_1,
    # beta_2,
    # weight_decay,
    # which_optim,
    # which_scheduler,
    # lr_decay_exp,
    # warmup_t_rate,
    # warmup_lr_init,
    # warmup_lr_min,
    max_epoch,
    avhubert_return_res_output,
    load_avhubert_pretrained_weight,
    avhubert_layer_loaded,
    use_avhubert_video_modality,
    use_avhubert_audio_modality,
    use_avhubert_encoder,
    check_point_start_separate_save_dir,
    start_ckpt_path_separate_save_dir,
    avhubert_encoder_embed_dim,
    avhubert_encoder_layers,
    avhubert_encoder_ffn_embed_dim,
    avhubert_encoder_attention_heads,
    debug,
    wandb_conf,
    subject,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec_avhubert',
            'train=nar',
            'test=nar',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            f'train.module_is_fixed={module_is_fixed}',
            f'train.use_jsut_corpus={use_jsut_corpus}',
            f'train.use_jvs_corpus={use_jvs_corpus}',
            f'train.lr={lr}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            # f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            # f'train.which_scheduler={which_scheduler}',
            # f'train.lr_decay_exp={lr_decay_exp}',
            # f'train.warmup_t_rate={warmup_t_rate}',
            # f'train.warmup_lr_init={warmup_lr_init}',
            # f'train.warmup_lr_min={warmup_lr_min}',
            f'train.max_epoch={max_epoch}',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            f'train.use_segment_masking={use_time_masking}',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'model.avhubert_return_res_output={avhubert_return_res_output}',
            f'model.load_avhubert_pretrained_weight={load_avhubert_pretrained_weight}',
            f'model.avhubert_layer_loaded={avhubert_layer_loaded}',
            f'model.use_avhubert_video_modality={use_avhubert_video_modality}',
            f'model.use_avhubert_audio_modality={use_avhubert_audio_modality}',
            f'model.use_avhubert_encoder={use_avhubert_encoder}',
            f'model.avhubert_config.base.encoder_embed_dim={avhubert_encoder_embed_dim}',
            f'model.avhubert_config.base.encoder_layers={avhubert_encoder_layers}',
            f'model.avhubert_config.base.encoder_ffn_embed_dim={avhubert_encoder_ffn_embed_dim}',
            f'model.avhubert_config.base.encoder_attention_heads={avhubert_encoder_attention_heads}',
        ],
        subject=subject,
        body=f'finish {run_filename_train}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec_avhubert',
            'train=nar',
            'test=nar',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            f'train.module_is_fixed={module_is_fixed}',
            f'train.use_jsut_corpus={use_jsut_corpus}',
            f'train.use_jvs_corpus={use_jvs_corpus}',
            f'train.lr={lr}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            # f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            # f'train.which_scheduler={which_scheduler}',
            # f'train.lr_decay_exp={lr_decay_exp}',
            # f'train.warmup_t_rate={warmup_t_rate}',
            # f'train.warmup_lr_init={warmup_lr_init}',
            # f'train.warmup_lr_min={warmup_lr_min}',
            f'train.max_epoch={max_epoch}',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            f'train.use_segment_masking={use_time_masking}',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'model.avhubert_return_res_output={avhubert_return_res_output}',
            f'model.avhubert_layer_loaded={avhubert_layer_loaded}',
            f'model.load_avhubert_pretrained_weight={load_avhubert_pretrained_weight}',
            f'model.use_avhubert_video_modality={use_avhubert_video_modality}',
            f'model.use_avhubert_audio_modality={use_avhubert_audio_modality}',
            f'model.use_avhubert_encoder={use_avhubert_encoder}',
            f'model.avhubert_config.base.encoder_embed_dim={avhubert_encoder_embed_dim}',
            f'model.avhubert_config.base.encoder_layers={avhubert_encoder_layers}',
            f'model.avhubert_config.base.encoder_ffn_embed_dim={avhubert_encoder_ffn_embed_dim}',
            f'model.avhubert_config.base.encoder_attention_heads={avhubert_encoder_attention_heads}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.debug={debug}',
        ],
        subject=subject,
        body=f'finish {run_filename_generate}'
    )
    send_email(subject=subject, body=get_result(result_dir))
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    clean_trash()
    return checkpoint_path_best


def main():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'
    # avhubert_layer_loaded_list = ['resnet', 'transformer', 'all']
    avhubert_layer_loaded_list = ['all']

    # 軽量モデル
    run_nar(
        checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        run_filename_train='train_nar_with_ex_avhubert.py',
        run_filename_generate='generate_nar_with_ex.py',
        metric_for_select='val_loss_list',
        use_spatial_aug=True,
        use_time_masking=True,
        module_is_fixed=[],
        use_jsut_corpus=False,
        use_jvs_corpus=False,
        lr=1.0e-3,
        max_epoch=200,
        avhubert_return_res_output=False,
        load_avhubert_pretrained_weight=False,
        avhubert_layer_loaded='',
        use_avhubert_video_modality=True,
        use_avhubert_audio_modality=False,
        use_avhubert_encoder=True,
        check_point_start_separate_save_dir=False,
        start_ckpt_path_separate_save_dir='',
        avhubert_encoder_embed_dim=256,
        avhubert_encoder_layers=6,
        avhubert_encoder_ffn_embed_dim=1024,
        avhubert_encoder_attention_heads=4,
        debug=debug,
        wandb_conf=wandb_conf,
        subject=subject,
    )

    # randomly initialized
    run_nar(
        checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        run_filename_train='train_nar_with_ex_avhubert.py',
        run_filename_generate='generate_nar_with_ex.py',
        metric_for_select='val_loss_list',
        use_spatial_aug=True,
        use_time_masking=True,
        module_is_fixed=[],
        use_jsut_corpus=False,
        use_jvs_corpus=False,
        lr=1.0e-3,
        max_epoch=200,
        avhubert_return_res_output=False,
        load_avhubert_pretrained_weight=False,
        avhubert_layer_loaded='',
        use_avhubert_video_modality=True,
        use_avhubert_audio_modality=False,
        use_avhubert_encoder=True,
        check_point_start_separate_save_dir=False,
        start_ckpt_path_separate_save_dir='',
        avhubert_encoder_embed_dim=768,
        avhubert_encoder_layers=12,
        avhubert_encoder_ffn_embed_dim=3072,
        avhubert_encoder_attention_heads=12,
        debug=debug,
        wandb_conf=wandb_conf,
        subject=subject,
    )

    for avhubert_layer_loaded in avhubert_layer_loaded_list:
        if avhubert_layer_loaded == 'resnet':
            module_is_fixed = ['avhubert_resnet']
        elif avhubert_layer_loaded == 'transformer':
            module_is_fixed = ['avhubert_transformer']
        elif avhubert_layer_loaded == 'all':
            module_is_fixed = ['avhubert']

        # load pretrained avhubert. learning randomly initialized layers only.
        checkpoint_path_best = run_nar(
            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            run_filename_train='train_nar_with_ex_avhubert.py',
            run_filename_generate='generate_nar_with_ex.py',
            metric_for_select='val_loss_list',
            use_spatial_aug=True,
            use_time_masking=True,
            module_is_fixed=module_is_fixed,
            use_jsut_corpus=False,
            use_jvs_corpus=False,
            lr=1.0e-3,
            max_epoch=100,
            avhubert_return_res_output=False,
            load_avhubert_pretrained_weight=True,
            avhubert_layer_loaded=avhubert_layer_loaded,
            use_avhubert_video_modality=True,
            use_avhubert_audio_modality=False,
            use_avhubert_encoder=True,
            check_point_start_separate_save_dir=False,
            start_ckpt_path_separate_save_dir='',
            avhubert_encoder_embed_dim=768,
            avhubert_encoder_layers=12,
            avhubert_encoder_ffn_embed_dim=3072,
            avhubert_encoder_attention_heads=12,
            debug=debug,
            wandb_conf=wandb_conf,
            subject=subject,
        )

        # load trained entire model. finetuning all layers.
        run_nar(
            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            run_filename_train='train_nar_with_ex_avhubert.py',
            run_filename_generate='generate_nar_with_ex.py',
            metric_for_select='val_loss_list',
            use_spatial_aug=True,
            use_time_masking=True,
            module_is_fixed=[],
            use_jsut_corpus=False,
            use_jvs_corpus=False,
            lr=1.0e-4,
            max_epoch=50,
            avhubert_return_res_output=False,
            load_avhubert_pretrained_weight=False,
            avhubert_layer_loaded='',
            use_avhubert_video_modality=True,
            use_avhubert_audio_modality=False,
            use_avhubert_encoder=True,
            check_point_start_separate_save_dir=True,
            start_ckpt_path_separate_save_dir=checkpoint_path_best,
            avhubert_encoder_embed_dim=768,
            avhubert_encoder_layers=12,
            avhubert_encoder_ffn_embed_dim=3072,
            avhubert_encoder_attention_heads=12,
            debug=debug,
            wandb_conf=wandb_conf,
            subject=subject,
        )

        # load pretrained avhubert. finetuning all layers.
        run_nar(
            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
            run_filename_train='train_nar_with_ex_avhubert.py',
            run_filename_generate='generate_nar_with_ex.py',
            metric_for_select='val_loss_list',
            use_spatial_aug=True,
            use_time_masking=True,
            module_is_fixed=[],
            use_jsut_corpus=False,
            use_jvs_corpus=False,
            lr=1.0e-3,
            max_epoch=150,
            avhubert_return_res_output=False,
            load_avhubert_pretrained_weight=True,
            avhubert_layer_loaded=avhubert_layer_loaded,
            use_avhubert_video_modality=True,
            use_avhubert_audio_modality=False,
            use_avhubert_encoder=True,
            check_point_start_separate_save_dir=False,
            start_ckpt_path_separate_save_dir='',
            avhubert_encoder_embed_dim=768,
            avhubert_encoder_layers=12,
            avhubert_encoder_ffn_embed_dim=3072,
            avhubert_encoder_attention_heads=12,
            debug=debug,
            wandb_conf=wandb_conf,
            subject=subject,
        )

    # audio experiments
    run_nar(
        checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
        run_filename_train='train_nar_with_ex_avhubert.py',
        run_filename_generate='generate_nar_with_ex.py',
        metric_for_select='val_loss_list',
        use_spatial_aug=False,
        use_time_masking=False,
        module_is_fixed=['avhubert'],
        use_jsut_corpus=False,
        use_jvs_corpus=False,
        lr=1.0e-3,
        max_epoch=100,
        avhubert_return_res_output=False,
        load_avhubert_pretrained_weight=True,
        avhubert_layer_loaded='all',
        use_avhubert_video_modality=False,
        use_avhubert_audio_modality=True,
        use_avhubert_encoder=True,
        check_point_start_separate_save_dir=False,
        start_ckpt_path_separate_save_dir='',
        avhubert_encoder_embed_dim=768,
        avhubert_encoder_layers=12,
        avhubert_encoder_ffn_embed_dim=3072,
        avhubert_encoder_attention_heads=12,
        debug=debug,
        wandb_conf=wandb_conf,
        subject=subject,
    )


if __name__ == '__main__':
    main()