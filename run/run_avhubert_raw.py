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
    '''
    checkpointが保存されるディレクトリから、一番最後の日付のところを取得
    '''
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
    '''
    ある日付のcheckpointディレクトリから、metric_for_selectの値が最も小さいファイルを取得
    '''
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
    corpus,
    speaker,
    which_decoder,
    lr,
    # beta_1,
    # beta_2,
    # weight_decay,
    # which_optim,
    which_scheduler,
    lr_decay_exp,
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
    avhubert_audio_pretrain,
    prompt_tuning,
    use_soft_prompt,
    n_prompt_tokens,
    use_prompt_block,
    prompt_block_se_r,
    debug,
    wandb_conf,
    subject,
    message,
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
            f'train.corpus={corpus}',
            f'train.speaker={speaker}',
            f'model.which_decoder={which_decoder}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            # f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            f'train.which_scheduler={which_scheduler}',
            f'train.lr_decay_exp={lr_decay_exp}',
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
            f'model.avhubert_audio_pretrain={avhubert_audio_pretrain}',
            f'model.prompt_tuning={prompt_tuning}',
            f'model.avhubert_config.base.use_soft_prompt={use_soft_prompt}',
            f'model.avhubert_config.base.n_prompt_tokens={n_prompt_tokens}',
            f'model.avhubert_config.base.use_prompt_block={use_prompt_block}',
            f'model.avhubert_config.base.prompt_block_se_r={prompt_block_se_r}',
        ],
        subject=subject,
        body=f'finish {run_filename_train}. {message}'
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
            f'train.corpus={corpus}',
            f'train.speaker={speaker}',
            f'model.which_decoder={which_decoder}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            # f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            f'train.which_scheduler={which_scheduler}',
            f'train.lr_decay_exp={lr_decay_exp}',
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
            f'model.avhubert_audio_pretrain={avhubert_audio_pretrain}',
            f'model.prompt_tuning={prompt_tuning}',
            f'model.avhubert_config.base.use_soft_prompt={use_soft_prompt}',
            f'model.avhubert_config.base.n_prompt_tokens={n_prompt_tokens}',
            f'model.avhubert_config.base.use_prompt_block={use_prompt_block}',
            f'model.avhubert_config.base.prompt_block_se_r={prompt_block_se_r}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.speaker={speaker}',
            f'test.debug={debug}',
        ],
        subject=subject,
        body=f'finish {run_filename_generate}. {message}'
    )
    send_email(subject=subject, body=get_result(result_dir))
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    clean_trash()
    return checkpoint_path_best


def run_ar(
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
    corpus,
    speaker,
    which_decoder,
    lr,
    # beta_1,
    # beta_2,
    weight_decay,
    # which_optim,
    which_scheduler,
    lr_decay_exp,
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
    training_method,
    avhubert_encoder_embed_dim,
    avhubert_encoder_layers,
    avhubert_encoder_ffn_embed_dim,
    avhubert_encoder_attention_heads,
    avhubert_audio_pretrain,
    prenet_dropout,
    prenet_inner_channels,
    use_attention,
    taco_dec_conv_kernel_size,
    debug,
    wandb_conf,
    subject,
    message,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_train}',
            'model=mspec_avhubert',
            'train=ar',
            'test=ar',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            f'train.module_is_fixed={module_is_fixed}',
            f'train.use_jsut_corpus={use_jsut_corpus}',
            f'train.use_jvs_corpus={use_jvs_corpus}',
            f'train.lr={lr}',
            f'train.corpus={corpus}',
            f'train.speaker={speaker}',
            f'model.which_decoder={which_decoder}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            f'train.which_scheduler={which_scheduler}',
            f'train.lr_decay_exp={lr_decay_exp}',
            # f'train.warmup_t_rate={warmup_t_rate}',
            # f'train.warmup_lr_init={warmup_lr_init}',
            # f'train.warmup_lr_min={warmup_lr_min}',
            f'train.max_epoch={max_epoch}',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            f'train.use_segment_masking={use_time_masking}',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'train.training_method={training_method}',
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
            f'model.avhubert_audio_pretrain={avhubert_audio_pretrain}',
            f'model.taco_lip_prenet_dropout={prenet_dropout}',
            f'model.taco_dec_prenet_inner_channels={prenet_inner_channels}',
            f'model.taco_use_attention={use_attention}',
            f'model.taco_dec_conv_kernel_size={taco_dec_conv_kernel_size}',
        ],
        subject=subject,
        body=f'finish {run_filename_train}. {message}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/{run_filename_generate}',
            'model=mspec_avhubert',
            'train=ar',
            'test=ar',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            f'train.module_is_fixed={module_is_fixed}',
            f'train.use_jsut_corpus={use_jsut_corpus}',
            f'train.use_jvs_corpus={use_jvs_corpus}',
            f'train.lr={lr}',
            f'train.corpus={corpus}',
            f'train.speaker={speaker}',
            f'model.which_decoder={which_decoder}',
            # f'train.beta_1={beta_1}',
            # f'train.beta_2={beta_2}',
            f'train.weight_decay={weight_decay}',
            # f'train.which_optim={which_optim}',
            f'train.which_scheduler={which_scheduler}',
            f'train.lr_decay_exp={lr_decay_exp}',
            # f'train.warmup_t_rate={warmup_t_rate}',
            # f'train.warmup_lr_init={warmup_lr_init}',
            # f'train.warmup_lr_min={warmup_lr_min}',
            f'train.max_epoch={max_epoch}',
            f'train.use_horizontal_flip={use_spatial_aug}',
            f'train.use_random_crop={use_spatial_aug}',
            f'train.use_segment_masking={use_time_masking}',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'train.training_method={training_method}',
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
            f'model.avhubert_audio_pretrain={avhubert_audio_pretrain}',
            f'model.taco_lip_prenet_dropout={prenet_dropout}',
            f'model.taco_dec_prenet_inner_channels={prenet_inner_channels}',
            f'model.taco_use_attention={use_attention}',
            f'model.taco_dec_conv_kernel_size={taco_dec_conv_kernel_size}',
            f'test.model_path={get_last_checkpoint_path(checkpoint_dir)}',
            f'test.metric_for_select={metric_for_select}',
            f'test.speaker={speaker}',
            f'test.debug={debug}',
        ],
        subject=subject,
        body=f'finish {run_filename_generate}. {message}'
    )
    send_email(subject=subject, body=get_result(result_dir))
    checkpoint_path_last = get_last_checkpoint_path(checkpoint_dir)
    checkpoint_path_best = get_best_checkpoint_path(checkpoint_path_last, metric_for_select)
    clean_trash()
    return checkpoint_path_best


def experiments_avhubert_transferability():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    training_param_list = [
        {
            'which_scheduler': 'warmup',
            'lr_decay_exp': 0.98,
            'max_epoch': 30,
        },
    ]
    avhubert_layer_loaded_list = ['all']
    data_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
    ]
    which_decoder_list = [
        'restc',
    ]
    for training_param in training_param_list:
        for data in data_list:
            for which_decoder in which_decoder_list:
                # randomly initialized lightweight model
                run_nar(
                    checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                    result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                    run_filename_train='train_nar_with_ex_avhubert_raw.py',
                    run_filename_generate='generate_nar_with_ex_raw.py',
                    metric_for_select='val_loss_list',
                    use_spatial_aug=True,
                    use_time_masking=True,
                    module_is_fixed=[],
                    use_jsut_corpus=False,
                    use_jvs_corpus=False,
                    lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                    corpus=data['corpus'],
                    speaker=data['speaker'],
                    which_decoder=which_decoder,
                    which_scheduler=training_param['which_scheduler'],
                    lr_decay_exp=training_param['lr_decay_exp'],
                    max_epoch=training_param['max_epoch'],
                    avhubert_return_res_output=False,
                    load_avhubert_pretrained_weight=False,
                    avhubert_layer_loaded='',
                    use_avhubert_video_modality=True,
                    use_avhubert_audio_modality=False,
                    use_avhubert_encoder=False,
                    check_point_start_separate_save_dir=False,
                    start_ckpt_path_separate_save_dir='',
                    avhubert_encoder_embed_dim=768,
                    avhubert_encoder_layers=12,
                    avhubert_encoder_ffn_embed_dim=3072,
                    avhubert_encoder_attention_heads=12,
                    avhubert_audio_pretrain=False,
                    debug=debug,
                    wandb_conf=wandb_conf,
                    subject=subject,
                    message='randomly initialized lightweight model.'
                )
                
                # randomly initialized avhubert
                run_nar(
                    checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                    result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                    run_filename_train='train_nar_with_ex_avhubert_raw.py',
                    run_filename_generate='generate_nar_with_ex_raw.py',
                    metric_for_select='val_loss_list',
                    use_spatial_aug=True,
                    use_time_masking=True,
                    module_is_fixed=[],
                    use_jsut_corpus=False,
                    use_jvs_corpus=False,
                    lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                    corpus=data['corpus'],
                    speaker=data['speaker'],
                    which_decoder=which_decoder,
                    which_scheduler=training_param['which_scheduler'],
                    lr_decay_exp=training_param['lr_decay_exp'],
                    max_epoch=training_param['max_epoch'],
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
                    avhubert_audio_pretrain=False,
                    debug=debug,
                    wandb_conf=wandb_conf,
                    subject=subject,
                    message='randomly initialzed avhubert.'
                )

                for avhubert_layer_loaded in avhubert_layer_loaded_list:
                    if avhubert_layer_loaded == 'resnet':
                        module_is_fixed = ['avhubert_resnet']
                    elif avhubert_layer_loaded == 'transformer':
                        module_is_fixed = ['avhubert_transformer']
                    elif avhubert_layer_loaded == 'all':
                        module_is_fixed = ['avhubert']

                    # load pretrained avhubert. training only decoder.
                    run_nar(
                        checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                        result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                        run_filename_train='train_nar_with_ex_avhubert_raw.py',
                        run_filename_generate='generate_nar_with_ex_raw.py',
                        metric_for_select='val_loss_list',
                        use_spatial_aug=True,
                        use_time_masking=True,
                        module_is_fixed=module_is_fixed,
                        use_jsut_corpus=False,
                        use_jvs_corpus=False,
                        lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                        corpus=data['corpus'],
                        speaker=data['speaker'],
                        which_decoder=which_decoder,
                        which_scheduler=training_param['which_scheduler'],
                        lr_decay_exp=training_param['lr_decay_exp'],
                        max_epoch=training_param['max_epoch'],
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
                        avhubert_audio_pretrain=False,
                        debug=debug,
                        wandb_conf=wandb_conf,
                        subject=subject,
                        message='load pretrained avhubert. training only decoder.'
                    )

                    # load pretrained avhubert. finetuning all layers.
                    run_nar(
                        checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                        result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                        run_filename_train='train_nar_with_ex_avhubert_raw.py',
                        run_filename_generate='generate_nar_with_ex_raw.py',
                        metric_for_select='val_loss_list',
                        use_spatial_aug=True,
                        use_time_masking=True,
                        module_is_fixed=[],
                        use_jsut_corpus=False,
                        use_jvs_corpus=False,
                        lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                        corpus=data['corpus'],
                        speaker=data['speaker'],
                        which_decoder=which_decoder,
                        which_scheduler=training_param['which_scheduler'],
                        lr_decay_exp=training_param['lr_decay_exp'],
                        max_epoch=training_param['max_epoch'],
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
                        avhubert_audio_pretrain=False,
                        debug=debug,
                        wandb_conf=wandb_conf,
                        subject=subject,
                        message='load pretrained avhubert, finetuning all layers.'
                    )


def experiments_prompt_tuning():
    debug = False
    wandb_conf = 'debug' if debug else 'nar'
    subject = 'プログラム経過'

    training_param_list = [
        {
            'which_scheduler': 'warmup',
            'lr_decay_exp': 0.98,
            'max_epoch': 30,
        },
    ]
    data_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
    ]
    which_decoder_list = [
        'restc',
    ]
    n_prompt_tokens_list = [
        10,
        25,
        50,
    ]
    prompt_block_se_r_list = [
        16,
    ]
    for training_param in training_param_list:
        for data in data_list:
            for which_decoder in which_decoder_list:
                for n_prompt_tokens in n_prompt_tokens_list:
                    for prompt_block_se_r in prompt_block_se_r_list:
                        check_point_path_best = run_nar(
                            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                            run_filename_train='train_nar_with_ex_avhubert_raw.py',
                            run_filename_generate='generate_nar_with_ex_raw.py',
                            metric_for_select='val_loss_list',
                            use_spatial_aug=True,
                            use_time_masking=True,
                            module_is_fixed=[],
                            use_jsut_corpus=False,
                            use_jvs_corpus=False,
                            lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                            corpus=data['corpus'],
                            speaker=data['speaker'],
                            which_decoder=which_decoder,
                            which_scheduler=training_param['which_scheduler'],
                            lr_decay_exp=training_param['lr_decay_exp'],
                            max_epoch=training_param['max_epoch'],
                            avhubert_return_res_output=False,
                            load_avhubert_pretrained_weight=True,
                            avhubert_layer_loaded='all',
                            use_avhubert_video_modality=True,
                            use_avhubert_audio_modality=False,
                            use_avhubert_encoder=True,
                            check_point_start_separate_save_dir=False,
                            start_ckpt_path_separate_save_dir='',
                            avhubert_encoder_embed_dim=768,
                            avhubert_encoder_layers=12,
                            avhubert_encoder_ffn_embed_dim=3072,
                            avhubert_encoder_attention_heads=12,
                            avhubert_audio_pretrain=False,
                            prompt_tuning=True,
                            use_soft_prompt=True,
                            n_prompt_tokens=n_prompt_tokens,
                            use_prompt_block=True,
                            prompt_block_se_r=prompt_block_se_r,
                            debug=debug,
                            wandb_conf=wandb_conf,
                            subject=subject,
                            message='prompt tuning.'
                        )
                        run_nar(
                            checkpoint_dir=Path('~/lip2sp_pytorch/check_point/nar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                            result_dir=Path('~/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                            run_filename_train='train_nar_with_ex_avhubert_raw.py',
                            run_filename_generate='generate_nar_with_ex_raw.py',
                            metric_for_select='val_loss_list',
                            use_spatial_aug=True,
                            use_time_masking=True,
                            module_is_fixed=[],
                            use_jsut_corpus=False,
                            use_jvs_corpus=False,
                            lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                            corpus=data['corpus'],
                            speaker=data['speaker'],
                            which_decoder=which_decoder,
                            which_scheduler=training_param['which_scheduler'],
                            lr_decay_exp=training_param['lr_decay_exp'],
                            max_epoch=training_param['max_epoch'],
                            avhubert_return_res_output=False,
                            load_avhubert_pretrained_weight=True,
                            avhubert_layer_loaded='all',
                            use_avhubert_video_modality=True,
                            use_avhubert_audio_modality=False,
                            use_avhubert_encoder=True,
                            check_point_start_separate_save_dir=True,
                            start_ckpt_path_separate_save_dir=check_point_path_best,
                            avhubert_encoder_embed_dim=768,
                            avhubert_encoder_layers=12,
                            avhubert_encoder_ffn_embed_dim=3072,
                            avhubert_encoder_attention_heads=12,
                            avhubert_audio_pretrain=False,
                            prompt_tuning=False,
                            use_soft_prompt=True,
                            n_prompt_tokens=n_prompt_tokens,
                            use_prompt_block=True,
                            prompt_block_se_r=prompt_block_se_r,
                            debug=debug,
                            wandb_conf=wandb_conf,
                            subject=subject,
                            message='load prompt tuned avhubert, finetuning all layers.'
                        )
            


def experiments_ar_decoder():
    debug = False
    wandb_conf = 'debug' if debug else 'ar'
    subject = 'プログラム経過'

    training_param_list = [
        {
            'which_scheduler': 'warmup',
            'lr_decay_exp': 0.98,
            'max_epoch': 200,
        },
    ]
    avhubert_layer_loaded_list = ['all']
    data_list = [
        {
            'corpus': ['ATR'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
        {
            'corpus': ['ATR', 'BASIC5000'],
            'speaker': ["F01_kablab", "M01_kablab"],
        },
    ]
    which_decoder_list = [
        'tacotron',
    ]
    prenet_dropout_list = [
        # 0.25,
        0.5,
        # 0.75,
    ]
    prenet_inner_channels_list = [
        # 32,
        # 64,
        # 128,
        256,
    ]
    training_method_list = [
        'teacher_forcing',
        # 'scheduled_sampling',
    ]
    use_attention_list = [True]
    taco_dec_conv_kernel_size_list = [75, 125]
    weight_decay_list = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
    for training_param in training_param_list:
        for data in data_list:
            for which_decoder in which_decoder_list:
                for avhubert_layer_loaded in avhubert_layer_loaded_list:
                    for prenet_dropout in prenet_dropout_list:
                        for prenet_inner_channels in prenet_inner_channels_list:
                            for training_method in training_method_list:
                                for use_attention in use_attention_list:
                                    for taco_dec_conv_kernel_size in taco_dec_conv_kernel_size_list:
                                        for weight_decay in weight_decay_list:
                                            # load pretrained avhubert. finetuning all layers.
                                            run_ar(
                                                checkpoint_dir=Path('~/lip2sp_pytorch/check_point/ar/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                                                result_dir=Path('~/lip2sp_pytorch/result/ar/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(),
                                                run_filename_train='train_ar_with_ex_avhubert_raw.py',
                                                run_filename_generate='generate_ar_with_ex_raw.py',
                                                metric_for_select='val_loss_list',
                                                use_spatial_aug=True,
                                                use_time_masking=True,
                                                module_is_fixed=[],
                                                use_jsut_corpus=False,
                                                use_jvs_corpus=False,
                                                lr=1.0e-4 if which_decoder == 'transformer' else 1.0e-3,
                                                weight_decay=weight_decay,
                                                corpus=data['corpus'],
                                                speaker=data['speaker'],
                                                which_decoder=which_decoder,
                                                which_scheduler=training_param['which_scheduler'],
                                                lr_decay_exp=training_param['lr_decay_exp'],
                                                max_epoch=training_param['max_epoch'],
                                                avhubert_return_res_output=False,
                                                load_avhubert_pretrained_weight=True,
                                                avhubert_layer_loaded=avhubert_layer_loaded,
                                                use_avhubert_video_modality=True,
                                                use_avhubert_audio_modality=False,
                                                use_avhubert_encoder=True,
                                                check_point_start_separate_save_dir=False,
                                                start_ckpt_path_separate_save_dir='',
                                                training_method=training_method,
                                                avhubert_encoder_embed_dim=768,
                                                avhubert_encoder_layers=12,
                                                avhubert_encoder_ffn_embed_dim=3072,
                                                avhubert_encoder_attention_heads=12,
                                                avhubert_audio_pretrain=False,
                                                prenet_dropout=prenet_dropout,
                                                prenet_inner_channels=prenet_inner_channels,
                                                use_attention=use_attention,
                                                taco_dec_conv_kernel_size=taco_dec_conv_kernel_size,
                                                debug=debug,
                                                wandb_conf=wandb_conf,
                                                subject=subject,
                                                message='load pretrained avhubert, finetuning all layers.'
                                            )


def main():
    # experiments_avhubert_transferability()
    # experiments_ar_decoder()
    experiments_prompt_tuning()


if __name__ == '__main__':
    main()