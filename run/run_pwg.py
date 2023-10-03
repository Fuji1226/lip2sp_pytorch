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
    result_path = result_dir / 'accuracy_pwg.txt'
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


def run_pwg(
    checkpoint_dir, 
    result_dir, 
    run_filename_train, 
    run_filename_generate,
    metric_for_select,
    check_point_start_separate_save_dir,
    start_ckpt_path_separate_save_dir,
    lr_gen,
    lr_disc,
    which_scheduler,
    max_epoch,
    debug,
    wandb_conf,
    subject,
):
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/parallelwavegan/{run_filename_train}',
            'model=mspec_avhubert',
            'train=default',
            'test=default',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=False',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'train.lr_gen={lr_gen}',
            f'train.lr_disc={lr_disc}',
            f'train.which_scheduler={which_scheduler}',
            f'train.max_epoch={max_epoch}',
        ],
        subject=subject,
        body=f'finish {run_filename_train}'
    )
    run_program(
        script=[
            'python',
            f'/home/minami/lip2sp_pytorch/parallelwavegan/{run_filename_generate}',
            'model=mspec_avhubert',
            'train=default',
            'test=default',
            f'wandb_conf={wandb_conf}',
            f'train.debug={debug}',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=False',
            f'train.check_point_start_separate_save_dir={check_point_start_separate_save_dir}',
            f'train.start_ckpt_path_separate_save_dir={start_ckpt_path_separate_save_dir}',
            f'train.lr_gen={lr_gen}',
            f'train.lr_disc={lr_disc}',
            f'train.which_scheduler={which_scheduler}',
            f'train.max_epoch={max_epoch}',
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
    return checkpoint_path_best


def main():
    debug = False
    wandb_conf = 'debug' if debug else 'default'
    subject = 'プログラム経過'

    checkpoint_path_best = run_pwg(
        checkpoint_dir=Path('~/lip2sp_pytorch/parallelwavegan/check_point/default/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(), 
        result_dir=Path('~/lip2sp_pytorch/parallelwavegan/result/default/generate/avhubert_preprocess_fps25_gray/mspec_avhubert').expanduser(), 
        run_filename_train='pwg_train.py', 
        run_filename_generate='pwg_generate.py',
        metric_for_select='val_epoch_loss_gen_all_list',
        check_point_start_separate_save_dir=False,
        start_ckpt_path_separate_save_dir='',
        lr_gen=1.0e-3,
        lr_disc=1.0e-3,
        which_scheduler='exp',
        max_epoch=100,
        debug=debug,
        wandb_conf=wandb_conf,
        subject=subject,
    )

    # run_pwg(
    #     checkpoint_dir=Path('~/lip2sp_pytorch/parallelwavegan/check_point/default/avhubert_preprocess_fps25_gray').expanduser(), 
    #     result_dir=Path('~/lip2sp_pytorch/parallelwavegan/result/default/generate/avhubert_preprocess_fps25_gray').expanduser(), 
    #     run_filename_train='pwg_train.py', 
    #     run_filename_generate='pwg_generate.py',
    #     metric_for_select='val_epoch_loss_gen_all_list',
    #     check_point_start_separate_save_dir=True,
    #     start_ckpt_path_separate_save_dir=checkpoint_path_best,
    #     lr_gen=1.0e-4,
    #     lr_disc=1.0e-4,
    #     which_scheduler='warmup',
    #     max_epoch=50,
    #     debug=debug,
    #     wandb_conf=wandb_conf,
    #     subject=subject,
    # )



if __name__ == '__main__':
    main()