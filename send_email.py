import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess


def send_email(subject, body, to_email, from_email, password):
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
    gmail_username = "tmnm13009@gmail.com"
    gmail_password = "igumvwzbztowbigt"  # googleアカウントでアプリパスワードを取得し、それを利用する
    to_email = "tmnm13009@gmail.com"

    result = subprocess.run(script, capture_output=False, text=False)
    send_email(subject, body, to_email, gmail_username, gmail_password)


def main():
    run_program(
        script=[
            'python', 'train_audio_ae.py',
            'model=mspec80',
            'train=audio_ae',
            'test=audio_ae',
            'wandb_conf=audio_ae',
            'train.debug=False',
            'train.module_is_fixed=["lip_encoder"]',
            'train.use_jsut_corpus=True',
            'train.use_jvs_corpus=True',
            'train.apply_upsampling=False',
            'train.lr=1.0e-3',
            'train.lr_decay_exp=0.92',
            'train.max_epoch=50',
            'train.pretrained_model_path=""',
            'model.ae_emb_dim=32',
        ],
        subject='プログラム経過',
        body='finish train_audio_ae.py',
    )


if __name__ == '__main__':
    main()