import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import subprocess


def send_email(subject, body, to_email, from_email, password):
    """Gmailを使ってメールを送信する関数"""
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
    gmail_username = "tmnm13009@gmail.com"  # Gmailのユーザー名を入力
    gmail_password = "igumvwzbztowbigt"  # googleアカウントでアプリパスワードを取得し、それを利用する
    to_email = "tmnm13009@gmail.com"  # 送信先のメールアドレスを入力

    result = subprocess.run(script, capture_output=False, text=False)
    stdout_content = result.stdout
    # body += stdout_content
    send_email(subject, body, to_email, gmail_username, gmail_password)


def main():
    debug = False

    run_program(
        script=[
            'python', '/home/minami/lip2sp_pytorch/data_process/upload_to_ito.py', 
            '--toFolder', '/home/usr4/r70264c/Lip2Wav',
            '--fromFolder', '/home/minami/Lip2Wav/Dataset_split_wide_fps25',
        ],
        subject='itoへのアップロード経過',
        body='finish Dataset_split_wide_fps25',
    )
    run_program(
        script=[
            'python', '/home/minami/lip2sp_pytorch/data_process/upload_to_ito.py', 
            '--toFolder', '/home/usr4/r70264c/Lip2Wav',
            '--fromFolder', '/home/minami/Lip2Wav/bbox_split_wide_fps25',
        ],
        subject='itoへのアップロード経過',
        body='finish bbox_split_wide_fps25',
    )
    run_program(
        script=[
            'python', '/home/minami/lip2sp_pytorch/data_process/upload_to_ito.py', 
            '--toFolder', '/home/usr4/r70264c/Lip2Wav',
            '--fromFolder', '/home/minami/Lip2Wav/landmark_split_wide_fps25',
        ],
        subject='itoへのアップロード経過',
        body='finish landmark_split_wide_fps25',
    )

    # if debug:
    #     interval_seconds = 10
    #     schedule.every(interval_seconds).seconds.do(run_program)
    # else:
    #     interval_hours = 1
    #     schedule.every(interval_hours).hours.do(run_program)

    # while True:
    #     schedule.run_pending()
    #     time.sleep(1)


if __name__ == "__main__":
    main()