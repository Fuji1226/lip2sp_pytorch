'''
itoからローカルにデータをダウンロードするときに使用する
itoのbashを起動したときにito_shardが実行されるとバグるので、実行前に.bashrcでコメントアウトしておく
'''
import os
import sys
from pathlib import Path
import paramiko
import scp
import argparse
import random
from tqdm import tqdm


def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )


def main():
    hostname = 'ito'

    # ito上のダウンロードしたいファイル（ディレクトリ）までの絶対パスを指定。
    # 何もない状態からの場合、一旦ここのリストに入れているものを全てダウンロードしてください。
    # /home/usr4/r70264cのところは人によって異なるので、変更お願いします。
    from_folder_list = [
        Path("/home/usr4/r70264c/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/train/F01_kablab/mspec80"),
        Path("/home/usr4/r70264c/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/val/F01_kablab/mspec80"),
        Path("/home/usr4/r70264c/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/test/F01_kablab/mspec80"),
        Path("/home/usr4/r70264c/dataset/lip/emb/F01_kablab"),
        Path("/home/usr4/r70264c/dataset/lip/utt_small"),
        Path("/home/usr4/r70264c/lip2sp_pytorch/csv"),
    ]
    for from_folder in from_folder_list:
        home_dir_remote = list(from_folder.parents)[-4]
        relative_path = from_folder.relative_to(home_dir_remote)
        to_folder = Path.home() / relative_path
        num_files_to_download = 10000     # ダウンロードしたいファイル数の上限を設定

        to_folder.mkdir(parents=True, exist_ok=True)

        config_file = os.path.expanduser('~/.ssh/config')
        ssh_config = paramiko.SSHConfig()
        ssh_config.parse(open(config_file, 'r'))
        config = ssh_config.lookup(hostname)

        with paramiko.SSHClient() as ssh:
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(
                hostname=config['hostname'],
                username=config['user'],
                key_filename=config['identityfile'],
            )

            with ssh.open_sftp() as sftp:
                remote_files = sftp.listdir(str(from_folder))

                # num_files_to_downloadを超えるときはランダムに取得して制限
                if len(remote_files) > num_files_to_download:
                    selected_files = random.sample(remote_files, num_files_to_download)
                else:
                    selected_files = remote_files

                with scp.SCPClient(ssh.get_transport(), progress=None) as scpc:
                    for filename in tqdm(selected_files):
                        remote_path = from_folder / filename
                        local_path = to_folder / filename
                        scpc.get(
                            remote_path=str(remote_path),
                            local_path=str(local_path),
                            preserve_times=False,
                        )


if __name__ == '__main__':
    main()