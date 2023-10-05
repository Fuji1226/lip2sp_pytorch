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


def download(
    from_folder,
    hostname,
    num_files_to_download,
):
    home_dir_remote = list(from_folder.parents)[-4]
    relative_path = from_folder.relative_to(home_dir_remote)
    to_folder = Path.home() / relative_path
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
            remote_file_list = sftp.listdir(str(from_folder))
            if len(remote_file_list) > num_files_to_download:
                selected_file_list = random.sample(remote_file_list, num_files_to_download)
            else:
                selected_file_list = remote_file_list

            with scp.SCPClient(ssh.get_transport(), progress=None) as scpc:
                for selected_file in tqdm(selected_file_list):
                    remote_path = from_folder / selected_file
                    local_path = to_folder / selected_file
                    scpc.get(
                        remote_path=str(remote_path),
                        local_path=str(local_path),
                        preserve_times=False,
                    )


def download_utt_files(
    from_folder,
    hostname,
    local_data_dir,
):
    home_dir_remote = list(from_folder.parents)[-4]
    relative_path = from_folder.relative_to(home_dir_remote)
    to_folder = Path.home() / relative_path
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
            remote_file_list = sftp.listdir(str(from_folder))
            remote_file_list = [f.split(".")[0] for f in remote_file_list]
            local_file_list = list(local_data_dir.glob("**/*.npz"))
            local_file_list = [f.stem for f in local_file_list]
            required_file_list = list(set(remote_file_list) & set(local_file_list))
            required_file_list = [f + ".txt" for f in required_file_list]

            with scp.SCPClient(ssh.get_transport(), progress=None) as scpc:
                for required_file in tqdm(required_file_list):
                    remote_path = from_folder / required_file
                    local_path = to_folder / required_file
                    scpc.get(
                        remote_path=str(remote_path),
                        local_path=str(local_path),
                        preserve_times=False,
                    )    


def main():
    hostname = 'ito'

    # ito上のダウンロードしたいファイル（ディレクトリ）までの絶対パスを指定。
    # 何もない状態からの場合、一旦ここのリストに入れているものを全てダウンロードしてください。
    # ito_usr_pathは人によって異なるので、変更お願いします。
    ito_usr_path = "/home/usr4/r70264c"
    from_folder_list = [
        Path(f"{ito_usr_path}/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/train/F01_kablab/mspec80"),
        Path(f"{ito_usr_path}/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/val/F01_kablab/mspec80"),
        Path(f"{ito_usr_path}/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray/test/F01_kablab/mspec80"),
        Path(f"{ito_usr_path}/dataset/lip/emb/F01_kablab"),
        Path(f"{ito_usr_path}/lip2sp_pytorch/csv"),
    ]
    for from_folder in from_folder_list:
        download(
            from_folder=from_folder,
            hostname=hostname,
            num_files_to_download=100,  # ダウンロードするファイル数
        )

    download_utt_files(
        from_folder=Path(f"{ito_usr_path}/dataset/lip/utt_small"),
        hostname=hostname,
        local_data_dir=Path("~/dataset/lip/np_files/face_cropped_max_size_fps25_0_25_gray").expanduser(),
    )
    


if __name__ == '__main__':
    main()