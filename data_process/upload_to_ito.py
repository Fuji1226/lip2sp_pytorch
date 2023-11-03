'''
ローカルからitoへのデータのアップロード
itoのbashを起動したときにito_shardが実行されるとバグるので、.bashrcでコメントアウトしておく
'''

import os
import sys
import paramiko
import scp
import argparse


def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )


def main():
    # argparser = argparse.ArgumentParser()
    # argparser.add_argument('--toFolder')
    # argparser.add_argument('--fromFolder')
    # args = argparser.parse_args()
    # to_folder = args.toFolder 
    # from_folder = args.fromFolder

    hostname = 'ito'
    to_folder = '/home/usr4/r70264c/dataset/lip'
    from_folder = '/home/minami/dataset/lip/avhubert_feature_ja_2023:10:12_19-34-33'

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

        with scp.SCPClient(ssh.get_transport(), progress=progress) as scpc:
            scpc.put(
                files=from_folder,
                remote_path=to_folder,
                recursive=True,
                preserve_times=False,
            )


if __name__ == '__main__':
    main()