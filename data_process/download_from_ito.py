'''
itoからローカルにデータをダウンロードするときに使用する
itoのbashを起動したときにito_shardが実行されるとバグるので、.bashrcでコメントアウトしておく
'''

import os
import sys
import subprocess
import glob
import paramiko
import scp
import argparse


def progress(filename, size, sent):
    sys.stdout.write("%s's progress: %.2f%%   \r" % (filename, float(sent)/float(size)*100) )


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--toFolder')
    argparser.add_argument('--fromFolder')
    args = argparser.parse_args()

    hostname = 'ito'
    to_folder = args.toFolder
    from_folder = args.fromFolder

    print(f'to_folder = {to_folder}')
    print(f'from_folder = {from_folder}')

    os.makedirs(to_folder, exist_ok=True)

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
            scpc.get(
                remote_path=from_folder,
                local_path=to_folder,
                recursive=True,
                preserve_times=False,
            )


if __name__ == '__main__':
    main()