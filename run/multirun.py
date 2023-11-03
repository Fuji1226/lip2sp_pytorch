import subprocess


def main():
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/run/run_avhubert_raw.py',
        ]
    )
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/run/run_pwg.py',
        ]
    )


if __name__ == '__main__':
    main()