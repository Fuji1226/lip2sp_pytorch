import subprocess


def main():
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/data_process/tcd_timit/save_landmark.py',
        ]
    )
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/data_process/tcd_timit/avhubert_crop.py',
        ]
    )


if __name__ == '__main__':
    main()