import subprocess


def run_program(
    toFolder,
    fromFolder,
):
    subprocess.run(
        [
            'python',
            '/home/minami/lip2sp_pytorch/data_process/download_from_ito.py',
            '--toFolder',
            toFolder,
            '--fromFolder',
            fromFolder,
        ]
    )
    

def main():
    speaker_list = ['F01_kablab', 'F02_kablab', 'M01_kablab', 'M04_kablab']
    for speaker in speaker_list:
        run_program(
            toFolder='/home/minami/dataset/lip/cropped_fps25',
            fromFolder=f'/home/usr4/r70264c/dataset/lip/cropped_fps25/{speaker}',
        )
        run_program(
            toFolder='/home/minami/dataset/lip/landmark_fps25',
            fromFolder=f'/home/usr4/r70264c/dataset/lip/landmark_fps25/{speaker}',
        )


if __name__ == '__main__':
    main()