import os
from pathlib import Path
from subprocess import run
from tqdm import tqdm


fps_list = [25,]
debug = False

dir_name = "mov"
#? '/home/user/dataset/kab2022/mov'にある動画のfpsを変更


def main():
    #speaker_list = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab", "F01_kablab_20220930", "F01_kablab_all"]
    speaker_list = ["kab2022"]
    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        data_dir = Path(f"~/dataset/{speaker}/{dir_name}").expanduser()

        for fps in fps_list:
            video_path = sorted(list(data_dir.glob("*.mp4")))

            print(f"fps = {fps}")

            for path in video_path:
                save_dir = Path(f"~/dataset/{speaker}/{dir_name}_fps{fps}").expanduser()
                os.makedirs(save_dir, exist_ok=True)
                save_path = save_dir / f"{path.stem}.mp4"

                if save_path.exists():
                    continue

                try:
                    cmd = f"ffmpeg -i {str(path)} -r {fps} {str(save_path)}"
                    run(cmd, shell=True, check=True)
                except:
                    continue

                if debug:
                    break


if __name__ == "__main__":
    main()