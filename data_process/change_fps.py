import os
from pathlib import Path
from subprocess import run
from tqdm import tqdm


fps_list = [25,]
debug = False

dir_name = "face_cropped_nn"
orig_margin = 0
orig_fps = 50

def main():
    speaker_list = ["F01_kablab", "F02_kablab", "M01_kablab", "M04_kablab"]

    for speaker in speaker_list:
        data_dir = Path(f"~/dataset/lip/{dir_name}_{orig_margin}_{orig_fps}").expanduser()
        data_dir = data_dir / speaker

        for fps in fps_list:
            video_path = sorted(list(data_dir.glob("*.mp4")))

            print(f"fps = {fps}")

            for path in tqdm(video_path):
                save_dir = Path(f"~/dataset/lip/{dir_name}_{orig_margin}_{fps}").expanduser()
                save_dir = save_dir / speaker
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