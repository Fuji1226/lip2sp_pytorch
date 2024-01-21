import subprocess
from pathlib import Path
import os


def main():
    subprocess.run(
        [
            "yt-dlp",
            "-f",
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "-o",
            "./videos/%(title)s.%(ext)s",
            "https://www.youtube.com/watch?v=0x8vuTrq3jc",
        ]
    )
    data_path_list = Path("/Users/minami/class/research/for_python/videos").glob("*.mp4")
    for data_path in data_path_list:
        save_path = Path(str(data_path).replace(data_path.stem, data_path.stem + "_fps25"))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(data_path),
                "-r",
                "25",
                "-c:a",
                "copy",
                str(save_path),
            ]
        )
        os.remove(str(data_path))
        os.rename(str(save_path), str(data_path))
        
    

if __name__ == '__main__':
    main()