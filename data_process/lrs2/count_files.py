from pathlib import Path
import os

data_dir = Path("~/lrs2").expanduser()


def main():
    print("count files")
    landmark_dir = data_dir / "landmark"

    file_list = []
    for curdir, dirs, files in os.walk(landmark_dir):
        for file in files:
            file = Path(file)
            file_list.append(file)

    print(len(file_list))


if __name__ == "__main__":
    main()