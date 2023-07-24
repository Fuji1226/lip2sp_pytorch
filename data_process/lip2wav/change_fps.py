from pathlib import Path
from subprocess import run
from tqdm import tqdm

fps = 25


def main():
    data_dir = Path("~/Lip2Wav/Dataset").expanduser()
    save_dir = Path("~/Lip2Wav/Dataset_fps25").expanduser()
    speaker_list = ["chem", "chess", "dl", "eh", "hs"]

    for speaker in speaker_list:
        print(f"speaker = {speaker}")
        spk_dir = data_dir / speaker / "intervals"
        data_path_list = list(spk_dir.glob("*/*.mp4"))
        
        for data_path in data_path_list:
            save_path = save_dir / speaker / "intervals" / data_path.parents[0].name
            save_path.mkdir(parents=True, exist_ok=True)
            save_path = save_path / f"{data_path.stem}{data_path.suffix}"
            if save_path.exists():
                continue
            
            try:    
                cmd = f"ffmpeg -i {str(data_path)} -r {fps} {str(save_path)}"
                run(cmd, shell=True, check=True)
            except:
                print("error", data_path)
                continue


if __name__ == "__main__":
    main()