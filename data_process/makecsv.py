import pandas as pd
import os

def remove_wav_extension(csv_path, output_csv, output_folder_path):
    # CSVファイルを読み込む
    df = pd.read_csv(csv_path)

    # 'filename' 列の .wav 拡張子を削除
    df['filename'] = df['filename'].str.replace(r'\.wav$', '', regex=True)

        # CSVファイルのフルパスを作成
    output_csv_path = os.path.join(output_folder_path, output_csv)

    # CSVファイルとして保存
    df.to_csv(output_csv_path, index=False)

    print(f"拡張子を削除したCSVを保存しました: {output_csv}")

# 使用例
input_csv = '/home/user/2HEAVD/F1/audio/audio_val.csv'
output_csv = "val.csv"
output_folder_path= '/home/user/2HEAVD/F1'
remove_wav_extension(input_csv, output_csv, output_folder_path)
