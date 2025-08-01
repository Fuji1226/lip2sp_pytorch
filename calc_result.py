#同一ディレクトリ内に複数あるフォルダ(filename→emotion_type_number)の一階層下にあるnpzファイル(accuracy_metrics.npz)をそれぞれ読み込み、filenameの一部(emotion)毎に中身の平均値を計算、保存する

import numpy as np
from pathlib import Path
from collections import defaultdict
import os

# npzファイルが入っているフォルダ群のある親ディレクトリ
base_dir = Path('/home/user/lip2sp_pytorch/result/nar/generate/avhubert_preprocess_fps25_gray/master/2025:08:01_14-13-42/4/test_data/audio/hifigan/F1').expanduser().resolve()

# 感情タイプごとに値を蓄積する辞書
metrics_by_emotion = defaultdict(list)

# 各サブディレクトリを走査（例: happiness_001, sadness_002, etc.）
for subdir in base_dir.iterdir():
    if subdir.is_dir():
        # emotionタイプを抽出（最初のアンダースコアまで）
        emotion = subdir.name.split("_")[0]

        # 1階層下の accuracy_metrics.npz を探す
        npz_path = subdir / "accuracy_metrics.npz"
        if npz_path.exists():
            data = np.load(npz_path)
            # 辞書に変換して保存（複数のmetricがあることを想定）
            metrics_by_emotion[emotion].append({k: data[k] for k in data.files})
        else:
            print(f"Warning: {npz_path} not found.")

# 平均値を計算し保存
output_dir = base_dir / "averaged_metrics_by_emotion"
output_dir.mkdir(exist_ok=True)

for emotion, metric_list in metrics_by_emotion.items():
    # 各キーごとに平均計算
    mean_metrics = {}
    keys = metric_list[0].keys()
    for key in keys:
        stacked = np.stack([m[key] for m in metric_list])
        mean_metrics[key] = np.mean(stacked, axis=0)

    # 保存
    output_path = output_dir / f"{emotion}_mean_metrics.npz"
    np.savez(output_path, **mean_metrics)
    print(f"Saved: {output_path}")

    #作成したnpzファイルを読み込みtxtファイルに保存

    with open(output_dir / f"{emotion}_mean_metrics.txt", "w") as f:
        for key, value in mean_metrics.items():
            if isinstance(value, np.ndarray):
                value_str = ', '.join(map(str, value))
            else:
                value_str = str(value)
            f.write(f"{key}: {value_str}\n")

    #作成したnpzファイルの削除
    os.remove(output_path)


