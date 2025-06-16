import os

dir_path = "/home/user/vcc18/data/scp/"
print("=== ディレクトリ内のファイル一覧 ===")
for f in os.listdir(dir_path):
    print(f" - {f}")