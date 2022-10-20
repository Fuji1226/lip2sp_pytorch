#!/bin/bash

#PJM -L "rscunit=ito-b"        
#PJM -L "rscgrp=ito-g-1"
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "elapse=168:00:00"
#PJM -m b
#PJM -m e
#PJM -j
#PJM -X


# 自分が使うCUDAのバージョンを指定
# TensorFlowやPyTorchのバージョンは，
# CUDAのバージョンに合わせてインストールしましょう
module load cuda/11.0


# どこからこのシェルスクリプトを実行したとしても，
# scripts/の直下に移動してからPythonファイルを実行するようにします。
dir_project="$(dirname $(cd $(dirname $0); pwd))"
# cd "${dir_project}/scripts"
cd "${dir_project}"
file_main="generate.py"
model_path=/home/usr1/q70261a/lip2sp_pytorch_all/lip2sp_920_re/check_point/default/lip/2022:10:18_21-29-26_GLU_test1018/mspec80_320.ckpt


# これより下に，Pythonなどを実行するコマンドを書きます。
# 実際はもう少しごちゃごちゃした内容を記述していることが多いです。
python $file_main train.debug=False model=mspec80_glu model_path=$model_path
# python $file_main model.n_layers=1,2 -m
