#!/bin/bash

#PJM -L "rscunit=ito-b"        
#PJM -L "rscgrp=ito-g-1-dbg"
#PJM -L "vnode=1"
#PJM -L "vnode-core=9"
#PJM -L "elapse=30:00"
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


# これより下に，Pythonなどを実行するコマンドを書きます。
# 実際はもう少しごちゃごちゃした内容を記述していることが多いです。
model=/home/usr1/q70261a/lip2sp_pytorch_all/lip2sp_920_re/check_point/default/lip/2022:10:25_15-23-11_mask_test_loss_test_layer2_mentanance/mspec80_250.ckpt
tag=mask_test_loss_test_layer2_mentanance

python $file_main train.debug=False tag=$tag model_path=$model
# python $file_main model.n_layers=1,2 -m
