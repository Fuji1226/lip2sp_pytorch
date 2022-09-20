#!/bin/bash

#PJM -L "rscunit=ito-b"        
#PJM -L "rscgrp=ito-g-1"
#PJM -L "vnode=1"
#PJM -L "elapse=120:34:56"
#PJM --mail-list minami.taisuke.326@s.kyushu-u.ac.jp
#PJM -m b
#PJM -m e
#PJM -j
#PJM -X


# 自分が使うCUDAのバージョンを指定
# TensorFlowやPyTorchのバージョンは，
# CUDAのバージョンに合わせてインストールしましょう
module load cuda/10.1


# どこからこのシェルスクリプトを実行したとしても，
# scripts/の直下に移動してからPythonファイルを実行するようにします。
dir_project="$(dirname $(cd $(dirname $0); pwd))"
# cd "${dir_project}/scripts"
cd "${dir_project}/data_process"
file_main="make_txt.py"


# これより下に，Pythonなどを実行するコマンドを書きます。
# 実際はもう少しごちゃごちゃした内容を記述していることが多いです。
python $file_main 
# python $file_main model.n_layers=1,2 -m

