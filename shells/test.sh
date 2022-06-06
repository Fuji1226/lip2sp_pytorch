#!/bin/bash

#PJM -L "rscunit=ito-b"
#PJM -L "rscgrp=ito-g-1"
#PJM -L "vnode=1"
#PJM -L "elapse=120:00:00"
#PJM -m b
#PJM -m e
#PJM -j
#PJM -X

module load cuda/10.1

export OPENBLAS_NUM_THREADS=1

dir_project="$(dirname $(cd $(dirname $0); pwd))"
cd "${dir_project}"

file_main="train.py"

python $file_main