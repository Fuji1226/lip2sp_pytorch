#!/bin/bash

# このスクリプトは，ローカル（自分のPC）とITOのデータの送受信を行うためのものです。
# PyTorchやpytorch-lightningの使用を想定していますが，
# そうでない場合にも使用可能です。
# 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# コマンドを実行する前に，「準備」にあるSSHの設定と，
# 「設定」の変数の値の設定を行なってください。
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# 
# 基本的な使い方は以下の通りになります（shellsディレクトリ内にいると仮定しています）。
# - ./rsyncito.sh dev2svr
#   - ローカルからITOへ，データを送ります
# - ./rsyncito.sh svr2dev
#   - ITOからローカルへ，データを送ります（.ckptファイルを除く）
# 
# その他に，次のような使い方もできます（shellsディレクトリ内にいると仮定しています）。
# - ./rsyncito.sh svr2dev --include='*.ckpt'
#   - ITOからローカルへ，データを送ります（.ckptファイルを含む）
# - ./rsyncito.sh svr2any your/target/directory/path/
#   - ITOからyour/target/directory/path/へ，データを送ります
#   
# このスクリプトは，コマンドラインツールrsyncに依存しています。
# dev2svr，svr2dev，svr2anyのそれぞれに，キャッシュファイルなどを除くために
# rsyncの`--exclude`オプションを指定しています。
# 必要に応じて，rsyncの別のオプションを追加で指定することができます。
# dev2svr，svr2dev，svr2anyのそれぞれにある`exclude`オプションをキャンセルすることもできます。
# (例: `./rsyncito.sh svr2dev --include='*.ckpt'`)


################################################################################
### 準備; 最初にSSHの設定を完了させてください
################################################################################
# `~/.ssh/config`に以下の設定を記述してください.
# Host ito
# ServerAliveInterval 60
# ServerAliveCountMax 3
# AddKeysToAgent yes
# UseKeychain yes
# HostName ito.cc.kyushu-u.ac.jp
# User {支給されたITOのユーザ名（例: a01234z）}
# IdentityFile {ITOとやり取りするためのSSHの秘密鍵のパス（例: ~/.ssh/id_rsa）}


################################################################################
### 設定; 最初に以下の変数に，自分の環境に適した値を入力してください
################################################################################
# ITOの自分のホームディレクトリは以下のようになっています
# /home/USR_DIRNAME/USER_NAME

USER_NAME='r70264c'  # 支給されたITOのユーザ名 (例: a01234z)
USR_DIRNAME='usr4'  # usr数字; 例: usr4


################################################################################
### 処理; 以下は，自分の好みに合わせて必要に応じて変更してください
################################################################################
dir_local_raw=$(dirname $(cd $(dirname $0); pwd))
dir_name=`echo "${dir_local_raw}" | sed -e 's/.*\/\([^\/]*\)$/\1/'`
dir_local="${dir_local_raw}/"
dir_remote="ito:/home/${USR_DIRNAME}/${USER_NAME}/${dir_name}/"

# Development directory -> Server directory
# Usage: ` ./rsyncito.sh dev2svr `
if [ "$1" = 'dev2svr' ]; then
    echo "From: ${dir_local}"
    echo "To  : ${dir_remote}"
    rsync "${dir_local}" "${dir_remote}" -av ${@:2} \
        --exclude='rsyncito.sh' --exclude='results/' \
        --exclude='.env' \
        --exclude='.DS_Store' --exclude='Thumbs.db' \
        --exclude='.git' --exclude='.gitignore' \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' --exclude='.pytest_cache/' \
        --exclude='check_point/' --exclude='multirun/' --exclude='outputs/' --exclude='result/' --exclude='wandb'\
        --exclude='*.cpt' --exclude='*.ckpt' --exclude='conf/' --exclude='*.py'

# Server directory -> Development directory
# Usage: ` ./rsyncito.sh svr2dev `
# NOTE: By default, .ckpt files are NOT downloaded due to its large size.
# If you want to download the .ckpt files, use the following command:
# ```
# ./rsyncito.sh svr2dev --include='*.ckpt'
# ```
# The priority of `--include` and `--exclude` options of `rsync` is
# in the same as the order of declaration.
# The above command inserts `--include='*.ckpt'` at `${@:2}` and cancels
# the following `--exclude='*.ckpt'`.
elif [ "$1" = 'svr2dev' ]; then
    echo "From: ${dir_remote}"
    echo "To  : ${dir_local}"
    rsync "${dir_remote}" "${dir_local}" -av ${@:2} \
        --exclude='venv/' --exclude='*.ckpt' --exclude='datasets/' --exclude='notebooks/' --exclude='scripts/' --exclude='shells/.d*' --exclude='shells/*.sh.o*' \
        --exclude='.env' \
        --exclude='.DS_Store' --exclude='Thumbs.db' \
        --exclude='.git' --exclude='.gitignore' \
        --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' --exclude='.pytest_cache/'\
        --exclude='check_point/' --exclude='multirun/' --exclude='outputs/' --exclude='*.pth' --exclude='wandb'\
        --exclude='*.cpt' --exclude='*.ckpt' --exclude='result/' --exclude='data_check/' --exclude='result_final/'\
        --exclude='check/'

# Server directory -> Any directory
# Usage: ` ./rsyncito.sh svr2any your/target/directory/path/ `
elif [ "$1" = 'svr2any' ]; then
    if [ -n "$2" ]; then
        echo "From: ${dir_remote}"
        echo "To  : $2"
        rsync ${dir_remote} $2 -av ${@:3} \
            --exclude='venv/' --exclude='scripts/.d*' --exclude='scripts/*.sh.o*' \
            --exclude='.env' \
            --exclude='.DS_Store' --exclude='Thumbs.db' \
            --exclude='.git' --exclude='.gitignore' \
            --exclude='__pycache__' --exclude='*.pyc' --exclude='.ipynb_checkpoints' --exclude='.pytest_cache/'
    else
        echo 'If $1 is "svr2any", $2 must be given.'
    fi
else
    echo '$1 must be "dev2svr", "svr2dev" or "svr2any".'
fi
