#!/bin/bash

T=`date +%m%d%H%M%S`

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log

root=/qnap/home_archive/wlin38/coey

python main.py --data-path  $root/gsc147/ --batch-size 12 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
# python main.py --resume ../output/ckpt_epoch_best.pth --data-path  $root/gsc147/ --batch-size 1 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
