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

data_path=/home/wlin38/fixedpoint_prompt_counting/data_fsc147/train_fsc147

# training
python main.py --data-path  $data_path --batch-size 12 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log

# evaluation
# python main.py --eval --resume ../output/ckpt_epoch_best.pth --data-path  $data_path --batch-size 1 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
