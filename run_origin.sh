#!/bin/bash
uname -a
date

CS_PATH='/RAID_20T/RAID_2/zm/LIP/'
LR=1e-2
WD=5e-4
BS=4
GPU_IDS=0,1,2,3
RESTORE_FROM='./checkpoints/resnet101-imagenet.pth'
INPUT_SIZE='384,384'
SNAPSHOT_DIR='./checkpoints/LIP_HRNet_miou/'
DATASET='train'
NUM_CLASSES=20
EPOCHS=150
START=0
LIST_PATH='/RAID_20T/RAID_2/zm/LIP/code_need/train_path.txt'
SAVE_STEP=2
MODEL='seg_hrnet'
python -u train.py --data-dir ${CS_PATH}\
       --model ${MODEL}\
       --random-mirror\
       --random-scale\
       --iouloss\
       --restore-from ''\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS} \
       --start-epoch ${START}\
       --list_path ${LIST_PATH}\
       --save_step ${SAVE_STEP}