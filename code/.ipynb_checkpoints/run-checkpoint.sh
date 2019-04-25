#!/bin/bash
uname -a
#date
#env
date
CS_PATH='/export/home/zm/dataset/LIP/'
LR=1e-3
WD=5e-4
BS=24
GPU_IDS=0,1,2,3
RESTORE_FROM='./models/snapshots/resnet101-imagenet.pth'
INPUT_SIZE='384,384'  
SNAPSHOT_DIR='./models/snapshots1/'
DATASET='train'
NUM_CLASSES=20
START_EPOCH=0
EPOCHS=150

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python -u train_origin.py --data-dir ${CS_PATH} \
       --random-mirror\
       --random-scale\
       --restore-from ${RESTORE_FROM}\
       --gpu ${GPU_IDS}\
       --learning-rate ${LR}\
       --weight-decay ${WD}\
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --snapshot-dir ${SNAPSHOT_DIR}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES} \
       --epochs ${EPOCHS} \
       --start-epoch ${START_EPOCH}

