#!/bin/bash
uname -a
#date
#env
date
CS_PATH='/home/liuwu1/notespace/dataset'
LR=1e-2
WD=5e-4
BS=12
GPU_IDS=0,1,2,3
RESTORE_FROM='../../models/hrnet_w48_lip_cls20_473x473.pth'
INPUT_SIZE='480,480'
SNAPSHOT_DIR='./models/LIP_HRNet_origin/'
DATASET='train'
NUM_CLASSES=20
EPOCHS=30
START=0
LOSS='softmax'
LIST_PATH='/home/liuwu1/notespace/dataset/LIP/LIP_CIHP_train_path_new.txt'
SAVE_STEP=2

if [[ ! -e ${SNAPSHOT_DIR} ]]; then
    mkdir -p  ${SNAPSHOT_DIR}
fi

python -u train_psp.py --data-dir ${CS_PATH} \
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
       --start-epoch ${START} \
       --loss ${LOSS} \
       --list_path ${LIST_PATH}\
       --save_step ${SAVE_STEP}
