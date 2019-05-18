#!/bin/bash

CS_PATH='/export/home/zm/dataset/LIP/'
BS=32
GPU_IDS='3'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='./models/LIP_epoch_9.pth'
DATASET='val'
NUM_CLASSES=20
OUTPUTS='./outputs/CIHP_LIP_HRNet_ohem/'

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}
