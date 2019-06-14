#!/bin/bash

CS_PATH='/home/liuwu1/notespace/dataset/LIP/'
BS=16
GPU_IDS='0,1,2,3'
INPUT_SIZE='480,480'
SNAPSHOT_FROM='train_test/HRNet_origin/models/LIP_HRNet_origin_iou/LIP_epoch_9.pth'
DATASET='hrnet_origin'
NUM_CLASSES=20
OUTPUTS='./outputs/denseaspp/'
LIST_PATH='/home/liuwu1/notespace/dataset/LIP/val_id.txt'
MODEL='hrnet_origin'
M_RATE=0.8

python evaluate.py --data-dir ${CS_PATH} \
       --gpu ${GPU_IDS} \
       --batch-size ${BS} \
       --input-size ${INPUT_SIZE}\
       --restore-from ${SNAPSHOT_FROM}\
       --dataset ${DATASET}\
       --num-classes ${NUM_CLASSES}\
       --save-dir ${OUTPUTS}\
       --list_path ${LIST_PATH}\
       --model ${MODEL}\
       --m_rate ${M_RATE}
