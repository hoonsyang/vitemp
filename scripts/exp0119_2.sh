#!/bin/bash

# Fix
TRAIN_DIR='/home/vilab/yhm/Beyond-ImageNet-Attack'
LOG_DIR='/home/vilab/yhm/Beyond-ImageNet-Attack/logs'
# Victim model
MODEL_TYPE='vgg16'
# Method
RN=False
DA=False
FA=True
# Training options
LOAD_PRETRAINED_G=True
TRAINING_EPOCHS=10
BATCH_SIZE=16
LEARNING_RATE=0.00005
# Exp name
DATE='exp0119'
EXP_NAME='lr'+$(echo ${LEARNING_RATE})
GPU_ID='2'

# Train
python3 ${TRAIN_DIR}/train.py \
    --model_type ${MODEL_TYPE} \
    --RN ${RN} \
    --DA ${DA} \
    --FA ${FA} \
    --load_pretrained_G ${LOAD_PRETRAINED_G} \
    --epochs ${TRAINING_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LEARNING_RATE} \
    --exp ${EXP_NAME} \
    --output_dir ${LOG_DIR}/${DATE}/${EXP_NAME} \
    --gpu_id ${GPU_ID}
    