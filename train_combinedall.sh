#!/usr/bin/env bash
# Allow sh invocation to re-exec this Bash script.
if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi
MODEL='fastmritest'
BASE_PATH='/root/autodl-tmp'
DATASET_TYPE='fastmri_knee' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='gaussian' #'cartesian' #,'gaussian'
ACC_FACTORS='5x' #'4x','5x','8x','16x'
BATCH_SIZE=4
NUM_EPOCHS=15
LR=1e-4
LR_ETA_MIN=1e-6
WEIGHT_DECAY=1e-4
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'
CHECKPOINT=${EXP_DIR}'/best_model.pt'
# echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --resume --checkpoint ${CHECKPOINT}
# python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --resume --checkpoint ${CHECKPOINT}
echo python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --lr-eta-min ${LR_ETA_MIN} --weight-decay ${WEIGHT_DECAY} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --mask-resample
python train.py --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --lr-eta-min ${LR_ETA_MIN} --weight-decay ${WEIGHT_DECAY} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --mask-resample
