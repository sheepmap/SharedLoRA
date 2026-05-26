MODEL='<model name or experiment name>'
BASE_PATH='<base path>'
PRETRAINED_CHECKPOINT='<path to pretrained model.pt>'
DATASET_TYPE='<folder name with the type of MRI contrast>' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='<foldername with the type of mask>' #'cartesian' #,'gaussian'
ACC_FACTORS='<folder name with the acceleration factor number followed character - x>' #'4x' #,'5x','8x'
BATCH_SIZE=4
NUM_EPOCHS=5
LR=0.0001
LR_STEP_SIZE=2
LR_GAMMA=0.1
LR_ETA_MIN=1e-7
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# MELoRA settings (following glue_finetune.sh style)
MELORA_R="8,8"
MELORA_ALPHA="16,16"
MELORA_DROPOUT=0.05
MELORA_TARGET="dynHPF.conv,attn.project_out,up_sample_layers"

echo python train_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --lr-step-size ${LR_STEP_SIZE} --lr-gamma ${LR_GAMMA} --lr-eta-min ${LR_ETA_MIN} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET}

python train_lora.py \
    --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} \
    --batch-size ${BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --lr ${LR} \
    --lr-step-size ${LR_STEP_SIZE} \
    --lr-gamma ${LR_GAMMA} \
    --lr-eta-min ${LR_ETA_MIN} \
    --device ${DEVICE} \
    --exp-dir ${EXP_DIR} \
    --train-path ${TRAIN_PATH} \
    --validation-path ${VALIDATION_PATH} \
    --dataset_type ${DATASET_TYPE} \
    --usmask_path ${USMASK_PATH} \
    --acceleration_factor ${ACC_FACTORS} \
    --mask_type ${MASK_TYPE} \
    --melora_r ${MELORA_R} \
    --melora_alpha ${MELORA_ALPHA} \
    --melora_dropout ${MELORA_DROPOUT} \
    --melora_target ${MELORA_TARGET}
