MODEL='IXIT2test'
BASE_PATH='/root/autodl-tmp/SHFormer-master'
PRETRAINED_CHECKPOINT='/root/autodl-tmp/SHFormer-master/experiments/ixi_t2/cartesian/acc_4x/IXIT2test/best_model.pt'
DATASET_TYPE='ixi_t2' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='cartesian' #'cartesian' #,'gaussian'
ACC_FACTORS='8x' #'4x' #,'5x','8x'
BATCH_SIZE=6
NUM_EPOCHS=5
LR=5e-5
LR_STEP_SIZE=3
LR_ETA_MIN=1e-6
LR_GAMMA=0.5
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# MELoRA settings (following glue_finetune.sh style)
MELORA_R="8,8"
MELORA_ALPHA="16,16"
MELORA_DROPOUT=0.05
MELORA_TARGET="up_sample_layers"
#ffn.project_out,ffn.project_in,ffn.dwconv,

# Feature loss settings
FEAT_LOSS_ALPHA=0.1          # 特征损失权重，设为 0 禁用
REF_ACC_FACTOR='4x'          # 参考分支的低倍欠采样倍数

echo python train_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --lr-step-size ${LR_STEP_SIZE} --lr-gamma ${LR_GAMMA} --lr-eta-min ${LR_ETA_MIN} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET} --feat-loss-alpha ${FEAT_LOSS_ALPHA} --ref-acceleration-factor ${REF_ACC_FACTOR}

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
    --melora_target ${MELORA_TARGET} \
    --feat-loss-alpha ${FEAT_LOSS_ALPHA} \
    --ref-acceleration-factor ${REF_ACC_FACTOR}
