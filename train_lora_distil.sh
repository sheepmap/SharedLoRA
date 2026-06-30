MODEL='IXIT2test'
BASE_PATH='/root/autodl-tmp/SHFormer-master'
PRETRAINED_CHECKPOINT='/root/autodl-tmp/SHFormer-master/experiments/ixi_t2/cartesian/acc_4x/IXIT2test/best_model.pt'
DATASET_TYPE='ixi_t2' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='cartesian' #'cartesian' #,'gaussian'
ACC_FACTORS='16x' #'4x' #,'5x','8x'
BATCH_SIZE=5
NUM_EPOCHS=3
LR=1e-4
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# MELoRA settings (following glue_finetune.sh style)
MELORA_R="8,8"
MELORA_ALPHA="16,16"
MELORA_DROPOUT=0.05
MELORA_TARGET="down_sample_layers.0,up_sample_layers"
#ffn.project_out,ffn.project_in,ffn.dwconv,

# Distillation settings
DISTIL_ALPHA=0.1             # 蒸馏损失权重，设为 0 禁用
REF_ACC_FACTOR='4x'          # 参考分支的低倍欠采样倍数

# Data augmentation settings
AUG_FLIP=true                # 高斯噪声增广开关（sigma=25/50），true 为启用
DISTIL_MULTI_SCALE=true      # 多尺度蒸馏损失开关，true 为启用

$AUG_FLIP && AUG_FLIP_ARG="--aug-flip" || AUG_FLIP_ARG=""
$DISTIL_MULTI_SCALE && DISTIL_MS_ARG="--distil-multi-scale" || DISTIL_MS_ARG=""

echo python train_lora_distil.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET} --distil-alpha ${DISTIL_ALPHA} --ref-acceleration-factor ${REF_ACC_FACTOR} ${AUG_FLIP_ARG} ${DISTIL_MS_ARG}

python train_lora_distil.py \
    --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} \
    --batch-size ${BATCH_SIZE} \
    --num-epochs ${NUM_EPOCHS} \
    --lr ${LR} \
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
    --distil-alpha ${DISTIL_ALPHA} \
    --ref-acceleration-factor ${REF_ACC_FACTOR} \
    ${AUG_FLIP_ARG} \
    ${DISTIL_MS_ARG}
