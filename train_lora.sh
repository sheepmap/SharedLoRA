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
MELORA_TARGET="up_sample_layers"
#ffn.project_out,ffn.project_in,ffn.dwconv,

# Feature loss settings
FEAT_LOSS_ALPHA=0.1          # 特征损失权重，设为 0 禁用
REF_ACC_FACTOR='4x'          # 参考分支的低倍欠采样倍数
FEAT_EXTRACT_LAYERS='0'    # 特征提取层索引，如 "0,1,2" 或 "1,2"，使用 "conv" 时 --feat-extract-layers 应设为 0
FEAT_EXTRACT_CASCADE='2'  # 提取哪些级联的特征，-1 表示最后一个，如 "2,3,4"
FEAT_EXTRACT_LAYER_TYPE='conv'  # 从哪些层提取特征: "up"(up_sample_layers), "down"(down_sample_layers), "conv"(瓶颈层), "both"(up+down)

echo python train_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --mask_type ${MASK_TYPE} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET} --feat-loss-alpha ${FEAT_LOSS_ALPHA} --ref-acceleration-factor ${REF_ACC_FACTOR} --feat-extract-layers ${FEAT_EXTRACT_LAYERS} --feat-extract-cascade ${FEAT_EXTRACT_CASCADE} --feat-extract-layer-type ${FEAT_EXTRACT_LAYER_TYPE}

python train_lora.py \
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
    --feat-loss-alpha ${FEAT_LOSS_ALPHA} \
    --ref-acceleration-factor ${REF_ACC_FACTOR} \
    --feat-extract-layers ${FEAT_EXTRACT_LAYERS} \
    --feat-extract-cascade ${FEAT_EXTRACT_CASCADE} \
    --feat-extract-layer-type ${FEAT_EXTRACT_LAYER_TYPE}
