MODEL='IXIT2test_keyrount_l1'
BASE_PATH='/root/autodl-tmp/SHFormer-master'
PRETRAINED_CHECKPOINT='/root/autodl-tmp/SHFormer-master/experiments/ixi_t2/cartesian/acc_4x/IXIT2test/best_model.pt'
DATASET_TYPE='ixi_t2' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='cartesian' #'cartesian' #,'gaussian'
ACC_FACTORS='4x,8x,16x'
DATA_ACC_FACTOR='16x'
BATCH_SIZE=5
NUM_EPOCHS=3
LR=1e-4
DEVICE='cuda:0'
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# MELoRA settings (following glue_finetune.sh style)
MELORA_R="8"
MELORA_ALPHA="16"
MELORA_DROPOUT=0.05
# MELORA_TARGET="down_sample_layers.0,up_sample_layers"
MELORA_TARGET="down_sample_layers.0,up_sample_layers.0.layers,up_sample_layers.1.layers,up_sample_layers.2.layers"
USE_LORA_GATE_NET=1
USE_LORA_AB_GATE=1
#ffn.project_out,ffn.project_in,ffn.dwconv,

LORA_GATE_NET_ARG=""
if [ "${USE_LORA_GATE_NET}" = "1" ]; then
    LORA_GATE_NET_ARG="--use-lora-gate-net"
fi

LORA_AB_GATE_ARG="--use-lora-ab-gate"
if [ "${USE_LORA_AB_GATE}" = "0" ]; then
    LORA_AB_GATE_ARG="--no-lora-ab-gate"
fi

echo python train_keyrount_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --data_acceleration_factor ${DATA_ACC_FACTOR} --mask_type ${MASK_TYPE} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET} ${LORA_GATE_NET_ARG} ${LORA_AB_GATE_ARG}

python train_keyrount_lora.py \
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
    --data_acceleration_factor ${DATA_ACC_FACTOR} \
    --mask_type ${MASK_TYPE} \
    --melora_r ${MELORA_R} \
    --melora_alpha ${MELORA_ALPHA} \
    --melora_dropout ${MELORA_DROPOUT} \
    --melora_target ${MELORA_TARGET} \
    ${LORA_GATE_NET_ARG} \
    ${LORA_AB_GATE_ARG} \
    --resume \
    --checkpoint ${EXP_DIR}/melora/r8_down_sample_layers.0_up_sample_layers.0.layers_up_sample_layers.1.layers_up_sample_layers.2.layers_gate_ab/checkpoint.pt
