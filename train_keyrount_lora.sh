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

# -----------------------------------------------------------------------------
# PEFT method guide
#   melora  : existing hierarchical LoRA baseline; LoRA gate net is supported.
#   dora    : DoRA; trains low-rank direction updates plus output magnitudes.
#   lora-xs : LoRA-XS; freezes SVD bases and trains only an r x r core matrix.
#
# For a fair main comparison, keep MELORA_TARGET, data, seed and training
# settings unchanged across methods. DoRA and LoRA-XS must not use the gate net.
# -----------------------------------------------------------------------------
PEFT_METHOD="melora"  # choose: melora, dora, lora-xs

# MELoRA settings
MELORA_R="8"
MELORA_ALPHA="16"
MELORA_DROPOUT=0.05
# MELORA_TARGET="down_sample_layers.0,up_sample_layers"
MELORA_TARGET="down_sample_layers.0,up_sample_layers.0.layers,up_sample_layers.1.layers,up_sample_layers.2.layers"
USE_LORA_GATE_NET=1
USE_LORA_AB_GATE=0

# DoRA rank scan: 2, 4, 8. Used only when PEFT_METHOD="dora".
DORA_RANK=8
DORA_ALPHA=16
DORA_DROPOUT=0.0

# LoRA-XS rank scan: 4, 8, 16. Used only when PEFT_METHOD="lora-xs".
LORA_XS_RANK=8
LORA_XS_ALPHA=1.0

LORA_GATE_NET_ARG=""
if [ "${USE_LORA_GATE_NET}" = "1" ]; then
    LORA_GATE_NET_ARG="--use-lora-gate-net"
fi

LORA_AB_GATE_ARG="--use-lora-ab-gate"
if [ "${USE_LORA_AB_GATE}" = "0" ]; then
    LORA_AB_GATE_ARG="--no-lora-ab-gate"
fi

PEFT_METHOD_ARGS="--peft-method ${PEFT_METHOD}"
PEFT_EXTRA_ARGS=""
PEFT_GATE_ARGS=""
case "${PEFT_METHOD}" in
    melora)
        PEFT_GATE_ARGS="${LORA_GATE_NET_ARG} ${LORA_AB_GATE_ARG}"
        ;;
    dora)
        PEFT_EXTRA_ARGS="--adapter-rank ${DORA_RANK} --adapter-alpha ${DORA_ALPHA} --adapter-dropout ${DORA_DROPOUT}"
        ;;
    lora-xs)
        PEFT_EXTRA_ARGS="--lora-xs-rank ${LORA_XS_RANK} --lora-xs-alpha ${LORA_XS_ALPHA}"
        ;;
    *)
        echo "Unsupported PEFT_METHOD: ${PEFT_METHOD}. Choose melora, dora, or lora-xs." >&2
        exit 1
        ;;
esac

if [ "${PEFT_METHOD}" != "melora" ] && [ "${USE_LORA_GATE_NET}" = "1" ]; then
    echo "Note: --use-lora-gate-net is omitted for ${PEFT_METHOD}; it is MELoRA-only."
fi

echo python train_keyrount_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --data_acceleration_factor ${DATA_ACC_FACTOR} --mask_type ${MASK_TYPE} ${PEFT_METHOD_ARGS} ${PEFT_EXTRA_ARGS} --melora_r ${MELORA_R} --melora_alpha ${MELORA_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET} ${PEFT_GATE_ARGS}

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
    ${PEFT_METHOD_ARGS} \
    ${PEFT_EXTRA_ARGS} \
    --melora_r ${MELORA_R} \
    --melora_alpha ${MELORA_ALPHA} \
    --melora_dropout ${MELORA_DROPOUT} \
    --melora_target ${MELORA_TARGET} \
    ${PEFT_GATE_ARGS} \
    # --resume \
    # --checkpoint ${EXP_DIR}/melora/r8_down_sample_layers.0_up_sample_layers.0.layers_up_sample_layers.1.layers_up_sample_layers.2.layers_gate_ab/checkpoint.pt
