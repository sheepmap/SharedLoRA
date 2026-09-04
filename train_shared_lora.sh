MODEL='IXIT2test_peft'
BASE_PATH='/root/autodl-tmp/SHFormer-master'
PRETRAINED_CHECKPOINT='/root/autodl-tmp/SHFormer-master/experiments/ixi_t2/cartesian/acc_4x/IXIT2test/best_model.pt'
DATASET_TYPE='ixi_t2' #,'mrbrain_flair','ixi_pd','ixi_t2'
MASK_TYPE='cartesian' #'cartesian' #,'gaussian'
ACC_FACTORS='4x,8x,16x' # acceleration factors
DATA_ACC_FACTOR='4x' # acceleration factor of the stored data
BATCH_SIZE=5
NUM_EPOCHS=3
LR=1e-4
DEVICE='cuda:0'
TRAIN_PATH=${BASE_PATH}'/datasets/'
VALIDATION_PATH=${BASE_PATH}'/datasets/'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# ---------------------------------------------------------------------------
# Method switch: shared_lora (ours) | convlora | melora | dora | pissa
# ---------------------------------------------------------------------------
PEFT_METHOD="shared_lora"

# --- melora-family settings (shared_lora / convlora / melora) ---
MELORA_DROPOUT=0.05
MELORA_TARGET="down_sample_layers.0,up_sample_layers.0.layers,up_sample_layers.1.layers,up_sample_layers.2.layers"

R_SHARED_LORA="8"      ; ALPHA_SHARED_LORA="16"     # single branch (no segmentation)
R_CONVLORA="8"         ; ALPHA_CONVLORA="16"        # single branch, no gate net
R_MELORA="8,8"         ; ALPHA_MELORA="16,16"       # segmented hierarchical branches

# --- DoRA (PEFT_METHOD=dora) ---
DORA_RANK=8
DORA_ALPHA=16
DORA_DROPOUT=0.0

# --- PiSSA (PEFT_METHOD=pissa) ---
PISSA_RANK=8
# PiSSA conventionally uses alpha = rank.
PISSA_ALPHA=${PISSA_RANK}

# EXP_DIR is defined after PEFT_METHOD so the method name lands in the path.
EXP_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTORS}'/'${MODEL}'_'${PEFT_METHOD}

# ---------------------------------------------------------------------------
# Map each method onto train_keyrount_lora.py arguments
# ---------------------------------------------------------------------------
PY_METHOD="melora"
ADAPTER_R=""
ADAPTER_ALPHA=""
MELORA_ARGS=""
PEFT_EXTRA_ARGS=""
PEFT_GATE_ARGS=""

case "${PEFT_METHOD}" in
    shared_lora)
        ADAPTER_R=${R_SHARED_LORA}
        ADAPTER_ALPHA=${ALPHA_SHARED_LORA}
        MELORA_ARGS="--melora_r ${ADAPTER_R} --melora_alpha ${ADAPTER_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET}"
        PEFT_GATE_ARGS="--use-lora-gate-net --no-lora-ab-gate"
        ;;
    convlora)
        ADAPTER_R=${R_CONVLORA}
        ADAPTER_ALPHA=${ALPHA_CONVLORA}
        MELORA_ARGS="--melora_r ${ADAPTER_R} --melora_alpha ${ADAPTER_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET}"
        PEFT_GATE_ARGS=""
        ;;
    melora)
        ADAPTER_R=${R_MELORA}
        ADAPTER_ALPHA=${ALPHA_MELORA}
        MELORA_ARGS="--melora_r ${ADAPTER_R} --melora_alpha ${ADAPTER_ALPHA} --melora_dropout ${MELORA_DROPOUT} --melora_target ${MELORA_TARGET}"
        PEFT_GATE_ARGS=""
        ;;
    dora)
        PY_METHOD="dora"
        PEFT_EXTRA_ARGS="--adapter-rank ${DORA_RANK} --adapter-alpha ${DORA_ALPHA} --adapter-dropout ${DORA_DROPOUT}"
        ;;
    pissa)
        PY_METHOD="pissa"
        PEFT_EXTRA_ARGS="--pissa-rank ${PISSA_RANK} --pissa-alpha ${PISSA_ALPHA}"
        ;;
    *)
        echo "Unsupported PEFT_METHOD: ${PEFT_METHOD}. Choose shared_lora, convlora, melora, dora, or pissa." >&2
        exit 1
        ;;
esac

echo python train_keyrount_lora.py --pretrained-checkpoint ${PRETRAINED_CHECKPOINT} --batch-size ${BATCH_SIZE} --num-epochs ${NUM_EPOCHS} --lr ${LR} --device ${DEVICE} --exp-dir ${EXP_DIR} --train-path ${TRAIN_PATH} --validation-path ${VALIDATION_PATH} --dataset_type ${DATASET_TYPE} --usmask_path ${USMASK_PATH} --acceleration_factor ${ACC_FACTORS} --data_acceleration_factor ${DATA_ACC_FACTOR} --mask_type ${MASK_TYPE} --peft-method ${PY_METHOD} ${PEFT_EXTRA_ARGS} ${MELORA_ARGS} ${PEFT_GATE_ARGS}

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
    --peft-method ${PY_METHOD} \
    ${PEFT_EXTRA_ARGS} \
    ${MELORA_ARGS} \
    ${PEFT_GATE_ARGS}
