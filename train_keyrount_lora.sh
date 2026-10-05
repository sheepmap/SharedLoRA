#!/usr/bin/env bash
# Canonical PEFT training entrypoint for integer acceleration factors.
# Edit the USER CONFIGURATION section below, then run: bash train_keyrount_lora.sh

# ========================= USER CONFIGURATION =========================
MODEL='fastmritest'
BASE_PATH='/root/autodl-tmp'
PRETRAINED_CHECKPOINT="${BASE_PATH}/experiments/fastmri_knee/gaussian/acc_8x/fastmritest/best_model.pt"
DATASET_TYPE='fastmri_knee'
MASK_TYPE='gaussian'
# Factors passed to Python during training.
ACC_FACTORS='2x,8x,10x'
# Exact directory token for this training run; keep it separate from validation factors.
TRAIN_ACC_FACTORS='2x,8x,10x'
DATA_ACC_FACTOR='10x'
MASK_GROUP='seed45'
BATCH_SIZE=4
NUM_EPOCHS=10
LR=5e-5
GRAD_CLIP=1.0
DEVICE='cuda:0'

# Choose the LoRA-family scheme here. This is the main setting to change:
#   shared_lora one branch + gate: R=8;   alpha=16;   gate=1; A/B gate=0
#   melora      segmented MELoRA: R=8,8; alpha=16,16; gate=0; A/B gate=0
#   convlora    one branch:        R=8;   alpha=16;   gate=0; A/B gate=0
#   dora        DoRA magnitude/direction adapter
#   pissa       PiSSA SVD-initialized adapter
PEFT_METHOD='shared_lora'

# Advanced settings. Usually leave these unchanged; they apply only to the
# corresponding method and do not override the method-specific R/alpha above.
MELORA_DROPOUT=0.05
MELORA_TARGET='down_sample_layers.0,up_sample_layers.0.layers,up_sample_layers.1.layers,up_sample_layers.2.layers'
DORA_RANK=8
DORA_ALPHA=16
DORA_DROPOUT=0.0
PISSA_RANK=8
PISSA_ALPHA=${PISSA_RANK}
# ======================= END USER CONFIGURATION =======================

# Derived paths are shared by all methods. Keep the run directory stable so
# validation/evaluation scripts can find the default MELoRA experiment.
EXP_DIR="${BASE_PATH}/experiments/${DATASET_TYPE}/${MASK_TYPE}/acc_${TRAIN_ACC_FACTORS}/${MODEL}_${MASK_GROUP}"
TRAIN_PATH="${BASE_PATH}/datasets/"
VALIDATION_PATH="${BASE_PATH}/datasets/"
USMASK_PATH="${BASE_PATH}/usmasks/"

# Translate the selected method into train_keyrount_lora.py arguments.
# The three MELoRA-family configurations intentionally differ as documented above.
PEFT_METHOD_ARGS=''
PEFT_EXTRA_ARGS=''
PEFT_GATE_ARGS=''
case "${PEFT_METHOD}" in
  melora)
    MELORA_R='8,8'
    MELORA_ALPHA='16,16'
    USE_LORA_GATE_NET=0
    USE_LORA_AB_GATE=0
    PEFT_METHOD_ARGS='--peft-method melora'
    PEFT_GATE_ARGS='--no-lora-ab-gate'
    ;;
  shared_lora)
    MELORA_R='8'
    MELORA_ALPHA='16'
    USE_LORA_GATE_NET=1
    USE_LORA_AB_GATE=0
    PEFT_METHOD_ARGS='--peft-method melora'
    PEFT_GATE_ARGS='--use-lora-gate-net --no-lora-ab-gate'
    ;;
  convlora)
    MELORA_R='8'
    MELORA_ALPHA='16'
    USE_LORA_GATE_NET=0
    USE_LORA_AB_GATE=0
    PEFT_METHOD_ARGS='--peft-method melora'
    PEFT_GATE_ARGS='--no-lora-ab-gate'
    ;;
  dora)
    PEFT_METHOD_ARGS='--peft-method dora'
    PEFT_EXTRA_ARGS="--adapter-rank ${DORA_RANK} --adapter-alpha ${DORA_ALPHA} --adapter-dropout ${DORA_DROPOUT}"
    ;;
  pissa)
    PEFT_METHOD_ARGS='--peft-method pissa'
    PEFT_EXTRA_ARGS="--pissa-rank ${PISSA_RANK} --pissa-alpha ${PISSA_ALPHA}"
    ;;
  *)
    echo "Unsupported PEFT_METHOD: ${PEFT_METHOD}. Choose melora, shared_lora, convlora, dora, or pissa." >&2
    exit 1
    ;;
esac

# All modes call the same Python implementation; only the adapter arguments
# above change when PEFT_METHOD changes.
python train_keyrount_lora.py \
  --pretrained-checkpoint "${PRETRAINED_CHECKPOINT}" \
  --batch-size "${BATCH_SIZE}" \
  --num-epochs "${NUM_EPOCHS}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --exp-dir "${EXP_DIR}" \
  --train-path "${TRAIN_PATH}" \
  --validation-path "${VALIDATION_PATH}" \
  --dataset_type "${DATASET_TYPE}" \
  --usmask_path "${USMASK_PATH}" \
  --acceleration_factor "${ACC_FACTORS}" \
  --data_acceleration_factor "${DATA_ACC_FACTOR}" \
  --mask_type "${MASK_TYPE}" \
  --mask_group "${MASK_GROUP}" \
  ${PEFT_METHOD_ARGS} \
  ${PEFT_EXTRA_ARGS} \
  --melora_r "${MELORA_R}" \
  --melora_alpha "${MELORA_ALPHA}" \
  --melora_dropout "${MELORA_DROPOUT}" \
  --melora_target "${MELORA_TARGET}" \
  --grad-clip "${GRAD_CLIP}" \
  ${PEFT_GATE_ARGS}



