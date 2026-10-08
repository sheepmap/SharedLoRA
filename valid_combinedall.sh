#!/usr/bin/env bash
# Canonical PEFT validation/inference entrypoint.
# Edit the USER CONFIGURATION section to match train_keyrount_lora.sh.

# ========================= USER CONFIGURATION =========================
MODEL='fastmritest'
BASE_PATH='/root/autodl-tmp'
DATASET_TYPE='fastmri_knee'
MASK_TYPE='gaussian'
# Must match the exact directory token used by the training run.
# Exact directory token created by the training run, including commas.
TRAIN_ACC_FACTORS='2x,8x,10x'
# Individual factors to validate; use space-separated values for the loop.
ACC_FACTORS='1x 2x 4x 8x 10x' #推理所采用的采样率，这里以加速因子的形式输入，可以换算为采样率，1x对应全采样，2x对应50%，4x对应25%，以此类推。当前设置的范围在10%到100
MASK_GROUP='seed45'
BATCH_SIZE=1
DEVICE='cuda:0'
# Must match the training entrypoint's PEFT_METHOD. The adapter subdirectory
# below is generated from this value and the same rank/target/gate rules used
# by train_keyrount_lora.py; do not type the long adapter path manually.
PEFT_METHOD='melora'
MELORA_TARGET='down_sample_layers.0,up_sample_layers.0.layers,up_sample_layers.1.layers,up_sample_layers.2.layers'
DORA_RANK=8
DORA_ALPHA=16
PISSA_RANK=8
PISSA_ALPHA=${PISSA_RANK}
# ======================= END USER CONFIGURATION =======================

RUN_DIR="${BASE_PATH}/experiments/${DATASET_TYPE}/${MASK_TYPE}/acc_${TRAIN_ACC_FACTORS}/${MODEL}_${MASK_GROUP}"
# Base checkpoint used together with the adapter.
CHECKPOINT="${BASE_PATH}/experiments/${DATASET_TYPE}/${MASK_TYPE}/acc_8x/${MODEL}/best_model.pt"
DATA_PATH="${BASE_PATH}/datasets/${DATASET_TYPE}/gaussian/test/multi_acc"
USMASK_PATH="${BASE_PATH}/usmasks/"

# Keep this naming logic aligned with train_keyrount_lora.py:
#   melora-family: <root>/r<rank>_<target>_gate_<single|ab>
#   dora/pissa:    peft/<method>/r<rank>_<target>_alpha_<alpha>
TARGET_SUBDIR="${MELORA_TARGET//,/_}"
case "${PEFT_METHOD}" in
  melora)
    MELORA_R='8,8'
    USE_LORA_AB_GATE=0
    ADAPTER_SUBDIR="melora/r${MELORA_R//,/_}_${TARGET_SUBDIR}_gate_single"
    ;;
  shared_lora)
    MELORA_R='8'
    USE_LORA_AB_GATE=0
    ADAPTER_SUBDIR="melora/r${MELORA_R}_${TARGET_SUBDIR}_gate_single"
    ;;
  convlora)
    MELORA_R='8'
    USE_LORA_AB_GATE=0
    ADAPTER_SUBDIR="melora/r${MELORA_R}_${TARGET_SUBDIR}_gate_single"
    ;;
  dora)
    ADAPTER_SUBDIR="peft/dora/r${DORA_RANK}_${TARGET_SUBDIR}_alpha_${DORA_ALPHA}"
    ;;
  pissa)
    ADAPTER_SUBDIR="peft/pissa/r${PISSA_RANK}_${TARGET_SUBDIR}_alpha_${PISSA_ALPHA}"
    ;;
  *)
    echo "Unsupported PEFT_METHOD: ${PEFT_METHOD}. Choose melora, shared_lora, convlora, dora, or pissa." >&2
    exit 1
    ;;
esac

LORA_PATH="${RUN_DIR}/${ADAPTER_SUBDIR}/adapter_best.pt"

for ACC_FACTOR in ${ACC_FACTORS}; do
  OUT_DIR="${RUN_DIR}/results_${ACC_FACTOR}"
  python valid.py \
    --checkpoint "${CHECKPOINT}" \
    --out-dir "${OUT_DIR}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
    --data-path "${DATA_PATH}" \
    --acceleration_factor "${ACC_FACTOR}" \
    --dataset_type "${DATASET_TYPE}" \
    --mask_type "${MASK_TYPE}" \
    --usmask_path "${USMASK_PATH}" \
    --mask_group "${MASK_GROUP}" \
    --use_lora \
    --lora_path "${LORA_PATH}"
done



