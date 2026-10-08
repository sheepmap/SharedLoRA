#!/usr/bin/env bash
set -e

# TRAIN_ACC_FACTORS must match the training/validation run directory.
# All ACC_FACTORS are aggregated into one report file with an [AVG] row.
MODEL='fastmritest'
BASE_PATH='/root/autodl-tmp'
DATASET_TYPE='fastmri_knee'
MASK_TYPE='gaussian'
# Exact directory token created by the training run, including commas.
TRAIN_ACC_FACTORS='2x,8x,10x'
ACC_FACTORS='2x 4x 8x 10x'
MASK_GROUP='seed45'
RUN_DIR="${BASE_PATH}/experiments/${DATASET_TYPE}/${MASK_TYPE}/acc_${TRAIN_ACC_FACTORS}/${MODEL}_${MASK_GROUP}"
TARGET_PATH="${BASE_PATH}/datasets/${DATASET_TYPE}/gaussian/test/multi_acc"
# evaluate.py appends/replaces one [factor] row per invocation when this is set.
# This token names the single aggregated report file; it is independent of ACC_FACTORS.
REPORT_FILE_ACC_FACTOR="${TRAIN_ACC_FACTORS}"

for ACC_FACTOR in ${ACC_FACTORS}; do
  PREDICTIONS_PATH="${RUN_DIR}/results_${ACC_FACTOR}"
  python evaluate.py \
    --target-path "${TARGET_PATH}" \
    --predictions-path "${PREDICTIONS_PATH}" \
    --report-path "${RUN_DIR}" \
    --acc-factor "${ACC_FACTOR}" \
    --report-file-acc-factor "${REPORT_FILE_ACC_FACTOR}" \
    --mask-type "${MASK_TYPE}" \
    --dataset-type "${DATASET_TYPE}"
done

REPORT_FILE="${RUN_DIR}/report_${DATASET_TYPE}_${MASK_TYPE}_${REPORT_FILE_ACC_FACTOR}.txt"
printf '\nFinal aggregated results:\n'
grep -E '^\[(AVG|VAR)\]' "${REPORT_FILE}"
