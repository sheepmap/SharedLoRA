#!/usr/bin/env bash
set -euo pipefail

# Resolve Python entrypoints relative to this script, even from another directory.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# TRAIN_ACC_FACTORS must match the training/validation run directory.
# Each test gets one report; all tests are then averaged per factor.
MODEL='fastmritest'
BASE_PATH='/root/autodl-tmp'
DATASET_TYPE='fastmri_knee'
MASK_TYPE='gaussian'
# Exact directory token created by the training run, including commas.
TRAIN_ACC_FACTORS='2x,8x,10x'
ACC_FACTORS='1x 2x 4x 8x 10x'
MASK_GROUP='seed45'
TESTS='test1 test2 test3 test4 test5'
RUN_DIR="${BASE_PATH}/experiments/${DATASET_TYPE}/${MASK_TYPE}/acc_${TRAIN_ACC_FACTORS}/${MODEL}_${MASK_GROUP}"
TARGET_ROOT="${BASE_PATH}/datasets/${DATASET_TYPE}/gaussian/test"
# evaluate.py appends/replaces one [factor] row per invocation when this is set.
# This token names the single aggregated report file; it is independent of ACC_FACTORS.
REPORT_FILE_ACC_FACTOR="${TRAIN_ACC_FACTORS}"
REPORT_PREFIX="report_${DATASET_TYPE}_${MASK_TYPE}_${REPORT_FILE_ACC_FACTOR}"
SUMMARY_FILE="${RUN_DIR}/${REPORT_PREFIX}_tests_avg.txt"
read -r -a TEST_LIST <<< "${TESTS}"
read -r -a FACTOR_LIST <<< "${ACC_FACTORS}"

if (( ${#TEST_LIST[@]} == 0 || ${#FACTOR_LIST[@]} == 0 )); then
  printf 'TESTS and ACC_FACTORS must not be empty.\n' >&2
  exit 1
fi

# Check every input before changing reports or starting the 25 evaluations.
shopt -s nullglob
for TEST in "${TEST_LIST[@]}"; do
  TARGET_PATH="${TARGET_ROOT}/${TEST}"
  if [[ ! -d "${TARGET_PATH}" ]]; then
    printf 'Missing test directory: %s\n' "${TARGET_PATH}" >&2
    exit 1
  fi
  TARGET_FILES=()
  for TARGET_FILE in "${TARGET_PATH}"/*.h5; do
    [[ -f "${TARGET_FILE}" ]] && TARGET_FILES+=("${TARGET_FILE}")
  done
  if (( ${#TARGET_FILES[@]} == 0 )); then
    printf 'No .h5 files in test directory: %s\n' "${TARGET_PATH}" >&2
    exit 1
  fi
  for ACC_FACTOR in "${FACTOR_LIST[@]}"; do
    PREDICTIONS_PATH="${RUN_DIR}/results_${ACC_FACTOR}"
    for TARGET_FILE in "${TARGET_FILES[@]}"; do
      PREDICTION_FILE="${PREDICTIONS_PATH}/${TARGET_FILE##*/}"
      if [[ ! -f "${PREDICTION_FILE}" ]]; then
        printf 'Missing prediction for %s / %s: %s\n' \
          "${TEST}" "${ACC_FACTOR}" "${PREDICTION_FILE}" >&2
        exit 1
      fi
    done
  done
done

# An interrupted run must not leave an old summary looking like a new result.
rm -f -- "${SUMMARY_FILE}"
for TEST in "${TEST_LIST[@]}"; do
  TARGET_PATH="${TARGET_ROOT}/${TEST}"
  REPORT_FILE="${RUN_DIR}/${REPORT_PREFIX}_${TEST}.txt"
  # evaluate.py appends factors, so start each test report afresh.
  : > "${REPORT_FILE}"
  printf '\nEvaluating %s\n' "${TEST}"
  for ACC_FACTOR in "${FACTOR_LIST[@]}"; do
    PREDICTIONS_PATH="${RUN_DIR}/results_${ACC_FACTOR}"
    python "${SCRIPT_DIR}/evaluate.py" \
      --target-path "${TARGET_PATH}" \
      --predictions-path "${PREDICTIONS_PATH}" \
      --report-path "${RUN_DIR}" \
      --acc-factor "${ACC_FACTOR}" \
      --report-file-acc-factor "${REPORT_FILE_ACC_FACTOR}_${TEST}" \
      --mask-type "${MASK_TYPE}" \
      --dataset-type "${DATASET_TYPE}"
  done
  printf 'Report: %s\n' "${REPORT_FILE}"
  grep -E '^\[AVG\]|^\[VAR\]\[' "${REPORT_FILE}"
done

python "${SCRIPT_DIR}/aggregate_test_reports.py" \
  --report-dir "${RUN_DIR}" \
  --report-prefix "${REPORT_PREFIX}" \
  --tests "${TEST_LIST[@]}" \
  --acc-factors "${FACTOR_LIST[@]}"

printf '\nFinal per-factor averages: %s\n' "${SUMMARY_FILE}"
cat -- "${SUMMARY_FILE}"
