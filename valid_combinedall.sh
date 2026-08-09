MODEL='<model name or experiment name>'
BASE_PATH='<base path>'
BATCH_SIZE=1
DEVICE='cuda:0'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# Generic PEFT adapter inference. The legacy variable names and CLI flags are
# retained so MELoRA commands continue to work unchanged.
# Supported adapter files: MELoRA, DoRA, and LoRA-XS. valid.py detects the
# method from LORA_PATH's sibling checkpoint.pt; do not pass it manually.
# Set USE_LORA=true and point LORA_PATH to adapter.pt or adapter_best.pt.
USE_LORA=false
# Path to PEFT adapter file (required when USE_LORA=true).
# MELoRA example:  .../melora/<run>/adapter_best.pt
# DoRA example:    .../peft/dora/<run>/adapter_best.pt
# LoRA-XS example: .../peft/lora-xs/<run>/adapter_best.pt
LORA_PATH=''
# Optional output override for adapter runs. Leave blank to preserve the
# historical MELoRA output directory; set to e.g. $(dirname ${LORA_PATH})/results
# for DoRA/LoRA-XS so their predictions are stored separately.
ADAPTER_RESULTS_PATH=''

for DATASET_TYPE in 'mrbrain_t1' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do
        for ACC_FACTOR in '4x' #'5x' '8x'
            do
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR}
            CHECKPOINT=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/best_model.pt'
            OUT_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/test/acc_'${ACC_FACTOR}
            if [ "${USE_LORA}" = true ]; then
                if [ -z "${LORA_PATH}" ]; then
                    echo "LORA_PATH must point to adapter.pt or adapter_best.pt when USE_LORA=true." >&2
                    exit 1
                fi
                # Optional separation keeps old MELoRA output paths unchanged.
                if [ -n "${ADAPTER_RESULTS_PATH}" ]; then
                    OUT_DIR=${ADAPTER_RESULTS_PATH}
                fi
                echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --use_lora --lora_path ${LORA_PATH}
                python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --use_lora --lora_path ${LORA_PATH}
            else
                echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE}
                python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH}
            fi
            done
        done
    done

