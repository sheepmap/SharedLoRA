MODEL='IXIT2test'
BASE_PATH='/root/autodl-tmp/SHFormer-master'

# evaluate.py is adapter-agnostic: it scores reconstruction H5 files only.
# Enable this when valid_combinedall.sh wrote predictions for a MELoRA, DoRA,
# or LoRA-XS adapter. Point to that adapter run's ``results`` directory.
# Default false preserves the original base-model evaluation paths below.
USE_ADAPTER_RESULTS=false
ADAPTER_RESULTS_PATH=''

for DATASET_TYPE in 'ixi_t2' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do
        REPORT_ACC_FACTOR=''
        for ACC_FACTOR in '8x' #'5x' '8x'
            do
            if [ -z "${REPORT_ACC_FACTOR}" ]; then
                REPORT_ACC_FACTOR=${ACC_FACTOR}
            fi
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR}
            TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/validation/acc_'${ACC_FACTOR}
            PREDICTIONS_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${ACC_FACTOR}'/'${MODEL}'/results'
            REPORT_PATH=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${REPORT_ACC_FACTOR}'/'${MODEL}
            if [ "${USE_ADAPTER_RESULTS}" = true ]; then
                if [ -z "${ADAPTER_RESULTS_PATH}" ]; then
                    echo "ADAPTER_RESULTS_PATH must be set when USE_ADAPTER_RESULTS=true." >&2
                    exit 1
                fi
                PREDICTIONS_PATH=${ADAPTER_RESULTS_PATH}
                REPORT_PATH=$(dirname "${ADAPTER_RESULTS_PATH}")
            fi
            python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --report-file-acc-factor ${REPORT_ACC_FACTOR} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
            done
        done
    done
