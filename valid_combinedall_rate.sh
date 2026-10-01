# Rate-mask variant of valid_combinedall.sh.
# Evaluates a model trained with data_prep/make_usmasks_rates.py masks
# (see train_keyrount_lora_rate.sh). MASK_GROUP must match the group used
# at training time: usmasks/<ds>/<mask_type>/<MASK_GROUP>/mask_rate<R>.npy.
# Undersampled inputs are synthesized on the fly from volfs (dataset.py
# SliceDataDev synthesizes whenever --mask_group is set).
MODEL='IXIT2test_keyrount_l1'
BASE_PATH='/root/autodl-tmp/SHFormer-master'
BATCH_SIZE=1
DEVICE='cuda:0'
USMASK_PATH=${BASE_PATH}'/usmasks/'

# Merged acceleration list used by the rate training run (train_keyrount_lora_rate.sh);
# only used to locate its experiment directory.
TRAIN_ACC_FACTORS='rate10,rate20,rate40,rate60,rate80,rate100'
MASK_GROUP='seed42'

# Generic PEFT adapter inference, same semantics as valid_combinedall.sh.
USE_LORA=false
LORA_PATH=''
ADAPTER_RESULTS_PATH=''

for DATASET_TYPE in 'ixi_t2' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do
        RUN_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${TRAIN_ACC_FACTORS}'/'${MODEL}'_'${MASK_GROUP}
        for ACC_FACTOR in 'rate10' 'rate20' 'rate40' 'rate60' 'rate80' 'rate100'
            do
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR}
            CHECKPOINT=${RUN_DIR}'/best_model.pt'
            OUT_DIR=${RUN_DIR}'/results_'${ACC_FACTOR}
            # Rate mode only reads volfs from the h5, so any acc_* directory works.
            DATA_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/test/acc_4x'
            if [ "${USE_LORA}" = true ]; then
                if [ -z "${LORA_PATH}" ]; then
                    echo "LORA_PATH must point to adapter.pt or adapter_best.pt when USE_LORA=true." >&2
                    exit 1
                fi
                if [ -n "${ADAPTER_RESULTS_PATH}" ]; then
                    OUT_DIR=${ADAPTER_RESULTS_PATH}
                fi
                echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --mask_group ${MASK_GROUP} --use_lora --lora_path ${LORA_PATH}
                python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --mask_group ${MASK_GROUP} --use_lora --lora_path ${LORA_PATH}
            else
                echo python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --mask_group ${MASK_GROUP}
                python valid.py --checkpoint ${CHECKPOINT} --out-dir ${OUT_DIR} --batch-size ${BATCH_SIZE} --device ${DEVICE} --data-path ${DATA_PATH} --acceleration_factor ${ACC_FACTOR} --dataset_type ${DATASET_TYPE} --mask_type ${MASK_TYPE} --usmask_path ${USMASK_PATH} --mask_group ${MASK_GROUP}
            fi
            done
        done
    done
