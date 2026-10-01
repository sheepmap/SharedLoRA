# Rate-mask variant of evaluate_combinedall.sh.
# Scores the reconstructions written by valid_combinedall_rate.sh against the
# fully sampled ground truth (volfs). evaluate.py never touches masks, so the
# only rate-specific parts are the paths and the report filename suffix, which
# carries MASK_GROUP to keep the five seed groups' reports separate.
MODEL='IXIT2test_keyrount_l1'
BASE_PATH='/root/autodl-tmp/SHFormer-master'

TRAIN_ACC_FACTORS='10x,5x,2.5x,1.67x,1.25x,1x'
MASK_GROUP='seed42'

for DATASET_TYPE in 'ixi_t2' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do
        RUN_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${TRAIN_ACC_FACTORS}'/'${MODEL}'_'${MASK_GROUP}
        for ACC_FACTOR in '10x' '5x' '2.5x' '1.67x' '1.25x' '1x'
            do
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR}
            # Ground truth is volfs; eval h5 (with volfs) live in multi_acc/.
            TARGET_PATH=${BASE_PATH}'/datasets/'${DATASET_TYPE}'/'${MASK_TYPE}'/test/multi_acc'
            PREDICTIONS_PATH=${RUN_DIR}'/results_'${ACC_FACTOR}
            REPORT_PATH=${RUN_DIR}
            python evaluate.py --target-path ${TARGET_PATH} --predictions-path ${PREDICTIONS_PATH} --report-path ${REPORT_PATH} --acc-factor ${ACC_FACTOR} --report-file-acc-factor ${ACC_FACTOR}_${MASK_GROUP} --mask-type ${MASK_TYPE} --dataset-type ${DATASET_TYPE}
            done
        done
    done
