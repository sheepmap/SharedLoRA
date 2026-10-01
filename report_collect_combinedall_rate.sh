# Rate-mask variant of report_collect_combinedall.sh.
# Prints the per-rate metric reports written by evaluate_combinedall_rate.sh.
MODEL='IXIT2test_keyrount_l1'
BASE_PATH='/root/autodl-tmp/SHFormer-master'

TRAIN_ACC_FACTORS='rate10,rate20,rate40,rate60,rate80,rate100'
MASK_GROUP='seed42'

for DATASET_TYPE in 'ixi_t2' #'mrbrain_flair' 'ixi_pd' 'ixi_t2'
    do
    for MASK_TYPE in 'cartesian' #'gaussian'
        do
        RUN_DIR=${BASE_PATH}'/experiments/'${DATASET_TYPE}'/'${MASK_TYPE}'/acc_'${TRAIN_ACC_FACTORS}'/'${MODEL}'_'${MASK_GROUP}
        for ACC_FACTOR in 'rate10' 'rate20' 'rate40' 'rate60' 'rate80' 'rate100'
            do
            echo ${DATASET_TYPE}','${MASK_TYPE}','${ACC_FACTOR}
            REPORT_PATH=${RUN_DIR}'/report_'${DATASET_TYPE}'_'${MASK_TYPE}'_'${ACC_FACTOR}'_'${MASK_GROUP}'.txt'
            echo ${REPORT_PATH}
            cat ${REPORT_PATH}
            echo "\n"
            done
        done
    done
