# preprocess_ixi.py
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ixitoh5 import batch_preprocess

BASE_PATH = 'C:/Users/admin/Desktop/1/代码/SHFormer-master'
OUTPUT_DIR = os.path.join(BASE_PATH, 'datasets')
MASK_BASE_PATH = BASE_PATH
ACC_FACTORS_EVAL = [4, 8, 16]
IXI_T2_VALID_DIR = os.path.join(
    BASE_PATH, 'datasets', 'ixi_t2', 'cartesian', 'origintestvalid', 'valid'
)
IXI_T2_TEST_DIR = os.path.join(
    BASE_PATH, 'datasets', 'ixi_t2', 'cartesian', 'origintestvalid', 'test'
)


def preprocess_dataset(dataset_type, input_dir, mask_type='cartesian', split_input_dirs=None):
    print('=' * 60)
    print(f'Start preprocessing {dataset_type} train split...')
    print('=' * 60)
    batch_preprocess(
        input_dir=input_dir,
        output_dir=OUTPUT_DIR,
        dataset_type=dataset_type,
        mask_type=mask_type,
        acc_factors=[ACC_FACTORS_EVAL[0]],
        mask_base_path=MASK_BASE_PATH,
        splits=('train',),
    )

    print('=' * 60)
    print(f'Start preprocessing {dataset_type} validation/test splits...')
    print('=' * 60)
    batch_preprocess(
        input_dir=input_dir,
        output_dir=OUTPUT_DIR,
        dataset_type=dataset_type,
        mask_type=mask_type,
        acc_factors=ACC_FACTORS_EVAL,
        mask_base_path=MASK_BASE_PATH,
        splits=('validation', 'test'),
        merge_eval_acc_factors=True,
        split_input_dirs=split_input_dirs,
    )


preprocess_dataset(
    dataset_type='ixi_t2',
    input_dir=os.path.join(BASE_PATH, 'IXI-T2'),
    split_input_dirs={
        'validation': IXI_T2_VALID_DIR,
        'test': IXI_T2_TEST_DIR,
    },
)

preprocess_dataset(
    dataset_type='ixi_pd',
    input_dir=os.path.join(BASE_PATH, 'IXI-PD'),
)

print('=' * 60)
print('All dataset preprocessing completed!')
print('=' * 60)
