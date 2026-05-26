# preprocess_ixi.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ixitoh5 import batch_preprocess

BASE_PATH = 'C:/Users/admin/Desktop/1/代码/SHFormer-master'

# 处理 IXI T2
print("=" * 60)
print("开始处理 IXI T2...")
print("=" * 60)
batch_preprocess(
    input_dir='C:/Users/admin/Desktop/1/代码/SHFormer-master/IXI-T2',
    output_dir=os.path.join(BASE_PATH, 'datasets'),
    dataset_type='ixi_t2',
    mask_type='cartesian',
    acc_factors=[4]
)

# 处理 IXI PD
print("=" * 60)
print("开始处理 IXI PD...")
print("=" * 60)
batch_preprocess(
    input_dir='C:/Users/admin/Desktop/1/代码/SHFormer-master/IXI-PD',
    output_dir=os.path.join(BASE_PATH, 'datasets'),
    dataset_type='ixi_pd',
    mask_type='cartesian',
    acc_factors=[4]
)

print("=" * 60)
print("所有数据预处理完成！")
print("=" * 60)