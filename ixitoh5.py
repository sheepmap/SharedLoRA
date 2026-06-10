import h5py
import numpy as np
import nibabel as nib
import os

def preprocess_ixi_to_h5(nifti_path, output_path, acc_factor=4, mask_type='cartesian',
                         mask_base_path=None, dataset_type='ixi_pd', split='train'):
    """
    将 IXI NIfTI 文件转换为项目所需的 H5 格式
    使用预生成的掩码（从 generate_masks.py 生成的 usmasks 目录加载）

    Args:
        nifti_path: 输入 NIfTI 文件路径
        output_path: 输出 H5 文件路径
        acc_factor: 加速因子 (4, 5, 8)
        mask_type: 掩码类型 ('cartesian' 或 'gaussian')
        mask_base_path: 预生成掩码的基础路径 (usmasks 目录的父目录)
        dataset_type: 数据集类型 ('ixi_pd', 'ixi_t2', 'mrbrain_t1', 'mrbrain_flair')
        split: 数据集划分 ('train', 'validation', 'test')，train 只保存 volfs
    """
    # 加载 NIfTI 文件
    img = nib.load(nifti_path)
    volume = img.get_fdata()  # shape: (H, W, D)

    # 归一化
    volume = (volume - volume.min()) / (volume.max() - volume.min())

    # 获取图像尺寸
    h, w, num_slices = volume.shape

    with h5py.File(output_path, 'w') as f:
        # 保存全采样体积
        f.create_dataset('volfs', data=volume.astype(np.float32))

        # 训练集只保存 volfs，验证集和测试集额外保存欠采样数据
        if split != 'train':
            # 加载预生成的掩码
            if mask_base_path is not None:
                acc_str = f"{acc_factor}x"
                mask_path = os.path.join(mask_base_path, 'usmasks', dataset_type, mask_type, f'mask_{acc_str}.npy')
                if os.path.exists(mask_path):
                    mask = np.load(mask_path)
                    print(f"Loaded pre-generated mask from: {mask_path}")
                    print(f"  Mask shape: {mask.shape}, Image shape: ({h}, {w})")
                else:
                    raise FileNotFoundError(f"Mask file not found: {mask_path}. Please run generate_masks.py first.")
            else:
                raise ValueError("mask_base_path must be provided to load pre-generated masks.")

            # 生成欠采样数据（使用预加载的掩码）
            img_volus = np.zeros_like(volume, dtype=np.float32)
            kspace_volus = np.zeros((h, w, num_slices), dtype=np.complex64)

            for i in range(num_slices):
                slice_img = volume[:, :, i]

                # FFT 到 k 空间
                kspace = np.fft.fft2(slice_img, norm='ortho')

                # 应用预加载的掩码（所有切片使用相同的掩码）
                us_kspace = kspace * mask
                us_img = np.abs(np.fft.ifft2(us_kspace, norm='ortho'))

                img_volus[:, :, i] = us_img
                kspace_volus[:, :, i] = us_kspace

            # 保存欠采样数据
            acc_str = f"{acc_factor}x"
            f.create_dataset(f'img_volus_{acc_str}', data=img_volus.astype(np.float32))
            f.create_dataset(f'kspace_volus_{acc_str}', data=kspace_volus.astype(np.complex64))
            print(f"Saved undersampled data with {acc_factor}x acceleration using pre-generated {mask_type} mask")
        else:
            print(f"Train split: saved volfs only (no undersampled data)")

# 批量处理示例
import os
import glob

def batch_preprocess(input_dir, output_dir, dataset_type='ixi_pd', mask_type='cartesian', 
                     acc_factors=[4, 5, 8], mask_base_path=None):
    """
    批量预处理 IXI 数据集
    
    Args:
        input_dir: 输入 NIfTI 文件目录
        output_dir: 输出 H5 文件目录
        dataset_type: 数据集类型 ('ixi_pd', 'ixi_t2', 'mrbrain_t1', 'mrbrain_flair')
        mask_type: 掩码类型 ('cartesian' 或 'gaussian')
        acc_factors: 加速因子列表 [4, 5, 8]
        mask_base_path: 预生成掩码的基础路径 (usmasks 目录的父目录)
                         如果为 None，则使用 output_dir 的父目录
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果没有提供 mask_base_path，使用 output_dir 的父目录
    if mask_base_path is None:
        mask_base_path = os.path.dirname(os.path.dirname(output_dir))
        # 或者使用当前文件的父目录
        # mask_base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    print(f"Using pre-generated masks from: {os.path.join(mask_base_path, 'usmasks')}")
    print(f"Dataset type: {dataset_type}, Mask type: {mask_type}")
    
    for acc in acc_factors:
        acc_str = f"{acc}x"
        
        # 创建文件夹结构
        train_dir = os.path.join(output_dir, dataset_type, mask_type, 'train', f'acc_{acc_str}')
        val_dir = os.path.join(output_dir, dataset_type, mask_type, 'validation', f'acc_{acc_str}')
        test_dir = os.path.join(output_dir, dataset_type, mask_type, 'test', f'acc_{acc_str}')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # 获取所有 NIfTI 文件
        nifti_files = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))

        # 划分训练集、验证集和测试集 (80% 训练, 10% 验证, 10% 测试)
        n_total = len(nifti_files)
        train_split = int(n_total * 0.8)
        val_split = int(n_total * 0.9)
        train_files = nifti_files[:train_split]
        val_files = nifti_files[train_split:val_split]
        test_files = nifti_files[val_split:]

        print(f"\nProcessing {acc}x acceleration...")
        print(f"  Train files: {len(train_files)}, Val files: {len(val_files)}, Test files: {len(test_files)}")

        # 处理训练集（只保存 volfs）
        for fpath in train_files:
            fname = os.path.basename(fpath).replace('.nii.gz', '.h5')
            output_path = os.path.join(train_dir, fname)
            preprocess_ixi_to_h5(fpath, output_path, acc, mask_type, mask_base_path, dataset_type, split='train')
            print(f"  [Train] Processed: {output_path}")

        # 处理验证集（保存 volfs + img_volus + kspace_volus）
        for fpath in val_files:
            fname = os.path.basename(fpath).replace('.nii.gz', '.h5')
            output_path = os.path.join(val_dir, fname)
            preprocess_ixi_to_h5(fpath, output_path, acc, mask_type, mask_base_path, dataset_type, split='validation')
            print(f"  [Val] Processed: {output_path}")

        # 处理测试集（保存 volfs + img_volus + kspace_volus）
        for fpath in test_files:
            fname = os.path.basename(fpath).replace('.nii.gz', '.h5')
            output_path = os.path.join(test_dir, fname)
            preprocess_ixi_to_h5(fpath, output_path, acc, mask_type, mask_base_path, dataset_type, split='test')
            print(f"  [Test] Processed: {output_path}")
    
    print("\n" + "=" * 60)
    print("Batch preprocessing completed!")
    print("=" * 60)

# 使用示例
if __name__ == '__main__':
    pass
    # 示例1: 使用预生成掩码处理单个文件
    # preprocess_ixi_to_h5(
    #     nifti_path='path/to/brain.nii.gz',
    #     output_path='path/to/output/brain.h5',
    #     acc_factor=4,
    #     mask_type='cartesian',
    #     mask_base_path='path/to/SHFormer-master',  # usmasks 目录的父目录，包含预生成的掩码文件
    #     dataset_type='ixi_pd'
    # )
    
    # 示例2: 批量处理（推荐）
    # 首先运行 generate_masks.py 生成掩码到 SHFormer-master/usmasks/ 目录
    # 然后运行 batch_preprocess，通过 mask_base_path 指定预生成掩码的位置
    # batch_preprocess(
    #     input_dir=r'C:\Users\admin\Desktop\1\代码\ixi\ixi_pd',  # NIfTI 文件目录
    #     output_dir=r'C:\Users\admin\Desktop\1\代码\datasets',     # 输出 H5 文件目录
    #     dataset_type='ixi_pd',
    #     mask_type='cartesian',
    #     acc_factors=[4, 5, 8],
    #     mask_base_path=r'C:\Users\admin\Desktop\1\代码\SHFormer-master'  # 预生成掩码所在路径 (usmasks 目录的父目录)
    # )