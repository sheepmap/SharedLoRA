# testdatamake.py
import h5py
import numpy as np
import nibabel as nib
import os
import glob

def preprocess_ixi_to_h5(nifti_path, output_path, acc_factor=4, mask_type='cartesian', 
                         mask_base_path=None, dataset_type='ixi_pd'):
    """
    将 IXI NIfTI 文件转换为项目所需的 H5 格式
    使用预生成的掩码（从 generate_masks.py 生成的 usmasks 目录加载）
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata()
    
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    h, w, num_slices = volume.shape
    
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
    
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volfs', data=volume.astype(np.float32))
        
        img_volus = np.zeros_like(volume, dtype=np.float32)
        kspace_volus = np.zeros((h, w, num_slices), dtype=np.complex64)
        
        for i in range(num_slices):
            slice_img = volume[:, :, i]
            kspace = np.fft.fft2(slice_img, norm='ortho')
            us_kspace = kspace * mask
            us_img = np.abs(np.fft.ifft2(us_kspace, norm='ortho'))
            
            img_volus[:, :, i] = us_img
            kspace_volus[:, :, i] = us_kspace
        
        acc_str = f"{acc_factor}x"
        f.create_dataset(f'img_volus_{acc_str}', data=img_volus.astype(np.float32))
        f.create_dataset(f'kspace_volus_{acc_str}', data=kspace_volus.astype(np.complex64))
        print(f"Saved undersampled data with {acc_factor}x acceleration using pre-generated {mask_type} mask")


def batch_convert(input_dir, output_dir, dataset_type='ixi_pd', mask_type='cartesian', 
                 acc_factors=[4], mask_base_path=None):
    """
    批量转换 NIfTI 为 H5 格式（不划分数据集）
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if mask_base_path is None:
        mask_base_path = os.path.dirname(os.path.dirname(output_dir))
    
    print(f"Using pre-generated masks from: {os.path.join(mask_base_path, 'usmasks')}")
    print(f"Dataset type: {dataset_type}, Mask type: {mask_type}")
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    
    for acc in acc_factors:
        acc_str = f"{acc}x"
        
        out_acc_dir = os.path.join(output_dir, dataset_type, mask_type, f'acc_{acc_str}')
        os.makedirs(out_acc_dir, exist_ok=True)
        
        nifti_files = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))
        
        if not nifti_files:
            print(f"  Warning: No .nii.gz files found in {input_dir}")
            continue
        
        print(f"\nProcessing {acc}x acceleration...")
        print(f"  Total files: {len(nifti_files)}")
        
        for fpath in nifti_files:
            fname = os.path.basename(fpath).replace('.nii.gz', '.h5')
            output_path = os.path.join(out_acc_dir, fname)
            preprocess_ixi_to_h5(fpath, output_path, acc, mask_type, mask_base_path, dataset_type)
            print(f"  Processed: {output_path}")
    
    print("\n" + "=" * 60)
    print("Batch conversion completed!")
    print("=" * 60)


if __name__ == '__main__':
    INPUT_DIR = '/path/to/IXI-T2'
    OUTPUT_DIR = '/path/to/datasets'
    DATASET_TYPE = 'ixi_t2'
    MASK_TYPE = 'cartesian'
    ACC_FACTORS = [4]
    MASK_BASE_PATH = '/path/to/SHFormer-master'
    
    batch_convert(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        dataset_type=DATASET_TYPE,
        mask_type=MASK_TYPE,
        acc_factors=ACC_FACTORS,
        mask_base_path=MASK_BASE_PATH
    )