import numpy as np
import os
import sys

# src/ root (where utils.py lives); this file sits in src/data_prep/
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from utils import cartesian_mask, gaussian_mask


def generate_masks(base_path, image_size=(256, 256)):
    """
    Generate Cartesian and Gaussian undersampling masks for different acceleration factors
    
    Args:
        base_path: Base path of the project (SHFormer-master)
        image_size: Tuple of (height, width) for the mask size, default (256, 256)
    """
    
    dataset_types = ['mrbrain_t1', 'mrbrain_flair','ixi_pd', 'ixi_t2']
    mask_types = ['cartesian', 'gaussian']
    acc_factors = [4, 5, 8]  # 4x and 5x acceleration
    
    usmasks_base = os.path.join(base_path, 'usmasks')
    
    for dataset_type in dataset_types:
        for mask_type in mask_types:
            for acc in acc_factors:
                # Create directory if not exists
                mask_dir = os.path.join(usmasks_base, dataset_type, mask_type)
                os.makedirs(mask_dir, exist_ok=True)
                
                # Generate mask filename: mask_4x.npy or mask_5x.npy
                mask_filename = f'mask_{acc}x.npy'
                mask_filepath = os.path.join(mask_dir, mask_filename)
                
                print(f'Generating {mask_type} mask with {acc}x acceleration for {dataset_type}...')
                
                # Generate the appropriate mask based on type
                if mask_type == 'cartesian':
                    # cartesian_mask expects shape as (1, H, W) format
                    mask = cartesian_mask((1, image_size[0], image_size[1]), float(acc))
                    # Remove batch dimension and take single slice
                    mask = mask[0]
                elif mask_type == 'gaussian':
                    # gaussian_mask expects size as (H, W) tuple
                    mask = gaussian_mask(image_size, float(acc))
                
                # Save mask as .npy file
                np.save(mask_filepath, mask)
                
                print(f'  [OK] Saved to: {mask_filepath}')
                print(f'    Shape: {mask.shape}, Dtype: {mask.dtype}')
                print(f'    Non-zero elements: {np.count_nonzero(mask)} / {mask.size} ({100*np.count_nonzero(mask)/mask.size:.2f}%)')
                print()
    
    print('=' * 60)
    print('All masks generated successfully!')
    print('=' * 60)


if __name__ == '__main__':
    # Project root (parent of src/); usmasks/ is generated here, matching USMASK_PATH in the training scripts
    base_path = os.path.dirname(SRC_DIR)
    
    print(f'Base path: {base_path}')
    print(f'Masks will be saved to: {os.path.join(base_path, "usmasks")}')
    print()
    
    generate_masks(base_path, image_size=(256, 256))
