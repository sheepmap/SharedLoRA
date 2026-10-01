import numpy as np
import os
import sys
import json
import zlib

# src/ root (where utils.py lives); this file sits in src/data_prep/
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from utils import cartesian_mask, gaussian_mask


def derive_seed(base_seed, dataset_type, mask_type, acc):
    """Deterministic per-combination seed so every mask file is reproducible."""
    key = f'{dataset_type}|{mask_type}|{acc}'
    return (base_seed + zlib.crc32(key.encode('utf-8'))) % (2 ** 32)


def generate_masks(base_path, image_size=(256, 256), seed=42, dataset_types=None):
    """
    Generate Cartesian and Gaussian undersampling masks for different acceleration factors

    Args:
        base_path: Base path of the project (SHFormer-master)
        image_size: Tuple of (height, width) for the mask size, default (256, 256)
        seed: Base random seed; each (dataset, mask_type, acc) combination gets a
              deterministic derived seed, so re-running reproduces identical masks.
        dataset_types: Subset of datasets to (re)generate, e.g. ['ixi_t2'].
                       None/empty means all datasets.
    """

    all_dataset_types = ['mrbrain_t1', 'mrbrain_flair', 'ixi_pd', 'ixi_t2']
    if dataset_types:
        unknown = [d for d in dataset_types if d not in all_dataset_types]
        if unknown:
            raise ValueError(f'Unknown dataset types: {unknown}. Expected subset of {all_dataset_types}.')
    else:
        dataset_types = all_dataset_types
    mask_types = ['cartesian', 'gaussian']
    acc_factors = [4, 5, 8]  # 4x and 5x acceleration

    usmasks_base = os.path.join(base_path, 'usmasks')
    seed_records = []
    
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

                # Reseed the global numpy RNG (used inside cartesian_mask/gaussian_mask)
                # before each generation so results are reproducible.
                combo_seed = derive_seed(seed, dataset_type, mask_type, acc)
                np.random.seed(combo_seed)
                print(f'  Seed: {combo_seed}')

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

                seed_records.append({
                    'dataset_type': dataset_type,
                    'mask_type': mask_type,
                    'acc': acc,
                    'seed': combo_seed,
                    'file': os.path.relpath(mask_filepath, base_path),
                })
                
                print(f'  [OK] Saved to: {mask_filepath}')
                print(f'    Shape: {mask.shape}, Dtype: {mask.dtype}')
                print(f'    Non-zero elements: {np.count_nonzero(mask)} / {mask.size} ({100*np.count_nonzero(mask)/mask.size:.2f}%)')
                print()
    
    seed_info_path = os.path.join(usmasks_base, 'seed_info.json')
    # Preserve records for masks not touched by this run (partial regeneration).
    existing = {}
    if os.path.exists(seed_info_path):
        with open(seed_info_path, encoding='utf-8') as f:
            existing = json.load(f) or {}
    regenerated = {(r['dataset_type'], r['mask_type'], r['acc']) for r in seed_records}
    kept = [r for r in existing.get('masks', [])
            if (r['dataset_type'], r['mask_type'], r['acc']) not in regenerated]
    with open(seed_info_path, 'w', encoding='utf-8') as f:
        json.dump({'base_seed': seed, 'image_size': list(image_size),
                   'masks': kept + seed_records}, f, indent=2, ensure_ascii=False)
    print(f'Seed info saved to: {seed_info_path}')

    print('=' * 60)
    print('All masks generated successfully!')
    print('=' * 60)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate undersampling masks')
    parser.add_argument('--seed', type=int, default=42,
                        help='Base random seed for reproducible mask generation (default: 42)')
    parser.add_argument('--dataset-types', default='',
                        help='Comma-separated subset of datasets to generate, e.g. "ixi_t2". '
                             'Default: all (mrbrain_t1,mrbrain_flair,ixi_pd,ixi_t2)')
    args = parser.parse_args()

    # Project root (parent of src/); usmasks/ is generated here, matching USMASK_PATH in the training scripts
    base_path = os.path.dirname(SRC_DIR)

    print(f'Base path: {base_path}')
    print(f'Base seed: {args.seed}')
    dataset_types = [d.strip() for d in args.dataset_types.split(',') if d.strip()]
    print(f'Datasets: {", ".join(dataset_types) if dataset_types else "all"}')
    print(f'Masks will be saved to: {os.path.join(base_path, "usmasks")}')
    print()

    generate_masks(base_path, image_size=(256, 256), seed=args.seed, dataset_types=dataset_types)
