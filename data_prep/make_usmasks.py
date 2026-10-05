import numpy as np
import os
import sys
import json
import zlib

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from utils import gaussian_mask


def derive_seed(base_seed, dataset_type, mask_type, acc):
    """Deterministic per-combination seed so every mask file is reproducible."""
    key = f'{dataset_type}|{mask_type}|{acc}|seed{base_seed}'
    return (base_seed + zlib.crc32(key.encode('utf-8'))) % (2 ** 32)


def generate_masks(base_path, image_size=(320, 320), dataset_types=None,
                   acc_factors=(2, 4, 8, 10), seeds=(49,)):
    all_dataset_types = ['mrbrain_t1', 'mrbrain_flair', 'ixi_pd', 'ixi_t2', 'fastmri_knee']
    if dataset_types:
        unknown = [d for d in dataset_types if d not in all_dataset_types]
        if unknown:
            raise ValueError(f'Unknown dataset types: {unknown}. Expected subset of {all_dataset_types}.')
    else:
        dataset_types = all_dataset_types
    mask_types = ['gaussian']

    usmasks_base = os.path.join(base_path, 'usmasks')
    seed_records = []

    for dataset_type in dataset_types:
        for mask_type in mask_types:
            for base_seed in seeds:
                mask_dir = os.path.join(usmasks_base, dataset_type, mask_type, f'seed{base_seed}')
                os.makedirs(mask_dir, exist_ok=True)

                for acc in acc_factors:
                    combo_seed = derive_seed(base_seed, dataset_type, mask_type, acc)
                    np.random.seed(combo_seed)
                    mask_filepath = os.path.join(mask_dir, f'mask_{acc}x.npy')
                    print(f'Generating {mask_type} {acc}x for {dataset_type}, seed{base_seed} (Seed: {combo_seed})')

                    if float(acc) <= 1:
                        # Fully sampled: all ones. gaussian_mask(size, 1.0) would hang here.
                        mask = np.ones(image_size, dtype=float)
                        print('  fully sampled (seed unused)')
                    elif mask_type == 'gaussian':
                        mask = gaussian_mask(image_size, float(acc))
                    else:
                        raise ValueError(f'unsupported mask type {mask_type}')

                    np.save(mask_filepath, mask)
                    seed_records.append({
                        'dataset_type': dataset_type, 'mask_type': mask_type,
                        'group_seed': base_seed, 'acc': acc, 'seed': combo_seed,
                        'file': os.path.relpath(mask_filepath, base_path),
                    })
                    print(f'  [OK] {mask_filepath}: {np.count_nonzero(mask)}/{mask.size} '
                          f'({100*np.count_nonzero(mask)/mask.size:.2f}%)')

    seed_info_path = os.path.join(usmasks_base, 'seed_info.json')
    existing = {}
    if os.path.exists(seed_info_path):
        with open(seed_info_path, encoding='utf-8') as f:
            existing = json.load(f) or {}
    regenerated = {(r['dataset_type'], r['mask_type'], r['group_seed'], r['acc']) for r in seed_records}
    kept = [r for r in existing.get('masks', [])
            if (r['dataset_type'], r['mask_type'], r.get('group_seed', -1), r['acc']) not in regenerated]
    with open(seed_info_path, 'w', encoding='utf-8') as f:
        json.dump({'image_size': list(image_size), 'acc_factors': list(acc_factors),
                   'seeds': list(seeds), 'masks': kept + seed_records},
                  f, indent=2, ensure_ascii=False)
    print(f'Seed info saved to: {seed_info_path}')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate undersampling masks')
    parser.add_argument('--dataset-types', default='fastmri_knee')
    args = parser.parse_args()
    base_path = os.path.dirname(SRC_DIR)
    dataset_types = [d.strip() for d in args.dataset_types.split(',') if d.strip()]
    generate_masks(base_path, dataset_types=dataset_types)