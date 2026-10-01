import numpy as np
import os
import sys
import json
import zlib
import argparse

# src/ root (where utils.py lives); this file sits in src/data_prep/
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC_DIR)

from utils import cartesian_mask, gaussian_mask

DATASET_TYPE = 'ixi_t2'
MASK_TYPES = ['cartesian', 'gaussian']
RATES = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]  # target sampling ratios (fraction of k-space kept)


def derive_seed(base_seed, mask_type, rate_pct):
    """Deterministic seed for one (group, mask_type, rate) combination."""
    key = f'{DATASET_TYPE}|{mask_type}|rate{rate_pct}|{base_seed}'
    return zlib.crc32(key.encode('utf-8')) % (2 ** 32)


def generate_group(base_path, image_size, seeds):
    Nx, Ny = image_size
    n_total = Nx * Ny
    usmasks_base = os.path.join(base_path, 'usmasks')
    records = []

    for mask_type in MASK_TYPES:
        for base_seed in seeds:
            out_dir = os.path.join(usmasks_base, DATASET_TYPE, mask_type, f'seed{base_seed}')
            os.makedirs(out_dir, exist_ok=True)

            for rate in RATES:
                pct = int(round(rate * 100))
                combo_seed = derive_seed(base_seed, mask_type, pct)

                # Fully sampled: no randomness involved, identical across seeds.
                if rate >= 1.0:
                    mask = np.ones(image_size, dtype=float)
                    print(f'{mask_type} seed{base_seed} rate{pct}%: fully sampled (seed unused)')
                else:
                    # Reseed the global numpy RNG (used inside cartesian_mask/gaussian_mask).
                    np.random.seed(combo_seed)
                    if mask_type == 'cartesian':
                        # 1D row sampling: pick the integer line count closest to Nx*rate.
                        n_lines = max(1, int(round(Nx * rate)))
                        mask = cartesian_mask((1, Nx, Ny), Nx / n_lines)[0]
                    else:
                        # 2D point sampling: exact point count round(N*rate).
                        n_points = max(1, int(round(n_total * rate)))
                        mask = gaussian_mask(image_size, n_total / n_points)

                mask_path = os.path.join(out_dir, f'mask_rate{pct}.npy')
                np.save(mask_path, mask)

                n_nonzero = int(np.count_nonzero(mask))
                actual_rate = n_nonzero / mask.size
                records.append({
                    'dataset_type': DATASET_TYPE,
                    'mask_type': mask_type,
                    'group_seed': base_seed,
                    'rate_target': rate,
                    'rate_actual': round(actual_rate, 6),
                    'seed': combo_seed,
                    'nonzero': n_nonzero,
                    'file': os.path.relpath(mask_path, base_path),
                })
                print(f'  [OK] {mask_type} seed{base_seed} rate{pct}%: '
                      f'{n_nonzero}/{mask.size} ({100 * actual_rate:.2f}%), seed={combo_seed}')

    info_path = os.path.join(usmasks_base, 'seed_info_rates.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({'dataset_type': DATASET_TYPE, 'image_size': list(image_size),
                   'group_seeds': seeds, 'rates': RATES, 'masks': records},
                  f, indent=2, ensure_ascii=False)
    print(f'Seed info saved to: {info_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate ixi_t2 masks at fixed sampling rates (10/20/40/60/80/100%) '
                    'in multiple seed groups')
    parser.add_argument('--seeds', default='42,43,44,45,46',
                        help='Comma-separated base seeds, one group per seed (default: 42,43,44,45,46)')
    parser.add_argument('--image-size', default='256,256',
                        help='Mask size as H,W (default: 256,256; matches IXI-T2 slices)')
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(',') if s.strip()]
    image_size = tuple(int(v) for v in args.image_size.split(','))
    base_path = os.path.dirname(SRC_DIR)

    print(f'Base path: {base_path}')
    print(f'Group seeds: {seeds}')
    print(f'Rates: {[f"{int(r*100)}%" for r in RATES]}')
    print()

    generate_group(base_path, image_size, seeds)

    print('=' * 60)
    print('All rate-group masks generated successfully!')
    print('=' * 60)
