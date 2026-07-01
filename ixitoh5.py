import glob
import os

import h5py
import nibabel as nib
import numpy as np


def preprocess_ixi_to_h5(
    nifti_path,
    output_path,
    acc_factor=4,
    mask_type='cartesian',
    mask_base_path=None,
    dataset_type='ixi_pd',
    split='train',
):
    """
    Convert one IXI NIfTI volume to the project's H5 format.

    Train split saves only `volfs`.
    Validation/test save `volfs` plus precomputed undersampled inputs.
    """
    img = nib.load(nifti_path)
    volume = img.get_fdata()

    volume = (volume - volume.min()) / (volume.max() - volume.min())
    h, w, num_slices = volume.shape

    with h5py.File(output_path, 'w') as f:
        f.create_dataset('volfs', data=volume.astype(np.float32))

        if split == 'train':
            print('Train split: saved volfs only (no undersampled data)')
            return

        if mask_base_path is None:
            raise ValueError('mask_base_path must be provided to load pre-generated masks.')

        acc_str = f'{acc_factor}x'
        mask_path = os.path.join(
            mask_base_path, 'usmasks', dataset_type, mask_type, f'mask_{acc_str}.npy'
        )
        if not os.path.exists(mask_path):
            raise FileNotFoundError(
                f'Mask file not found: {mask_path}. Please run generate_masks.py first.'
            )

        mask = np.load(mask_path)
        print(f'Loaded pre-generated mask from: {mask_path}')
        print(f'  Mask shape: {mask.shape}, Image shape: ({h}, {w})')

        img_volus = np.zeros_like(volume, dtype=np.float32)
        kspace_volus = np.zeros((h, w, num_slices), dtype=np.complex64)

        for i in range(num_slices):
            slice_img = volume[:, :, i]
            kspace = np.fft.fft2(slice_img, norm='ortho')
            us_kspace = kspace * mask
            us_img = np.abs(np.fft.ifft2(us_kspace, norm='ortho'))
            img_volus[:, :, i] = us_img
            kspace_volus[:, :, i] = us_kspace

        f.create_dataset(f'img_volus_{acc_str}', data=img_volus.astype(np.float32))
        f.create_dataset(f'kspace_volus_{acc_str}', data=kspace_volus.astype(np.complex64))
        print(f'Saved undersampled data with {acc_factor}x acceleration using pre-generated {mask_type} mask')


def _split_nifti_files(input_dir):
    nifti_files = sorted(glob.glob(os.path.join(input_dir, '*.nii.gz')))
    n_total = len(nifti_files)
    train_split = int(n_total * 0.8)
    val_split = int(n_total * 0.9)
    return {
        'train': nifti_files[:train_split],
        'validation': nifti_files[train_split:val_split],
        'test': nifti_files[val_split:],
    }


def batch_preprocess(
    input_dir,
    output_dir,
    dataset_type='ixi_pd',
    mask_type='cartesian',
    acc_factors=(4, 5, 8),
    mask_base_path=None,
    splits=('train', 'validation', 'test'),
):
    """
    Batch preprocess IXI data.

    `splits` lets train and validation/test be generated independently.
    """
    os.makedirs(output_dir, exist_ok=True)

    if mask_base_path is None:
        mask_base_path = os.path.dirname(os.path.dirname(output_dir))

    valid_splits = ('train', 'validation', 'test')
    splits = tuple(splits)
    invalid_splits = [split for split in splits if split not in valid_splits]
    if invalid_splits:
        raise ValueError(f'Unsupported splits: {invalid_splits}. Expected subset of {valid_splits}.')

    split_to_files = _split_nifti_files(input_dir)

    print(f"Using pre-generated masks from: {os.path.join(mask_base_path, 'usmasks')}")
    print(f'Dataset type: {dataset_type}, Mask type: {mask_type}')
    print(
        f"Split sizes - train: {len(split_to_files['train'])}, "
        f"validation: {len(split_to_files['validation'])}, "
        f"test: {len(split_to_files['test'])}"
    )

    for acc in acc_factors:
        acc_str = f'{acc}x'
        print(f'\nProcessing {acc_str} acceleration...')

        for split in splits:
            split_dir = os.path.join(output_dir, dataset_type, mask_type, split, f'acc_{acc_str}')
            os.makedirs(split_dir, exist_ok=True)
            split_label = 'Val' if split == 'validation' else split.capitalize()

            for fpath in split_to_files[split]:
                fname = os.path.basename(fpath).replace('.nii.gz', '.h5')
                output_path = os.path.join(split_dir, fname)
                preprocess_ixi_to_h5(
                    fpath,
                    output_path,
                    acc_factor=acc,
                    mask_type=mask_type,
                    mask_base_path=mask_base_path,
                    dataset_type=dataset_type,
                    split=split,
                )
                print(f'  [{split_label}] Processed: {output_path}')

    print('\n' + '=' * 60)
    print('Batch preprocessing completed!')
    print('=' * 60)


if __name__ == '__main__':
    pass
