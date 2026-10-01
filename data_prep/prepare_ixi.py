# prepare_ixi.py
"""IXI preprocessing driver.

Converts IXI NIfTI volumes to the project's H5 format via
``ixi_to_h5.batch_preprocess``. All paths are passed on the command line,
so no code edits are needed:

    python data_prep/prepare_ixi.py --input-dir ../ixi --dataset-type ixi_t2

The train split stores only ``volfs`` (at the first acceleration factor);
validation/test additionally store precomputed undersampled image/k-space
pairs for every factor in ``--acc-factors``, merged into one H5 per volume.

By default the train/validation/test split is 80/10/10 inside ``--input-dir``.
Pre-split directories can be given explicitly:

    python data_prep/prepare_ixi.py --input-dir ../ixi --dataset-type ixi_t2 \
        --train-dir ../ixi \
        --validation-dir ../datasets/ixi_t2/cartesian/origintestvalid/valid \
        --test-dir ../datasets/ixi_t2/cartesian/origintestvalid/test
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ixi_to_h5 import batch_preprocess

# src/data_prep -> src -> project root (contains datasets/ and usmasks/)
DEFAULT_BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(description='IXI NIfTI -> project H5 preprocessing driver')
    parser.add_argument('--input-dir', required=True,
                        help='Directory containing the IXI .nii.gz volumes')
    parser.add_argument('--dataset-type', default='ixi_t2',
                        help='Dataset name used for output paths and mask lookup (default: ixi_t2)')
    parser.add_argument('--mask-type', default='cartesian', help='Mask type (default: cartesian)')
    parser.add_argument('--output-dir', default=os.path.join(DEFAULT_BASE_PATH, 'datasets'),
                        help='Output root for the datasets tree')
    parser.add_argument('--mask-base-path', default=DEFAULT_BASE_PATH,
                        help='Directory that contains usmasks/')
    parser.add_argument('--mask-group', default='',
                        help="Optional mask subdirectory under usmasks/<ds>/<mask_type>/, "
                             "e.g. seed42 for rate masks; empty = legacy flat layout")
    parser.add_argument('--volfs-only', action='store_true',
                        help='Store volfs only for validation/test (no img_volus_*/kspace_volus_* '
                             'precompute; undersampled inputs are synthesized at inference, '
                             'as in the rate-mask workflow)')
    parser.add_argument('--acc-factors', default='4,8,16',
                        help='Comma list of acceleration factors; train uses the first, '
                             'validation/test store all factors merged (default: 4,8,16)')
    parser.add_argument('--train-dir', default=None,
                        help='Optional pre-split train directory (default: split --input-dir internally)')
    parser.add_argument('--validation-dir', default=None,
                        help='Optional pre-split validation directory')
    parser.add_argument('--test-dir', default=None,
                        help='Optional pre-split test directory')
    parser.add_argument('--splits', default='train,validation,test',
                        help='Comma list of splits to process (subset of train,validation,test)')
    return parser.parse_args()


def preprocess_dataset(args):
    # Keep tokens as strings ('10', '2.5', ...) so non-integer acceleration
    # factors flow through to mask filenames (mask_2.5x.npy) and h5 keys.
    acc_factors = [x.strip() for x in args.acc_factors.split(',') if x.strip()]
    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())

    split_input_dirs = {}
    if args.train_dir:
        split_input_dirs['train'] = args.train_dir
    if args.validation_dir:
        split_input_dirs['validation'] = args.validation_dir
    if args.test_dir:
        split_input_dirs['test'] = args.test_dir
    split_input_dirs = split_input_dirs or None

    if 'train' in splits:
        print('=' * 60)
        print(f"Start preprocessing {args.dataset_type} train split...")
        print('=' * 60)
        batch_preprocess(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            mask_type=args.mask_type,
            acc_factors=[acc_factors[0]],
            mask_base_path=args.mask_base_path,
            splits=('train',),
            split_input_dirs=split_input_dirs,
            mask_group=args.mask_group,
            volfs_only=args.volfs_only,
        )

    eval_splits = tuple(s for s in ('validation', 'test') if s in splits)
    if eval_splits:
        print('=' * 60)
        print(f"Start preprocessing {args.dataset_type} validation/test splits...")
        print('=' * 60)
        batch_preprocess(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            dataset_type=args.dataset_type,
            mask_type=args.mask_type,
            acc_factors=acc_factors,
            mask_base_path=args.mask_base_path,
            splits=eval_splits,
            merge_eval_acc_factors=True,
            split_input_dirs=split_input_dirs,
            mask_group=args.mask_group,
            volfs_only=args.volfs_only,
        )


if __name__ == '__main__':
    args = parse_args()
    preprocess_dataset(args)
    print('=' * 60)
    print('All dataset preprocessing completed!')
    print('=' * 60)
