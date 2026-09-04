# prepare_fastmri.py
"""FastMRI preprocessing controller.

Converts fastMRI HDF5 files to the project's volume format via
``fastmri_to_h5.batch_preprocess``. All paths are passed on the command line,
so no code edits are needed:

    python data_prep/prepare_fastmri.py --input-dir ../fastmri --dataset-type fastmri_knee

``--input-dir`` is expected to contain ``train/``, ``validation/`` and
``test/`` sub-directories of already-split fastMRI h5 files; each
sub-directory can be overridden with ``--train-dir`` / ``--validation-dir``
/ ``--test-dir``. Train stores only normalized ``volfs`` (at the first
acceleration factor); validation/test store one merged H5 per source file
with normalized ``img_volus_<acc>`` / ``kspace_volus_<acc>`` for every
factor in ``--acc-factors``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastmri_to_h5 import batch_preprocess  # noqa: E402

# src/data_prep -> src -> project root (contains datasets/ and usmasks/)
BASE_DIR = SRC_DIR.parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description='fastMRI HDF5 -> project H5 preprocessing driver')
    parser.add_argument('--input-dir', required=True,
                        help='Root directory containing train/, validation/ and test/ sub-directories')
    parser.add_argument('--train-dir', default=None, help='Override the train input directory')
    parser.add_argument('--validation-dir', default=None, help='Override the validation input directory')
    parser.add_argument('--test-dir', default=None, help='Override the test input directory')
    parser.add_argument('--dataset-type', default='fastmri_knee',
                        help='Dataset name used for output paths and mask lookup (default: fastmri_knee)')
    parser.add_argument('--mask-type', default='cartesian', help='Mask type (default: cartesian)')
    parser.add_argument('--output-dir', default=str(BASE_DIR / 'datasets'),
                        help='Output root for the datasets tree')
    parser.add_argument('--mask-base-path', default=str(BASE_DIR),
                        help='Directory that contains usmasks/')
    parser.add_argument('--acc-factors-train', default='4',
                        help='Acceleration factors for the train split (default: 4)')
    parser.add_argument('--acc-factors', default='4,8,16',
                        help='Acceleration factors for validation/test, merged per volume (default: 4,8,16)')
    parser.add_argument('--splits', default='train,validation,test',
                        help='Comma list of splits to process (subset of train,validation,test)')
    parser.add_argument('--overwrite', action='store_true', help='Re-write existing output files')
    return parser.parse_args()


def main():
    args = parse_args()
    base = Path(args.input_dir)
    dirs = {
        'train': Path(args.train_dir) if args.train_dir else base / 'train',
        'validation': Path(args.validation_dir) if args.validation_dir else base / 'validation',
        'test': Path(args.test_dir) if args.test_dir else base / 'test',
    }
    splits = tuple(s.strip() for s in args.splits.split(',') if s.strip())
    acc_train = [int(x.strip()) for x in args.acc_factors_train.split(',') if x.strip()]
    acc_eval = [int(x.strip()) for x in args.acc_factors.split(',') if x.strip()]

    runners = {
        'train': (dirs['train'], acc_train),
        'validation': (dirs['validation'], acc_eval),
        'test': (dirs['test'], acc_eval),
    }

    for split in splits:
        if split not in runners:
            raise SystemExit(f'Unsupported split {split!r}; choose from train, validation, test')
        input_dir, acc_factors = runners[split]
        print('=' * 60)
        print(f'Start preprocessing {args.dataset_type} {split} split...')
        print('=' * 60)
        batch_preprocess(
            input_dir=input_dir,
            output_root=Path(args.output_dir),
            dataset_type=args.dataset_type,
            mask_type=args.mask_type,
            mask_base_path=Path(args.mask_base_path),
            split=split,
            acc_factors=acc_factors,
            merge_eval_acc_factors=(split != 'train'),
            overwrite=args.overwrite,
        )

    print('=' * 60)
    print('All FastMRI preprocessing completed!')
    print('=' * 60)


if __name__ == '__main__':
    main()
