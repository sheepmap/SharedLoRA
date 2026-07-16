"""FastMRI preprocessing controller.

This file is the Python equivalent of the former ``fastmridata.sh`` wrapper.
Edit the configuration below and run::

    python preprocess_fastmri.py

The actual HDF5 conversion is implemented by ``fastmridata.batch_preprocess``.
Input directories are already split; this controller does not repartition
files by ratio.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastmridata import batch_preprocess  # noqa: E402


# Project and data paths.  Update the INPUT_* paths to match the downloaded
# FastMRI directories.  Each input directory should already contain only one
# split's HDF5 files.
BASE_PATH = SRC_DIR.parent
OUTPUT_DIR = BASE_PATH / "datasets"
MASK_BASE_PATH = BASE_PATH

DATASET_TYPE = "fastmri_knee"
MASK_TYPE = "cartesian"

INPUT_TRAIN_DIR = BASE_PATH / "fastmri_knee" / "train"
INPUT_VALIDATION_DIR = BASE_PATH / "fastmri_knee" / "validation"
INPUT_TEST_DIR = BASE_PATH / "fastmri_knee" / "test"


# Split switches.  Set any combination to True; disabled splits are not read.
PROCESS_TRAIN = False
PROCESS_VALIDATION = True
PROCESS_TEST = True


# Train stores only normalized volfs.  Validation/test store one merged HDF5
# per source file with normalized img_volus_4x/kspace_volus_4x, etc.  The
# normalization matches ixitoh5.py: each complete volume is min-max scaled to
# [0, 1] before FFT, masking, and IFFT.
ACC_FACTORS_TRAIN = [4]
ACC_FACTORS_EVAL = [4, 8, 16]

OVERWRITE = False


def preprocess_train() -> None:
    print("=" * 60)
    print(f"Start preprocessing {DATASET_TYPE} train split...")
    print("=" * 60)
    batch_preprocess(
        input_dir=INPUT_TRAIN_DIR,
        output_root=OUTPUT_DIR,
        dataset_type=DATASET_TYPE,
        mask_type=MASK_TYPE,
        mask_base_path=MASK_BASE_PATH,
        split="train",
        acc_factors=ACC_FACTORS_TRAIN,
        overwrite=OVERWRITE,
    )


def preprocess_validation() -> None:
    print("=" * 60)
    print(f"Start preprocessing {DATASET_TYPE} validation split...")
    print("=" * 60)
    batch_preprocess(
        input_dir=INPUT_VALIDATION_DIR,
        output_root=OUTPUT_DIR,
        dataset_type=DATASET_TYPE,
        mask_type=MASK_TYPE,
        mask_base_path=MASK_BASE_PATH,
        split="validation",
        acc_factors=ACC_FACTORS_EVAL,
        merge_eval_acc_factors=True,
        overwrite=OVERWRITE,
    )


def preprocess_test() -> None:
    print("=" * 60)
    print(f"Start preprocessing {DATASET_TYPE} test split...")
    print("=" * 60)
    batch_preprocess(
        input_dir=INPUT_TEST_DIR,
        output_root=OUTPUT_DIR,
        dataset_type=DATASET_TYPE,
        mask_type=MASK_TYPE,
        mask_base_path=MASK_BASE_PATH,
        split="test",
        acc_factors=ACC_FACTORS_EVAL,
        merge_eval_acc_factors=True,
        overwrite=OVERWRITE,
    )


if __name__ == "__main__":
    enabled = False
    if PROCESS_TRAIN:
        enabled = True
        preprocess_train()
    if PROCESS_VALIDATION:
        enabled = True
        preprocess_validation()
    if PROCESS_TEST:
        enabled = True
        preprocess_test()
    if not enabled:
        raise SystemExit(
            "No split enabled. Set PROCESS_TRAIN, PROCESS_VALIDATION, "
            "or PROCESS_TEST to True."
        )

    print("=" * 60)
    print("All FastMRI preprocessing completed!")
    print("=" * 60)
