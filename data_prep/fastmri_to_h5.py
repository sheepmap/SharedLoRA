"""Convert FastMRI knee HDF5 files to the project's volume format.

The source FastMRI files store image reconstructions as
``reconstruction_rss`` with shape ``[slices, height, width]``.  The project
expects ``volfs`` with shape ``[height, width, slices]``.  Training files only
need ``volfs``; validation and test files can additionally contain the
precomputed undersampled image/k-space pairs used by ``valid.py``.

Example::

    python data_prep/fastmri_to_h5.py \
        --input-dir /data/fastmri/knee_train \
        --output-root /data/SHFormer/datasets \
        --dataset-type fastmri_knee \
        --mask-type cartesian \
        --mask-base-path /data/SHFormer \
        --split train \
        --acc-factors 4

    python data_prep/fastmri_to_h5.py \
        --input-dir /data/fastmri/knee_val \
        --output-root /data/SHFormer/datasets \
        --dataset-type fastmri_knee \
        --mask-type cartesian \
        --mask-base-path /data/SHFormer \
        --split validation \
        --acc-factors 4,8,16 \
        --merge-eval-acc-factors
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

try:
    import numpy as np
except ModuleNotFoundError as exc:  # pragma: no cover - environment check
    raise SystemExit(
        "fastmri_to_h5.py requires numpy and h5py. Install them in the Python "
        "environment used for preprocessing."
    ) from exc

try:
    import h5py
except ModuleNotFoundError as exc:  # pragma: no cover - environment check
    raise SystemExit(
        "fastmri_to_h5.py requires h5py. Install it in the Python environment "
        "used for preprocessing (for example: pip install h5py)."
    ) from exc


VALID_SPLITS = ("train", "validation", "test")


def parse_acc_factors(value: str) -> list[int]:
    """Parse ``4,8,16`` or ``4x,8x,16x`` into positive integer factors."""

    factors: list[int] = []
    for item in str(value).split(","):
        item = item.strip().lower()
        if not item:
            continue
        if item.endswith("x"):
            item = item[:-1]
        try:
            factor = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid acceleration factor: {item!r}") from exc
        if factor <= 0:
            raise ValueError(f"Acceleration factor must be positive: {factor}")
        if factor not in factors:
            factors.append(factor)
    if not factors:
        raise ValueError("At least one acceleration factor is required")
    return factors


def factor_label(factor: int) -> str:
    return f"{factor}x"


def _chunk_shape(height: int, width: int, slices: int) -> tuple[int, int, int]:
    """Return conservative chunks for slice-wise HDF5 writes."""

    return min(height, 64), min(width, 64), 1


def _copy_root_attributes(source: h5py.File, target: h5py.File) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value


def _validate_rss(source: h5py.File, path: Path) -> tuple[h5py.Dataset, int, int, int]:
    if "reconstruction_rss" not in source:
        raise KeyError(f"{path}: missing dataset /reconstruction_rss")

    rss = source["reconstruction_rss"]
    if not isinstance(rss, h5py.Dataset) or rss.ndim != 3:
        shape = getattr(rss, "shape", None)
        raise ValueError(
            f"{path}: /reconstruction_rss must be a 3-D dataset [S,H,W], got {shape}"
        )

    slices, height, width = (int(v) for v in rss.shape)
    if min(slices, height, width) <= 0:
        raise ValueError(f"{path}: /reconstruction_rss has invalid shape {rss.shape}")
    return rss, slices, height, width


def _load_mask(mask_base_path: Path, dataset_type: str, mask_type: str,
               factor: int, height: int, width: int) -> np.ndarray:
    mask_path = (
        mask_base_path
        / "usmasks"
        / dataset_type
        / mask_type
        / f"mask_{factor_label(factor)}.npy"
    )
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = np.load(mask_path)
    if mask.ndim != 2 or mask.shape != (height, width):
        raise ValueError(
            f"Mask {mask_path} has shape {mask.shape}; expected {(height, width)}"
        )
    return mask


def _volume_min_max(rss: h5py.Dataset, slices: int) -> tuple[float, float]:
    """Return the min/max over one complete RSS volume.

    IXI preprocessing normalizes each complete volume, rather than each
    individual slice.  Compute the same statistics here while keeping the
    source HDF5 dataset slice-wise to avoid loading the whole file at once.
    """

    volume_min = np.inf
    volume_max = -np.inf
    for slice_index in range(slices):
        image = np.asarray(rss[slice_index, :, :], dtype=np.float64)
        volume_min = min(volume_min, float(image.min()))
        volume_max = max(volume_max, float(image.max()))
    return volume_min, volume_max


def _normalize_slice(
    image: np.ndarray, volume_min: float, volume_max: float
) -> np.ndarray:
    """Apply IXI-style per-volume min-max normalization to one slice."""

    image = np.asarray(image, dtype=np.float32)
    value_range = volume_max - volume_min
    if value_range <= 0:
        # IXI normally has a non-constant volume, but avoid NaNs for a
        # degenerate input volume.
        return np.zeros_like(image, dtype=np.float32)
    return ((image - volume_min) / value_range).astype(np.float32, copy=False)


def _write_volfs(
    target: h5py.Dataset,
    rss: h5py.Dataset,
    slices: int,
    volume_min: float,
    volume_max: float,
) -> None:
    """Write normalized [S,H,W] RSS data as [H,W,S]."""

    for slice_index in range(slices):
        image = _normalize_slice(rss[slice_index, :, :], volume_min, volume_max)
        target[:, :, slice_index] = image


def _write_undersampled_fields(
    output: h5py.File,
    rss: h5py.Dataset,
    slices: int,
    height: int,
    width: int,
    mask: np.ndarray,
    factor: int,
    volume_min: float,
    volume_max: float,
) -> None:
    """Generate validation/test fields from the normalized RSS volume."""

    label = factor_label(factor)
    chunks = _chunk_shape(height, width, slices)
    image_ds = output.create_dataset(
        f"img_volus_{label}",
        shape=(height, width, slices),
        dtype=np.float32,
        chunks=chunks,
    )
    kspace_ds = output.create_dataset(
        f"kspace_volus_{label}",
        shape=(height, width, slices),
        dtype=np.complex64,
        chunks=chunks,
    )

    for slice_index in range(slices):
        # Use the same normalized image as volfs.  The FFT/mask/IFFT process
        # intentionally matches ixi_to_h5.py.
        image = _normalize_slice(rss[slice_index, :, :], volume_min, volume_max)
        kspace = np.fft.fft2(image, norm="ortho")
        undersampled_kspace = kspace * mask
        undersampled_image = np.abs(
            np.fft.ifft2(undersampled_kspace, norm="ortho")
        )
        image_ds[:, :, slice_index] = undersampled_image.astype(
            np.float32, copy=False
        )
        kspace_ds[:, :, slice_index] = undersampled_kspace.astype(
            np.complex64, copy=False
        )


def output_directory(
    output_root: Path,
    dataset_type: str,
    mask_type: str,
    split: str,
    factors: list[int],
    merge_eval_acc_factors: bool,
) -> Path:
    split_root = output_root / dataset_type / mask_type / split
    if split == "train":
        return split_root / f"acc_{factor_label(factors[0])}"
    if merge_eval_acc_factors:
        return split_root / "multi_acc"
    return split_root / f"acc_{factor_label(factors[0])}"


def process_file(
    source_path: Path,
    output_path: Path,
    split: str,
    factors: list[int],
    mask_base_path: Path | None,
    dataset_type: str,
    mask_type: str,
    overwrite: bool,
) -> bool:
    """Convert one file and return whether a new output was written."""

    if output_path.exists() and not overwrite:
        print(f"[skip] {output_path} already exists (use --overwrite to replace)")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(output_path.name + ".tmp")
    if temp_path.exists():
        temp_path.unlink()

    try:
        with h5py.File(source_path, "r") as source:
            rss, slices, height, width = _validate_rss(source, source_path)
            volume_min, volume_max = _volume_min_max(rss, slices)
            if split != "train" and mask_base_path is None:
                raise ValueError("--mask-base-path is required for validation/test")

            masks: dict[int, np.ndarray] = {}
            if split != "train":
                assert mask_base_path is not None
                for factor in factors:
                    masks[factor] = _load_mask(
                        mask_base_path,
                        dataset_type,
                        mask_type,
                        factor,
                        height,
                        width,
                    )

            with h5py.File(temp_path, "w") as target:
                _copy_root_attributes(source, target)
                target.attrs["volfs_source"] = "reconstruction_rss"
                target.attrs["volfs_source_layout"] = "[S,H,W]"
                target.attrs["volfs_layout"] = "[H,W,S]"
                target.attrs["normalization"] = "per-volume min-max to [0, 1]"
                target.attrs["normalization_min"] = volume_min
                target.attrs["normalization_max"] = volume_max

                volfs = target.create_dataset(
                    "volfs",
                    shape=(height, width, slices),
                    dtype=np.float32,
                    chunks=_chunk_shape(height, width, slices),
                )
                _write_volfs(volfs, rss, slices, volume_min, volume_max)

                if split != "train":
                    for factor in factors:
                        _write_undersampled_fields(
                            target,
                            rss,
                            slices,
                            height,
                            width,
                            masks[factor],
                            factor,
                            volume_min,
                            volume_max,
                        )

        os.replace(temp_path, output_path)
        print(
            f"[ok] {source_path.name}: volfs=({height},{width},{slices}), "
            f"dtype=float32, normalized=[0,1], split={split}"
        )
        if split != "train":
            print(f"     generated factors: {', '.join(factor_label(f) for f in factors)}")
        return True
    except Exception:
        if temp_path.exists():
            temp_path.unlink()
        raise


def iter_h5_files(input_dir: Path) -> Iterable[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".h5"
    )


def batch_preprocess(
    input_dir: str | Path,
    output_root: str | Path,
    dataset_type: str = "fastmri_knee",
    mask_type: str = "cartesian",
    mask_base_path: str | Path | None = None,
    split: str = "train",
    acc_factors: Iterable[int] = (4,),
    merge_eval_acc_factors: bool = False,
    overwrite: bool = False,
) -> tuple[int, int]:
    """Process all HDF5 files in one already-defined dataset split.

    This is the programmatic entry point used by ``prepare_fastmri.py``.
    The function deliberately does not split files by ratio: the caller gives
    it the directory that is already designated as train, validation, or test.
    """

    if split not in VALID_SPLITS:
        raise ValueError(f"Unsupported split {split!r}; expected {VALID_SPLITS}")

    input_path = Path(input_dir).expanduser().resolve()
    output_path = Path(output_root).expanduser().resolve()
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_path}")

    factors = list(acc_factors)
    if not factors:
        raise ValueError("At least one acceleration factor is required")
    factors = parse_acc_factors(",".join(str(factor) for factor in factors))

    if split == "train" and len(factors) != 1:
        raise ValueError("Train preprocessing accepts exactly one factor")
    if split != "train" and len(factors) > 1 and not merge_eval_acc_factors:
        raise ValueError(
            "Multiple validation/test factors require merge_eval_acc_factors=True"
        )

    mask_path = None
    if mask_base_path is not None:
        mask_path = Path(mask_base_path).expanduser().resolve()

    files = list(iter_h5_files(input_path))
    if not files:
        raise FileNotFoundError(f"No .h5 files found directly under {input_path}")

    destination = output_directory(
        output_path,
        dataset_type,
        mask_type,
        split,
        factors,
        merge_eval_acc_factors,
    )
    destination.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Processing {split} split")
    print(f"Input : {input_path}")
    print(f"Output: {destination}")
    print(f"Files : {len(files)}")
    if split != "train":
        print(f"Factors: {', '.join(factor_label(f) for f in factors)}")
    print("=" * 60)

    written = 0
    skipped = 0
    for source_path in files:
        output_file = destination / source_path.name
        if process_file(
            source_path,
            output_file,
            split,
            factors,
            mask_path,
            dataset_type,
            mask_type,
            overwrite,
        ):
            written += 1
        else:
            skipped += 1

    print("=" * 60)
    print(f"Completed: written={written}, skipped={skipped}")
    print("=" * 60)
    return written, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert FastMRI reconstruction_rss data to project volfs format"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-type", default="fastmri_knee")
    parser.add_argument("--mask-type", default="cartesian")
    parser.add_argument(
        "--mask-base-path",
        type=Path,
        default=None,
        help="Project root containing usmasks/; required for validation/test",
    )
    parser.add_argument("--split", choices=VALID_SPLITS, required=True)
    parser.add_argument(
        "--acc-factors",
        required=True,
        help="Comma-separated factors, for example 4 or 4,8,16",
    )
    parser.add_argument(
        "--merge-eval-acc-factors",
        action="store_true",
        help="Store all validation/test factors in one multi_acc directory",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_preprocess(
        input_dir=args.input_dir,
        output_root=args.output_root,
        dataset_type=args.dataset_type,
        mask_type=args.mask_type,
        mask_base_path=args.mask_base_path,
        split=args.split,
        acc_factors=parse_acc_factors(args.acc_factors),
        merge_eval_acc_factors=args.merge_eval_acc_factors,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
