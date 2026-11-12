"""Loading masks from .npy, .npz and NIfTI files."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

SUFFIXES = (".nii.gz", ".nii", ".npy", ".npz")


def stem(path: Path) -> str:
    name = path.name
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def load_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.name.endswith((".nii", ".nii.gz")):
        try:
            import nibabel
        except ImportError as exc:
            raise ImportError(
                'reading NIfTI needs nibabel: pip install "lesion-eval[nifti]"'
            ) from exc
        data = np.asanyarray(nibabel.load(str(path)).dataobj)
    elif path.suffix == ".npz":
        archive = np.load(path)
        data = archive[archive.files[0]]
    else:
        data = np.load(path)
    if data.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D mask, got {data.ndim} dimensions")
    return data > 0


def masks_in(folder: str | Path) -> dict[str, Path]:
    return {stem(p): p for p in sorted(Path(folder).iterdir()) if p.name.endswith(SUFFIXES)}


def load_patient_map(path: str | Path) -> dict[str, str]:
    """CSV with the columns ``study`` and ``patient``."""
    with open(path, newline="", encoding="utf-8") as f:
        return {row["study"]: row["patient"] for row in csv.DictReader(f)}
