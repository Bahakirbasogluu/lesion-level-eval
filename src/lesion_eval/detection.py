"""Per-study lesion matching.

Definitions:

- A lesion is a 26-connected component of the reference mask.
- A lesion is detected if at least one predicted voxel overlaps it.
- A predicted 26-connected component that overlaps no reference voxel is a false positive.
- Lesions are grouped into size bands by their voxel count.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

CONNECTIVITY_26 = np.ones((3, 3, 3), dtype=bool)
# Lower edges of the size bands in voxels: 1-9, 10-49, 50-199, 200-999, 1000-4999, >=5000.
DEFAULT_BANDS = (1, 10, 50, 200, 1000, 5000)


def band_of(size: int, edges: tuple[int, ...] = DEFAULT_BANDS) -> int:
    """Index of the band a lesion of ``size`` voxels falls into."""
    if size < edges[0]:
        raise ValueError(f"size {size} is below the first band edge {edges[0]}")
    return int(np.searchsorted(edges, size, side="right") - 1)


def band_label(index: int, edges: tuple[int, ...] = DEFAULT_BANDS) -> str:
    return (
        f">={edges[index]}" if index == len(edges) - 1 else f"{edges[index]}-{edges[index + 1] - 1}"
    )


def dice(reference: np.ndarray, prediction: np.ndarray) -> float:
    """Voxel Dice. Two empty masks agree perfectly and score 1."""
    ref, pred = reference.astype(bool), prediction.astype(bool)
    total = int(ref.sum()) + int(pred.sum())
    return 1.0 if total == 0 else 2.0 * int((ref & pred).sum()) / total


@dataclass
class StudyResult:
    study: str
    patient: str
    dice: float
    lesion_sizes: list[int] = field(default_factory=list)
    lesion_detected: list[bool] = field(default_factory=list)
    false_positives: int = 0
    false_positive_sizes: list[int] = field(default_factory=list)


def evaluate_study(
    reference: np.ndarray, prediction: np.ndarray, study: str = "", patient: str | None = None
) -> StudyResult:
    """Match one predicted mask against one reference mask."""
    if reference.shape != prediction.shape:
        raise ValueError(
            f"shape mismatch: reference {reference.shape}, prediction {prediction.shape}"
        )
    ref, pred = reference.astype(bool), prediction.astype(bool)
    result = StudyResult(
        study=study, patient=patient if patient is not None else study, dice=dice(ref, pred)
    )

    ref_labels, n_ref = ndimage.label(ref, structure=CONNECTIVITY_26)
    if n_ref:
        index = np.arange(1, n_ref + 1)
        result.lesion_sizes = ndimage.sum_labels(ref, ref_labels, index).astype(int).tolist()
        result.lesion_detected = (ndimage.maximum(pred, ref_labels, index) > 0).tolist()

    pred_labels, n_pred = ndimage.label(pred, structure=CONNECTIVITY_26)
    if n_pred:
        index = np.arange(1, n_pred + 1)
        touches_reference = ndimage.maximum(ref, pred_labels, index) > 0
        sizes = ndimage.sum_labels(pred, pred_labels, index).astype(int)
        result.false_positives = int((~touches_reference).sum())
        result.false_positive_sizes = sizes[~touches_reference].tolist()
    return result
