"""Lesion-level evaluation of 3D segmentation masks."""

from lesion_eval.bootstrap import Interval, cluster_bootstrap, paired_difference
from lesion_eval.detection import DEFAULT_BANDS, StudyResult, band_of, dice, evaluate_study
from lesion_eval.summary import summarize

__all__ = [
    "DEFAULT_BANDS",
    "Interval",
    "StudyResult",
    "band_of",
    "cluster_bootstrap",
    "dice",
    "evaluate_study",
    "paired_difference",
    "summarize",
]
