"""Patient-cluster bootstrap.

One patient can contribute several studies and many lesions, and those are not
independent. Resampling patients (with all their studies) keeps that dependence
inside each resample; resampling studies or lesions would understate the interval.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from lesion_eval.detection import StudyResult

Statistic = Callable[[Sequence[StudyResult]], float]


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float
    level: float = 0.95

    def __str__(self) -> str:
        return f"{self.estimate:.3f} [{self.low:.3f}, {self.high:.3f}]"


def _clusters(results: Sequence[StudyResult]) -> list[list[int]]:
    by_patient: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(results):
        by_patient[r.patient].append(i)
    return [by_patient[p] for p in sorted(by_patient)]


def _resamples(n_patients: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n_patients, size=(n_boot, n_patients))


def _interval(estimate: float, draws: list[float], level: float) -> Interval:
    values = np.asarray([d for d in draws if not np.isnan(d)])
    if values.size == 0:
        return Interval(estimate, float("nan"), float("nan"), level)
    alpha = (1 - level) / 2
    low, high = np.quantile(values, [alpha, 1 - alpha])
    return Interval(estimate, float(low), float(high), level)


def cluster_bootstrap(
    results: Sequence[StudyResult],
    statistic: Statistic,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile interval of ``statistic`` under resampling of patients.

    A resample in which the statistic is undefined (NaN, for example a size band
    with no lesions) is left out of the percentiles.
    """
    clusters = _clusters(results)
    draws = []
    for picks in _resamples(len(clusters), n_boot, seed):
        sample = [results[i] for p in picks for i in clusters[p]]
        draws.append(statistic(sample))
    return _interval(statistic(results), draws, level)


def paired_difference(
    a: Sequence[StudyResult],
    b: Sequence[StudyResult],
    statistic: Statistic,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Interval of statistic(b) - statistic(a) for two methods run on the same studies.

    Both methods are evaluated on the same resampled patients, so what the two
    share (the difficulty of each patient) cancels out of the difference.
    """
    if [(r.study, r.patient) for r in a] != [(r.study, r.patient) for r in b]:
        raise ValueError("the two result lists must cover the same studies in the same order")
    clusters = _clusters(a)
    draws = []
    for picks in _resamples(len(clusters), n_boot, seed):
        rows = [i for p in picks for i in clusters[p]]
        draws.append(statistic([b[i] for i in rows]) - statistic([a[i] for i in rows]))
    return _interval(statistic(b) - statistic(a), draws, level)
