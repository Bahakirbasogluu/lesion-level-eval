"""Statistics over a set of studies, and the table that puts them next to Dice."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from lesion_eval.bootstrap import Statistic, cluster_bootstrap
from lesion_eval.detection import DEFAULT_BANDS, StudyResult, band_label, band_of


def mean_dice(results: Sequence[StudyResult]) -> float:
    return float(np.mean([r.dice for r in results])) if results else float("nan")


def false_positives_per_study(results: Sequence[StudyResult]) -> float:
    return float(np.mean([r.false_positives for r in results])) if results else float("nan")


def detection_rate(band: int | None = None, edges: tuple[int, ...] = DEFAULT_BANDS) -> Statistic:
    """Fraction of reference lesions detected, pooled over studies; one band or all lesions."""

    def statistic(results: Sequence[StudyResult]) -> float:
        hit = total = 0
        for r in results:
            for size, detected in zip(r.lesion_sizes, r.lesion_detected):
                if band is None or band_of(size, edges) == band:
                    total += 1
                    hit += detected
        return hit / total if total else float("nan")

    return statistic


def summarize(
    results: Sequence[StudyResult],
    edges: tuple[int, ...] = DEFAULT_BANDS,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    def ci(statistic: Statistic) -> dict[str, float]:
        i = cluster_bootstrap(results, statistic, n_boot=n_boot, seed=seed)
        return {"estimate": i.estimate, "low": i.low, "high": i.high}

    bands = []
    for b in range(len(edges)):
        sizes = [
            (s, d)
            for r in results
            for s, d in zip(r.lesion_sizes, r.lesion_detected)
            if band_of(s, edges) == b
        ]
        bands.append(
            {
                "band": band_label(b, edges),
                "lesions": len(sizes),
                "detected": sum(d for _, d in sizes),
                "rate": ci(detection_rate(b, edges)),
            }
        )
    return {
        "studies": len(results),
        "patients": len({r.patient for r in results}),
        "lesions": sum(len(r.lesion_sizes) for r in results),
        "mean_dice": ci(mean_dice),
        "detection_all_lesions": ci(detection_rate(None, edges)),
        "false_positives_per_study": ci(false_positives_per_study),
        "bands": bands,
    }


def format_table(summary: dict[str, Any]) -> str:
    def cell(x: dict[str, float], pct: bool = False) -> str:
        k = 100 if pct else 1
        unit = "%" if pct else ""
        if np.isnan(x["estimate"]):
            return "n/a"
        return (
            f"{x['estimate'] * k:.1f}{unit} [{x['low'] * k:.1f}, {x['high'] * k:.1f}]"
            if pct
            else f"{x['estimate']:.3f} [{x['low']:.3f}, {x['high']:.3f}]"
        )

    lines = [
        f"{summary['studies']} studies, {summary['patients']} patients, {summary['lesions']} reference lesions",
        f"mean Dice                  {cell(summary['mean_dice'])}",
        f"false positives per study  {cell(summary['false_positives_per_study'])}",
        f"detection, all lesions     {cell(summary['detection_all_lesions'], pct=True)}",
        "",
        f"{'size (voxels)':<14}{'lesions':>8}{'detected':>10}   rate [95% CI]",
    ]
    for b in summary["bands"]:
        lines.append(
            f"{b['band']:<14}{b['lesions']:>8}{b['detected']:>10}   {cell(b['rate'], pct=True)}"
        )
    return "\n".join(lines)
