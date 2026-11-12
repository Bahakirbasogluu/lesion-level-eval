"""A synthetic cohort for the demo and the tests. No patient data is involved.

Reference masks hold spherical "lesions" whose sizes follow a long-tailed
distribution, so most lesions are small and most voxels belong to a few large
ones. The simulated predictor finds large lesions reliably and small ones
rarely, and adds a few spurious components. Its numbers illustrate the method;
they say nothing about any real model.
"""

from __future__ import annotations

import numpy as np

SHAPE = (48, 48, 48)


def _ball(shape: tuple[int, int, int], centre: np.ndarray, radius: float) -> np.ndarray:
    grid = np.ogrid[tuple(slice(0, n) for n in shape)]
    return sum((g - c) ** 2 for g, c in zip(grid, centre)) <= radius**2


def _place(
    mask: np.ndarray, rng: np.random.Generator, radius: float, clearance: int = 2
) -> np.ndarray | None:
    """Adds a ball that does not touch existing foreground; returns it, or None if no room was found."""
    margin = int(np.ceil(radius)) + clearance
    for _ in range(50):
        centre = rng.integers(margin, np.array(mask.shape) - margin)
        ball = _ball(mask.shape, centre, radius)
        if not (_ball(mask.shape, centre, radius + clearance) & mask).any():
            mask |= ball
            return ball
    return None


def make_cohort(
    n_patients: int = 40, seed: int = 0
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Returns (study, patient, reference, prediction) for one to three studies per patient."""
    rng = np.random.default_rng(seed)
    cohort = []
    for p in range(n_patients):
        skill = rng.uniform(
            0.7, 1.3
        )  # some patients are harder than others, across all their studies
        for s in range(rng.integers(1, 4)):
            reference = np.zeros(SHAPE, dtype=bool)
            prediction = np.zeros(SHAPE, dtype=bool)
            occupied = np.zeros(SHAPE, dtype=bool)
            for _ in range(rng.poisson(5)):
                radius = float(np.clip(rng.lognormal(0.55, 0.65), 0.8, 9.0))
                ball = _place(occupied, rng, radius)
                if ball is None:
                    continue
                reference |= ball
                size = int(ball.sum())
                p_detect = min(0.98, skill * (0.08 + 0.9 * (1 - np.exp(-size / 60))))
                if rng.random() < p_detect:
                    eroded = _ball(SHAPE, np.argwhere(ball).mean(axis=0), max(radius - 0.6, 0.5))
                    prediction |= eroded & ball if eroded.any() else ball
            for _ in range(rng.poisson(0.9)):
                ball = _place(occupied, rng, float(rng.uniform(0.8, 1.6)))
                if ball is not None:
                    prediction |= ball
            cohort.append((f"P{p:03d}-S{s + 1}", f"P{p:03d}", reference, prediction))
    return cohort
