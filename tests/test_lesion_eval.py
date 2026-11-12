import numpy as np
import pytest

from lesion_eval import (
    StudyResult,
    band_of,
    cluster_bootstrap,
    dice,
    evaluate_study,
    paired_difference,
    summarize,
)
from lesion_eval.cli import main
from lesion_eval.io import load_mask
from lesion_eval.summary import detection_rate, false_positives_per_study, mean_dice
from lesion_eval.synthetic import make_cohort


def empty():
    return np.zeros((12, 12, 12), dtype=bool)


def test_known_answer():
    ref, pred = empty(), empty()
    ref[1:3, 1:3, 1:3] = True  # 8 voxels, found
    ref[6:9, 6:9, 6:9] = True  # 27 voxels, missed
    ref[10, 10, 10] = True  # 1 voxel, missed
    pred[2, 2, 2] = True  # one voxel inside the first lesion
    pred[0, 10, 0] = True  # false positive, 1 voxel
    pred[4:6, 0, 0] = True  # false positive, 2 voxels
    r = evaluate_study(ref, pred, "s1", "p1")
    assert sorted(zip(r.lesion_sizes, r.lesion_detected)) == [(1, False), (8, True), (27, False)]
    assert r.false_positives == 2 and sorted(r.false_positive_sizes) == [1, 2]
    assert r.dice == pytest.approx(2 * 1 / (36 + 4))


def test_diagonal_neighbours_are_one_lesion():
    ref = empty()
    ref[1, 1, 1] = ref[2, 2, 2] = (
        True  # touch only at a corner: one component under 26-connectivity
    )
    assert evaluate_study(ref, empty()).lesion_sizes == [2]


def test_a_prediction_bridging_two_lesions_detects_both_and_is_no_false_positive():
    ref, pred = empty(), empty()
    ref[2, 2, 2] = ref[2, 2, 6] = True
    pred[2, 2, 2:7] = True
    r = evaluate_study(ref, pred)
    assert r.lesion_detected == [True, True] and r.false_positives == 0


def test_empty_masks():
    r = evaluate_study(empty(), empty())
    assert (r.dice, r.lesion_sizes, r.false_positives) == (1.0, [], 0)
    pred = empty()
    pred[0, 0, 0] = True
    assert dice(empty(), pred) == 0.0 and evaluate_study(empty(), pred).false_positives == 1


def test_shape_mismatch_is_an_error():
    with pytest.raises(ValueError, match="shape mismatch"):
        evaluate_study(empty(), np.zeros((3, 3, 3), dtype=bool))


@pytest.mark.parametrize(
    ("size", "band"),
    [(1, 0), (9, 0), (10, 1), (49, 1), (50, 2), (999, 3), (1000, 4), (5000, 5), (10**6, 5)],
)
def test_band_edges(size, band):
    assert band_of(size) == band


def study(patient, n, sizes_detected, fp=0, d=0.5):
    sizes, detected = zip(*sizes_detected) if sizes_detected else ((), ())
    return StudyResult(f"{patient}-{n}", patient, d, list(sizes), list(detected), fp)


def test_statistics_pool_lesions_over_studies():
    results = [
        study("a", 1, [(5, True), (5, False), (100, True)], fp=2, d=0.8),
        study("b", 1, [(5, False)], fp=0, d=0.4),
    ]
    assert detection_rate(0)(results) == pytest.approx(1 / 3)
    assert detection_rate(2)(results) == 1.0
    assert detection_rate(None)(results) == 0.5
    assert np.isnan(detection_rate(5)(results))
    assert false_positives_per_study(results) == 1.0 and mean_dice(results) == pytest.approx(0.6)


def test_bootstrap_resamples_patients_not_studies():
    # One patient with ten identical studies and nine patients with one study each.
    results = [study("big", n, [(5, True)]) for n in range(10)] + [
        study(f"p{i}", 1, [(5, False)]) for i in range(9)
    ]
    by_patient = cluster_bootstrap(results, detection_rate(0), n_boot=2000, seed=1)
    as_if_independent = cluster_bootstrap(
        [StudyResult(r.study, r.study, r.dice, r.lesion_sizes, r.lesion_detected, r.false_positives) for r in results],
        detection_rate(0), n_boot=2000, seed=1,
    )  # fmt: skip
    assert by_patient.estimate == as_if_independent.estimate == pytest.approx(10 / 19)
    assert by_patient.low == 0.0
    assert (by_patient.high - by_patient.low) > 1.5 * (
        as_if_independent.high - as_if_independent.low
    )


def test_bootstrap_is_reproducible_and_brackets_the_estimate():
    results = [
        evaluate_study(ref, pred, s, p) for s, p, ref, pred in make_cohort(n_patients=12, seed=3)
    ]
    a = cluster_bootstrap(results, mean_dice, n_boot=500, seed=7)
    b = cluster_bootstrap(results, mean_dice, n_boot=500, seed=7)
    assert a == b and a.low <= a.estimate <= a.high


def test_paired_difference_is_tighter_than_two_separate_intervals():
    base = [study(f"p{i}", 1, [(5, i % 2 == 0)] * 4, d=0.3 + 0.05 * (i % 10)) for i in range(30)]
    better = [
        StudyResult(
            r.study, r.patient, r.dice + 0.02, r.lesion_sizes, r.lesion_detected, r.false_positives
        )
        for r in base
    ]
    diff = paired_difference(base, better, mean_dice, n_boot=1000, seed=0)
    assert (
        diff.estimate == pytest.approx(0.02)
        and diff.low == pytest.approx(0.02)
        and diff.high == pytest.approx(0.02)
    )
    separate = cluster_bootstrap(base, mean_dice, n_boot=1000, seed=0)
    assert separate.high - separate.low > 0.02
    with pytest.raises(ValueError, match="same studies"):
        paired_difference(base, better[1:], mean_dice)


def test_summary_counts_match_the_per_study_results():
    results = [
        evaluate_study(ref, pred, s, p) for s, p, ref, pred in make_cohort(n_patients=10, seed=1)
    ]
    summary = summarize(results, n_boot=200)
    assert (
        summary["lesions"]
        == sum(b["lesions"] for b in summary["bands"])
        == sum(len(r.lesion_sizes) for r in results)
    )
    assert summary["patients"] == 10 and summary["studies"] == len(results)


def test_cli_scores_nifti_and_npy_folders(tmp_path, capsys):
    import nibabel

    ref_dir, pred_dir = tmp_path / "ref", tmp_path / "pred"
    ref_dir.mkdir(), pred_dir.mkdir()
    rows = ["study,patient"]
    for i, (s, p, ref, pred) in enumerate(make_cohort(n_patients=3, seed=2)):
        if i % 2:
            np.save(ref_dir / f"{s}.npy", ref)
            np.save(pred_dir / f"{s}.npy", pred)
        else:
            nibabel.save(
                nibabel.Nifti1Image(ref.astype(np.uint8), np.eye(4)), str(ref_dir / f"{s}.nii.gz")
            )
            nibabel.save(
                nibabel.Nifti1Image(pred.astype(np.uint8), np.eye(4)), str(pred_dir / f"{s}.nii.gz")
            )
        rows.append(f"{s},{p}")
    (tmp_path / "patients.csv").write_text("\n".join(rows), encoding="utf-8")
    assert load_mask(next(ref_dir.glob("*.nii.gz"))).dtype == bool
    assert (
        main(
            [
                "--n-boot",
                "100",
                "score",
                "--reference",
                str(ref_dir),
                "--prediction",
                str(pred_dir),
                "--patients",
                str(tmp_path / "patients.csv"),
            ]
        )
        == 0
    )
    assert "3 patients" in capsys.readouterr().out
