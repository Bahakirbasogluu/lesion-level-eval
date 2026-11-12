from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from lesion_eval.detection import DEFAULT_BANDS, evaluate_study
from lesion_eval.io import load_mask, load_patient_map, masks_in
from lesion_eval.summary import format_table, summarize
from lesion_eval.synthetic import make_cohort


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lesion-eval")
    parser.add_argument(
        "--bands", default=",".join(map(str, DEFAULT_BANDS)), help="lower band edges in voxels"
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print the summary as JSON")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="score a synthetic cohort")
    score = sub.add_parser("score", help="score predicted masks against reference masks")
    score.add_argument("--reference", type=Path, required=True, help="folder of reference masks")
    score.add_argument(
        "--prediction",
        type=Path,
        required=True,
        help="folder of predicted masks with the same file names",
    )
    score.add_argument(
        "--patients",
        type=Path,
        help="CSV with columns study,patient; without it each study is its own patient",
    )
    args = parser.parse_args(argv)
    edges = tuple(int(x) for x in args.bands.split(","))

    if args.command == "demo":
        results = [
            evaluate_study(ref, pred, study, patient)
            for study, patient, ref, pred in make_cohort(seed=args.seed)
        ]
    else:
        references, predictions = masks_in(args.reference), masks_in(args.prediction)
        missing = sorted(set(references) - set(predictions))
        if missing:
            parser.error(
                f"no prediction for: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}"
            )
        patients = load_patient_map(args.patients) if args.patients else {}
        if args.patients and (unmapped := sorted(set(references) - set(patients))):
            parser.error(f"no patient for: {', '.join(unmapped[:5])}")
        results = [
            evaluate_study(
                load_mask(path), load_mask(predictions[study]), study, patients.get(study)
            )
            for study, path in references.items()
        ]
    summary = summarize(results, edges, n_boot=args.n_boot, seed=args.seed)
    if args.json:
        json.dump(summary, sys.stdout, indent=2)
        print()
    else:
        print(format_table(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
