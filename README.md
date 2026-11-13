# lesion-eval

[![CI](https://github.com/Bahakirbasogluu/lesion-level-eval/actions/workflows/ci.yml/badge.svg)](https://github.com/Bahakirbasogluu/lesion-level-eval/actions/workflows/ci.yml)

Lesion-level evaluation for 3D segmentation masks. It answers the question a voxel-overlap score does not: of the lesions in the reference, how many were found, at which sizes, and at what cost in false positives.

Dice is dominated by large lesions. A mass of thousands of voxels outweighs a dozen small foci, so a model can post a reasonable Dice while missing most of the small ones. This package reports the numbers that expose that, with intervals that respect how the data is structured.

I use this style of evaluation in my MSc thesis on whole-body PET/CT lesion segmentation and wrote about it in [Dice Is Not Detection](https://medium.com/@bahakirbashov/dice-is-not-detection-what-a-0-659-dice-hid-about-small-lesions-a5c7cd7f4c9a). This repository is a standalone implementation of the method. It contains no patient data, no model weights and no results from the thesis; the demo runs on synthetic masks.

## Definitions

- A **lesion** is a 26-connected component of the reference mask.
- A lesion is **detected** if at least one predicted voxel overlaps it.
- A predicted component that overlaps no reference voxel is a **false positive**.
- Lesions are grouped into **size bands** by voxel count: 1-9, 10-49, 50-199, 200-999, 1000-4999, 5000 and above. The edges are a parameter.
- Every interval is a 95% **patient-cluster bootstrap**: patients are resampled with all their studies, because one patient can contribute several studies and many lesions, and those are not independent.

The detection criterion is deliberately lenient (any overlap). It measures whether a lesion was noticed, not how well it was outlined; Dice already covers the second.

## Quick start

```bash
pip install -e ".[dev]"
lesion-eval demo
pytest
```

Score your own masks (`.nii`, `.nii.gz`, `.npy` or `.npz`, same file names in both folders):

```bash
pip install -e ".[nifti]"
lesion-eval score --reference labels/ --prediction predictions/ --patients patients.csv
```

`patients.csv` has the columns `study,patient`. Without it every study is treated as its own patient, which makes the intervals too narrow whenever a patient has more than one study.

## Demo output

`lesion-eval demo` builds a synthetic cohort of 40 patients with one to three studies each. The simulated predictor finds large lesions reliably and small ones rarely. The numbers illustrate the report; they describe no real model.

```
75 studies, 40 patients, 353 reference lesions
mean Dice                  0.495 [0.417, 0.569]
false positives per study  0.880 [0.681, 1.092]
detection, all lesions     46.5% [39.1, 53.8]

size (voxels)  lesions  detected   rate [95% CI]
1-9                145        24   16.6% [11.0, 22.8]
10-49               88        37   42.0% [31.4, 52.7]
50-199              80        68   85.0% [75.3, 94.8]
200-999             31        28   90.3% [78.8, 100.0]
1000-4999            9         7   77.8% [44.4, 100.0]
>=5000               0         0   n/a
```

Note the last two rows: a band with nine lesions has an interval more than fifty points wide, and a band with no lesions is reported as `n/a`, not as zero.

## Python API

```python
from lesion_eval import evaluate_study, summarize, cluster_bootstrap, paired_difference
from lesion_eval.summary import detection_rate, format_table

results = [evaluate_study(reference, prediction, study_id, patient_id) for ... in cohort]
print(format_table(summarize(results)))

# One statistic with its interval: detection of 10-49 voxel lesions (band index 1).
print(cluster_bootstrap(results, detection_rate(band=1)))

# Two methods on the same studies: interval of the difference, on shared resamples.
print(paired_difference(results_a, results_b, detection_rate(band=1)))
```

`paired_difference` evaluates both methods on the same resampled patients, so patient difficulty cancels out. Comparing two separately computed intervals instead is less sensitive and can hide a real difference.

## What to report next to Dice

1. Detection rate per size band, with the band edges stated.
2. False positives per study next to every detection number. Detection without its false-positive cost is a threshold choice, not a result.
3. Intervals from a patient-level bootstrap, including the ones that cross zero.

## Limits

- A false-positive component is counted once whatever its size; the sizes are kept in `StudyResult.false_positive_sizes` if you want to weight them.
- A predicted component that bridges two lesions detects both. A lesion split into several predicted components is detected once.
- Band edges are in voxels. Convert to millilitres with your voxel volume when reporting, since a voxel count means different things on different grids.
- Percentile intervals are unreliable for bands with very few lesions; the lesion count is printed beside every rate for that reason.

## Tests

The suite uses hand-built masks with known answers (sizes, detections, false positives, Dice), checks the band edges, and checks two properties of the bootstrap: resampling patients gives a wider interval than treating clustered studies as independent, and a paired difference is tighter than two separate intervals.

## License

MIT
