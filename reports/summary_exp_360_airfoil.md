# Obstacle Shape Identification Summary

## Experiment Setup
- Solver mode: `synthetic` (this run used synthetic baseline)
- Shapes: circle, square, triangle, airfoil; Re set: [100, 200, 300]; perturbations per (shape, Re): 30
- Probes: N=32 at outlet x=L; geometry perturbation: dy in [-0.05, 0.05] * H, eps in [0, 0.06]
- Data rows in `features.csv`: 360, feature dimensions: 329
- Case execution: total=360, success=360, failed=0, failure rate=0.00%

## Baseline Split Metrics (Stratified Holdout)
- Split strategy: stratified by `(shape, Re)` with train=288 test=72 (requested test ratio=0.200, applied ratio=0.200)
- Accuracy: 0.9444
- Macro F1: 0.9448

## Multi-Seed Holdout Stability
- Seeds: [40, 41, 42, 43, 44]
- Accuracy mean±std: 0.9139 ± 0.0492
- Macro F1 mean±std: 0.9143 ± 0.0478
- Per-seed table: `reports/holdout_repeats.csv`

## Leave-One-Re-Out Generalization
- Train on Re != 100, test on Re=100 (n=120): accuracy=0.6583, macro F1=0.6532
- Train on Re != 200, test on Re=200 (n=120): accuracy=0.9083, macro F1=0.9088
- Train on Re != 300, test on Re=300 (n=120): accuracy=0.8000, macro F1=0.7873

## Robustness Sweep (|eps| bins)
- |eps|~0.00372: accuracy=0.7778 (n=54)
- |eps|~0.01115: accuracy=0.8679 (n=53)
- |eps|~0.01859: accuracy=0.9778 (n=45)
- |eps|~0.02603: accuracy=0.8667 (n=45)
- |eps|~0.03346: accuracy=0.9189 (n=37)
- |eps|~0.04090: accuracy=0.8913 (n=46)
- |eps|~0.04834: accuracy=0.9722 (n=36)
- |eps|~0.05577: accuracy=0.8409 (n=44)

## Most Influential Features (Permutation Importance)
- Importance method: permutation importance (test split)
- p020_band_4: 0.006804
- p030_f_peak: 0.005443
- adj_corr_lag_mean: 0.004062
- p023_f_peak: 0.003402
- p009_f_peak: 0.002729
- pod_energy_5: 0.000000
- p010_band_1: 0.000000
- p010_band_2: 0.000000
- p010_band_3: 0.000000
- p010_band_4: 0.000000

## Failed Cases
- None

## Key Conclusions
- Outlet probe velocity signatures are separable across obstacle shapes with the current feature set.
- Frequency-domain descriptors are the dominant contributors among top-ranked features.
- Accuracy increases slightly toward higher perturbation bins for this synthetic dataset.

## Next Steps
- Replace synthetic generator with OpenFOAM runs for physics-grounded validation while keeping raw CSV schema unchanged.
- Add more challenging perturbations (x-shift, obstacle rotation, inlet profile changes) and domain randomization.
- Benchmark temporal models (1D CNN / transformer) directly on probe sequences against feature-based baseline.