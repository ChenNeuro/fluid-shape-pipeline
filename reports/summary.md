# Obstacle Shape Identification Summary

## Experiment Setup
- Solver mode: `synthetic` (this run used synthetic baseline)
- Shapes: circle, square, triangle; Re set: [100, 200, 300]; perturbations per (shape, Re): 20
- Probes: N=32 at outlet x=L; geometry perturbation: dy in [-0.04, 0.04] * H, eps in [0, 0.05]
- Data rows in `features.csv`: 180, feature dimensions: 329
- Case execution: total=180, success=180, failed=0, failure rate=0.00%

## Baseline Split Metrics (Stratified Holdout)
- Split strategy: stratified by `(shape, Re)` with train=144 test=36 (requested test ratio=0.200, applied ratio=0.200)
- Accuracy: 0.9722
- Macro F1: 0.9722

## Multi-Seed Holdout Stability
- Seeds: [40, 41, 42, 43, 44]
- Accuracy mean±std: 0.9333 ± 0.0283
- Macro F1 mean±std: 0.9331 ± 0.0286
- Per-seed table: `reports/holdout_repeats.csv`

## Leave-One-Re-Out Generalization
- Train on Re != 100, test on Re=100 (n=60): accuracy=0.9000, macro F1=0.8960
- Train on Re != 200, test on Re=200 (n=60): accuracy=0.9333, macro F1=0.9332
- Train on Re != 300, test on Re=300 (n=60): accuracy=0.8833, macro F1=0.8833

## Robustness Sweep (|eps| bins)
- |eps|~0.00412: accuracy=0.9200 (n=50)
- |eps|~0.01236: accuracy=1.0000 (n=22)
- |eps|~0.02061: accuracy=0.9643 (n=28)
- |eps|~0.02885: accuracy=0.9200 (n=25)
- |eps|~0.03709: accuracy=0.9545 (n=22)
- |eps|~0.04533: accuracy=0.8788 (n=33)

## Most Influential Features (Permutation Importance)
- Importance method: permutation importance (test split)
- p028_f_peak: 0.018024
- p027_f_peak: 0.016638
- p001_f_peak: 0.016638
- p000_f_peak: 0.015251
- p012_f_peak: 0.013865
- adj_corr_lag_std: 0.012478
- p017_f_peak: 0.009705
- p017_band_2: 0.009705
- p022_f_peak: 0.009705
- p013_f_peak: 0.009705

## Failed Cases
- None

## Key Conclusions
- Outlet probe velocity signatures are separable across obstacle shapes with the current feature set.
- Frequency-domain descriptors are the dominant contributors among top-ranked features.
- Accuracy remains broadly stable across the tested outlet lens perturbation range.

## Next Steps
- Replace synthetic generator with OpenFOAM runs for physics-grounded validation while keeping raw CSV schema unchanged.
- Add more challenging perturbations (x-shift, obstacle rotation, inlet profile changes) and domain randomization.
- Benchmark temporal models (1D CNN / transformer) directly on probe sequences against feature-based baseline.