# Obstacle Shape Identification Summary

## Experiment Setup
- Solver mode: `synthetic` (this run used synthetic baseline)
- Shapes: circle, triangle, airfoil, diamond, bar; Re set: [100, 200, 300]; perturbations per (shape, Re): 30
- Probes: N=32 at outlet x=L; geometry perturbation: dy in [-0.05, 0.05] * H, eps in [0, 0.06]
- Data rows in `features.csv`: 450, feature dimensions: 329
- Case execution: total=450, success=450, failed=0, failure rate=0.00%

## Baseline Split Metrics (Stratified Holdout)
- Split strategy: stratified by `(shape, Re)` with train=360 test=90 (requested test ratio=0.200, applied ratio=0.200)
- Accuracy: 1.0000
- Macro F1: 1.0000

## Multi-Seed Holdout Stability
- Seeds: [40, 41, 42, 43, 44]
- Accuracy mean±std: 0.9956 ± 0.0054
- Macro F1 mean±std: 0.9956 ± 0.0054
- Per-seed table: `reports/holdout_repeats.csv`

## Leave-One-Re-Out Generalization
- Train on Re != 100, test on Re=100 (n=150): accuracy=0.7933, macro F1=0.7865
- Train on Re != 200, test on Re=200 (n=150): accuracy=1.0000, macro F1=1.0000
- Train on Re != 300, test on Re=300 (n=150): accuracy=0.9400, macro F1=0.9400

## Robustness Sweep (|eps| bins)
- |eps|~0.00374: accuracy=1.0000 (n=44)
- |eps|~0.01123: accuracy=0.9839 (n=62)
- |eps|~0.01872: accuracy=1.0000 (n=54)
- |eps|~0.02621: accuracy=1.0000 (n=52)
- |eps|~0.03370: accuracy=0.9697 (n=66)
- |eps|~0.04119: accuracy=1.0000 (n=50)
- |eps|~0.04868: accuracy=1.0000 (n=63)
- |eps|~0.05617: accuracy=1.0000 (n=59)

## Most Influential Features (Permutation Importance)
- Importance method: random forest impurity importance (fallback, permutation all-zero)
- adj_corr_lag_mean: 0.036630
- adj_corr_lag_std: 0.026768
- p012_a_peak: 0.026408
- p014_a_peak: 0.024615
- p013_a_peak: 0.021154
- p005_f_peak: 0.019905
- p015_a_peak: 0.019679
- p015_f_peak: 0.019014
- p016_f_peak: 0.018704
- p013_f_peak: 0.017489

## Failed Cases
- None

## Key Conclusions
- Outlet probe velocity signatures are separable across obstacle shapes with the current feature set.
- Top contributors are mixed across frequency, statistics, and cross-probe structure features.
- Accuracy remains broadly stable across the tested outlet lens perturbation range.

## Next Steps
- Replace synthetic generator with OpenFOAM runs for physics-grounded validation while keeping raw CSV schema unchanged.
- Add more challenging perturbations (x-shift, obstacle rotation, inlet profile changes) and domain randomization.
- Benchmark temporal models (1D CNN / transformer) directly on probe sequences against feature-based baseline.