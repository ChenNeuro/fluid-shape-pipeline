# Obstacle Shape Identification Summary

## Experiment Setup
- Solver mode: `synthetic` (this run used synthetic baseline)
- Shapes: circle, square, triangle; Re set: [100, 200, 300]; perturbations per (shape, Re): 5
- Probes: N=32 at outlet x=L; geometry perturbation: dy in [-0.02, 0.02] * H, eps in [0, 0.02]
- Data rows in `features.csv`: 45, feature dimensions: 329
- Case execution: total=45, success=45, failed=0, failure rate=0.00%

## Baseline Split Metrics (Stratified Holdout)
- Split strategy: stratified by `(shape, Re)` with train=36 test=9 (requested test ratio=0.200, applied ratio=0.200)
- Accuracy: 1.0000
- Macro F1: 1.0000

## Leave-One-Re-Out Generalization
- Train on Re != 100, test on Re=100 (n=15): accuracy=0.6667, macro F1=0.5556
- Train on Re != 200, test on Re=200 (n=15): accuracy=1.0000, macro F1=1.0000
- Train on Re != 300, test on Re=300 (n=15): accuracy=0.9333, macro F1=0.9327

## Robustness Sweep (|eps| bins)
- |eps|~0.00165: accuracy=1.0000 (n=11)
- |eps|~0.00495: accuracy=1.0000 (n=5)
- |eps|~0.00824: accuracy=1.0000 (n=8)
- |eps|~0.01154: accuracy=1.0000 (n=5)
- |eps|~0.01484: accuracy=1.0000 (n=5)
- |eps|~0.01813: accuracy=1.0000 (n=11)

## Most Influential Features (Permutation Importance)
- Importance method: random forest impurity importance (fallback, permutation all-zero)
- p029_f_peak: 0.034731
- p004_f_peak: 0.023750
- p001_f_peak: 0.023350
- p025_f_peak: 0.022115
- p020_f_peak: 0.021804
- p024_f_peak: 0.020833
- p002_f_peak: 0.020597
- p022_f_peak: 0.020505
- p000_f_peak: 0.016917
- adj_corr_lag_std: 0.016825

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