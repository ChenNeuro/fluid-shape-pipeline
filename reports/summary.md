# Obstacle Shape Identification Summary

## Experiment Setup
- Solver mode: `synthetic` (this run used synthetic baseline)
- Shapes: circle, square, triangle; Re set: [100, 200, 300]; perturbations per (shape, Re): 5
- Probes: N=32 at outlet x=L; geometry perturbation: dy in [-0.04, 0.04] * H, eps in [0, 0.05]
- Data rows in `features.csv`: 45, feature dimensions: 329
- Case execution: total=45, success=45, failed=0, failure rate=0.00%

## Baseline Split Metrics (Stratified Holdout)
- Split strategy: stratified by `(shape, Re)` with train=36 test=9 (requested test ratio=0.200, applied ratio=0.200)
- Accuracy: 1.0000
- Macro F1: 1.0000

## Multi-Seed Holdout Stability
- Seeds: [40, 41, 42, 43, 44]
- Accuracy mean±std: 0.9556 ± 0.0544
- Macro F1 mean±std: 0.9543 ± 0.0560
- Per-seed table: `reports/holdout_repeats.csv`

## Leave-One-Re-Out Generalization
- Train on Re != 100, test on Re=100 (n=15): accuracy=0.8667, macro F1=0.8611
- Train on Re != 200, test on Re=200 (n=15): accuracy=0.8667, macro F1=0.8667
- Train on Re != 300, test on Re=300 (n=15): accuracy=0.8667, macro F1=0.8611

## Robustness Sweep (|eps| bins)
- |eps|~0.00412: accuracy=0.9091 (n=11)
- |eps|~0.01236: accuracy=0.8000 (n=5)
- |eps|~0.02061: accuracy=0.7500 (n=8)
- |eps|~0.02885: accuracy=0.6000 (n=5)
- |eps|~0.03709: accuracy=1.0000 (n=5)
- |eps|~0.04533: accuracy=0.9091 (n=11)

## Most Influential Features (Permutation Importance)
- Importance method: random forest impurity importance (fallback, permutation all-zero)
- p029_f_peak: 0.018180
- p026_f_peak: 0.018130
- p011_band_0: 0.014536
- p010_band_0: 0.014027
- p015_band_5: 0.013540
- p010_f_peak: 0.013245
- p014_band_0: 0.012639
- p016_std: 0.011758
- p015_std: 0.011681
- p024_f_peak: 0.011481

## Failed Cases
- None

## Key Conclusions
- Outlet probe velocity signatures are separable across obstacle shapes with the current feature set.
- Frequency-domain descriptors are the dominant contributors among top-ranked features.
- Accuracy shows a noticeable drop for parts of the perturbation range, indicating sensitivity to outlet geometry mismatch.

## Next Steps
- Replace synthetic generator with OpenFOAM runs for physics-grounded validation while keeping raw CSV schema unchanged.
- Add more challenging perturbations (x-shift, obstacle rotation, inlet profile changes) and domain randomization.
- Benchmark temporal models (1D CNN / transformer) directly on probe sequences against feature-based baseline.