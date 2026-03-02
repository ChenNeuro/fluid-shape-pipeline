# Geometry Reconstruction Summary

## Split Setup
- Stratified by (shape, Re), train=144, test=36
- Repeated seeds: [40, 41, 42, 43, 44]

## Repeated Holdout Metrics
- MSE mean±std: 0.000559 ± 0.000043
- IoU mean±std: 0.7302 ± 0.0142
- Dice mean±std: 0.8399 ± 0.0091

## Leave-One-Re-Out
- Re=100 (n=60): MSE=0.000826, IoU=0.6849, Dice=0.8025
- Re=200 (n=60): MSE=0.000561, IoU=0.7354, Dice=0.8428
- Re=300 (n=60): MSE=0.000617, IoU=0.7293, Dice=0.8399