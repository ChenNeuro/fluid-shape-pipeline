# Geometry Reconstruction Summary

## Split Setup
- Stratified by (shape, Re), train=144, test=36
- Repeated seeds: [40, 41, 42, 43, 44]
- Selected method: `parametric_inverse`

## Method Leaderboard (Repeated Holdout)
- parametric_inverse: IoU=0.9579±0.0063, Dice=0.9774±0.0036, MSE=0.000323±0.000049, shape_acc=0.9778, dy_MAE=0.00221, eps_MAE=0.02138
- latent_ridge: IoU=0.7302±0.0159, Dice=0.8399±0.0101, MSE=0.000559±0.000048, shape_acc=0.0000, dy_MAE=0.00000, eps_MAE=0.00000

## Leave-One-Re-Out (Selected Method)
- Re=100 (n=60): MSE=0.000497, IoU=0.9098, Dice=0.9484, shape_acc=0.7833, dy_MAE=0.00257, eps_MAE=0.01957
- Re=200 (n=60): MSE=0.000287, IoU=0.9708, Dice=0.9846, shape_acc=0.9833, dy_MAE=0.00218, eps_MAE=0.02355
- Re=300 (n=60): MSE=0.000417, IoU=0.9298, Dice=0.9618, shape_acc=0.9167, dy_MAE=0.00263, eps_MAE=0.02176