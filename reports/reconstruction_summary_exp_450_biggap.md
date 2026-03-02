# Geometry Reconstruction Summary

## Split Setup
- Stratified by (shape, Re), train=360, test=90
- Repeated seeds: [40, 41, 42, 43, 44]
- Selected method: `parametric_inverse`

## Method Leaderboard (Repeated Holdout)
- parametric_inverse: IoU=0.9472±0.0050, Dice=0.9689±0.0036, mIoU3=0.9774±0.0020, MSE=0.000171±0.000017, shape_acc=0.9978, dy_MAE=0.00129, eps_MAE=0.01898
- latent_ridge: IoU=0.4200±0.0236, Dice=0.5439±0.0210, mIoU3=0.7914±0.0080, MSE=0.000612±0.000024, shape_acc=0.0000, dy_MAE=0.00000, eps_MAE=0.00000

## Leave-One-Re-Out (Selected Method)
- Re=100 (n=150): MSE=0.000422, IoU=0.8798, Dice=0.9184, mIoU3=0.9530, shape_acc=0.9200, dy_MAE=0.00212, eps_MAE=0.02556
- Re=200 (n=150): MSE=0.000167, IoU=0.9405, Dice=0.9643, mIoU3=0.9755, shape_acc=1.0000, dy_MAE=0.00127, eps_MAE=0.01829
- Re=300 (n=150): MSE=0.000325, IoU=0.9211, Dice=0.9464, mIoU3=0.9683, shape_acc=0.9400, dy_MAE=0.00151, eps_MAE=0.02050