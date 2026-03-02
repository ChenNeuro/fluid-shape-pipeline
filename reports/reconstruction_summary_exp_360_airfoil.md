# Geometry Reconstruction Summary

## Split Setup
- Stratified by (shape, Re), train=288, test=72
- Repeated seeds: [40, 41, 42, 43, 44]
- Selected method: `parametric_inverse`

## Method Leaderboard (Repeated Holdout)
- parametric_inverse: IoU=0.9340±0.0099, Dice=0.9569±0.0083, MSE=0.000242±0.000043, shape_acc=0.9778, dy_MAE=0.00163, eps_MAE=0.02093
- latent_ridge: IoU=0.4402±0.0143, Dice=0.5541±0.0159, MSE=0.000706±0.000021, shape_acc=0.0000, dy_MAE=0.00000, eps_MAE=0.00000

## Leave-One-Re-Out (Selected Method)
- Re=100 (n=120): MSE=0.000883, IoU=0.7485, Dice=0.7932, shape_acc=0.7667, dy_MAE=0.00198, eps_MAE=0.02349
- Re=200 (n=120): MSE=0.000237, IoU=0.9158, Dice=0.9491, shape_acc=0.9750, dy_MAE=0.00160, eps_MAE=0.02108
- Re=300 (n=120): MSE=0.000557, IoU=0.8543, Dice=0.8918, shape_acc=0.8500, dy_MAE=0.00247, eps_MAE=0.02531