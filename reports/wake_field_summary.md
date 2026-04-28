# Wake-Field Training Summary

- Main variant: `dist_multi_4ch`
- Main repeated holdout: acc=0.4000+/-0.0000, macroF1=0.2333+/-0.0000
- Single-scale (`dist_single_4ch`) vs multi-scale macroF1: 0.0667 -> 0.2333

## Repeated Holdout Comparison
- dist_multi_4ch: acc=0.4000+/-0.0000, macroF1=0.2333+/-0.0000, dy_MAE=0.03802, eps_MAE=0.04261, IoU=0.3033, Dice=0.4361
- dist_single_4ch: acc=0.2000+/-0.0000, macroF1=0.0667+/-0.0000, dy_MAE=0.03802, eps_MAE=0.03853, IoU=0.2625, Dice=0.3870

## Leave-One-Re-Out (Main Variant)
- Re=100: acc=0.2000, macroF1=0.0667, dy_MAE=0.02518, eps_MAE=0.05511, IoU=0.3752, Dice=0.5114
- Re=200: acc=0.5000, macroF1=0.4222, dy_MAE=0.03137, eps_MAE=0.03911, IoU=0.3460, Dice=0.4791
- Re=300: acc=0.2000, macroF1=0.0667, dy_MAE=0.02918, eps_MAE=0.03485, IoU=0.3257, Dice=0.4643
