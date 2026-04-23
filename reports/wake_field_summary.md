# Wake-Field Training Summary

- Main variant: `dist_multi_4ch`
- Main repeated holdout: acc=0.8867±0.1064, macroF1=0.8821±0.1142
- Single-scale (dist_only_4ch) vs multi-scale macroF1: 0.8386 -> 0.8821

## Repeated Holdout Comparison
- dist_multi_4ch: acc=0.8867±0.1064, macroF1=0.8821±0.1142, dy_MAE=0.04216, eps_MAE=0.04819, IoU=0.3555, Dice=0.4609
- dist_single_4ch: acc=0.8378±0.0958, macroF1=0.8386±0.0944, dy_MAE=0.04684, eps_MAE=0.04781, IoU=0.3211, Dice=0.4187

## Leave-One-Re-Out (Main Variant)
- Re=100: acc=0.7867, macroF1=0.7660, dy_MAE=0.04044, eps_MAE=0.04626, IoU=0.3842, Dice=0.4899
- Re=200: acc=0.9933, macroF1=0.9933, dy_MAE=0.05093, eps_MAE=0.05049, IoU=0.3070, Dice=0.4066
- Re=300: acc=0.8600, macroF1=0.8549, dy_MAE=0.04149, eps_MAE=0.04705, IoU=0.3207, Dice=0.4097
