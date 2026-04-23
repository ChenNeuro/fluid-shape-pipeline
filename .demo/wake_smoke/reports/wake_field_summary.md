# Wake-Field Training Summary

- Main variant: `full_half_quarter_hotspot_4ch`
- Main repeated holdout: acc=0.2667±0.0000, macroF1=0.1550±0.0000
- Full only vs main macroF1: 0.1706 -> 0.1550
- Full+half+quarter vs main macroF1: 0.0667 -> 0.1550
- Speed-only vs full 4-channel macroF1: 0.0667 vs 0.1550

## Repeated Holdout Comparison
- full_only_4ch: acc=0.2667±0.0000, macroF1=0.1706±0.0000, dy_MAE=0.02198, eps_MAE=0.04147, IoU=0.2500, Dice=0.3636
- full_half_quarter_hotspot_4ch: acc=0.2667±0.0000, macroF1=0.1550±0.0000, dy_MAE=0.03802, eps_MAE=0.03853, IoU=0.2990, Dice=0.4260
- full_half_quarter_4ch: acc=0.2000±0.0000, macroF1=0.0667±0.0000, dy_MAE=0.03802, eps_MAE=0.04147, IoU=0.2625, Dice=0.3870
- full_half_quarter_hotspot_speed: acc=0.2000±0.0000, macroF1=0.0667±0.0000, dy_MAE=0.02198, eps_MAE=0.03853, IoU=0.2865, Dice=0.4337

## Leave-One-Re-Out (Main Variant)
- Re=100: acc=0.2000, macroF1=0.0667, dy_MAE=0.02806, eps_MAE=0.04083, IoU=0.3674, Dice=0.5052
- Re=200: acc=0.2000, macroF1=0.0667, dy_MAE=0.03402, eps_MAE=0.03911, IoU=0.2846, Dice=0.4257
- Re=300: acc=0.2000, macroF1=0.0667, dy_MAE=0.03056, eps_MAE=0.03485, IoU=0.3942, Dice=0.5356
