# Pattern-Matching Audit

- Config: `exp_360_airfoil.yaml`
- Permutations: `120`

## Main Comparison
- Observed reconstruction: IoU=0.9382, Dice=0.9614, mIoU3=0.9732
- Random-pair control: IoU=0.3565, Dice=0.4706, mIoU3=0.7769
- 1-NN template baseline: IoU=0.7699, Dice=0.8516, mIoU3=0.9137

## Permutation Test (H0: input-target mapping is random)
- Null IoU mean±std: 0.3696 ± 0.0325; p-value=0.008264
- Null mIoU3 mean±std: 0.7821 ± 0.0109; p-value=0.008264

## Duplicate-Like Risk
- Near-duplicate rate (scaled NN dist < 1e-9): 0.000000
- NN distance mean: 7.857437
- NN distance p05: 5.929295

## Conclusion
- Current evidence rejects pure template pattern-matching as the primary explanation.