# Pattern-Matching Audit

- Config: `exp_450_biggap.yaml`
- Permutations: `60`

## Main Comparison
- Observed reconstruction: IoU=0.9414, Dice=0.9657, mIoU3=0.9745
- Random-pair control: IoU=0.3286, Dice=0.4492, mIoU3=0.7669
- 1-NN template baseline: IoU=0.8113, Dice=0.8787, mIoU3=0.9269

## Permutation Test (H0: input-target mapping is random)
- Null IoU mean±std: 0.3317 ± 0.0256; p-value=0.016393
- Null mIoU3 mean±std: 0.7685 ± 0.0086; p-value=0.016393

## Duplicate-Like Risk
- Near-duplicate rate (scaled NN dist < 1e-9): 0.000000
- NN distance mean: 5.477078
- NN distance p05: 3.320746

## Conclusion
- Shortcut risk remains non-negligible; strengthen controls or expand OOD tests.