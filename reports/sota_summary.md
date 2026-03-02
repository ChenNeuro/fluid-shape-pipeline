# SOTA Candidate Summary

- Selected model: `hard_vote_ensemble`
- Selection seed (holdout fit): `42`
- Holdout accuracy: 1.0000
- Holdout macro F1: 1.0000

## Candidate Leaderboard (Repeated Holdout)
- hard_vote_ensemble: acc=0.9778±0.0444, macroF1=0.9771±0.0457
- rf: acc=0.9556±0.0544, macroF1=0.9543±0.0560
- svc_rbf: acc=0.9556±0.0544, macroF1=0.9543±0.0560
- extratrees: acc=0.9333±0.0544, macroF1=0.9314±0.0560

## Leave-One-Re-Out
- Re=100: acc=0.8000, macroF1=0.7802 (n=15)
- Re=200: acc=0.8667, macroF1=0.8667 (n=15)
- Re=300: acc=0.7333, macroF1=0.7333 (n=15)

## Robustness Sweep
- |eps|~0.00412: acc=0.9091 (n=11)
- |eps|~0.01236: acc=1.0000 (n=5)
- |eps|~0.02061: acc=0.7500 (n=8)
- |eps|~0.02885: acc=0.8000 (n=5)
- |eps|~0.03709: acc=1.0000 (n=5)
- |eps|~0.04533: acc=0.9091 (n=11)