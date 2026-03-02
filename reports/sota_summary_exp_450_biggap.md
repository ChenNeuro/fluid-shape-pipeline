# SOTA Candidate Summary

- Selected model: `extratrees`
- Selection seed (holdout fit): `42`
- Holdout accuracy: 1.0000
- Holdout macro F1: 1.0000

## Candidate Leaderboard (Repeated Holdout)
- extratrees: acc=0.9978±0.0044, macroF1=0.9978±0.0044
- hard_vote_ensemble: acc=0.9978±0.0044, macroF1=0.9978±0.0044
- rf: acc=0.9956±0.0054, macroF1=0.9956±0.0054
- svc_rbf: acc=0.9956±0.0054, macroF1=0.9956±0.0054

## Leave-One-Re-Out
- Re=100: acc=0.9200, macroF1=0.9188 (n=150)
- Re=200: acc=1.0000, macroF1=1.0000 (n=150)
- Re=300: acc=0.9267, macroF1=0.9263 (n=150)

## Robustness Sweep
- |eps|~0.00374: acc=1.0000 (n=44)
- |eps|~0.01123: acc=0.9839 (n=62)
- |eps|~0.01872: acc=1.0000 (n=54)
- |eps|~0.02621: acc=0.9808 (n=52)
- |eps|~0.03370: acc=1.0000 (n=66)
- |eps|~0.04119: acc=1.0000 (n=50)
- |eps|~0.04868: acc=1.0000 (n=63)
- |eps|~0.05617: acc=1.0000 (n=59)