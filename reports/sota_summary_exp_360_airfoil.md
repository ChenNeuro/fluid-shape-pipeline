# SOTA Candidate Summary

- Selected model: `svc_rbf`
- Selection seed (holdout fit): `42`
- Holdout accuracy: 0.9722
- Holdout macro F1: 0.9721

## Candidate Leaderboard (Repeated Holdout)
- svc_rbf: acc=0.9861±0.0124, macroF1=0.9861±0.0125
- extratrees: acc=0.9778±0.0111, macroF1=0.9778±0.0111
- hard_vote_ensemble: acc=0.9778±0.0111, macroF1=0.9778±0.0111
- rf: acc=0.9139±0.0492, macroF1=0.9143±0.0478

## Leave-One-Re-Out
- Re=100: acc=0.8750, macroF1=0.8683 (n=120)
- Re=200: acc=0.9833, macroF1=0.9833 (n=120)
- Re=300: acc=0.9500, macroF1=0.9505 (n=120)

## Robustness Sweep
- |eps|~0.00372: acc=0.9815 (n=54)
- |eps|~0.01115: acc=0.9623 (n=53)
- |eps|~0.01859: acc=1.0000 (n=45)
- |eps|~0.02603: acc=0.9778 (n=45)
- |eps|~0.03346: acc=0.9730 (n=37)
- |eps|~0.04090: acc=0.9783 (n=46)
- |eps|~0.04834: acc=1.0000 (n=36)
- |eps|~0.05577: acc=0.9545 (n=44)