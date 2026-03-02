# SOTA Candidate Summary

- Selected model: `svc_rbf`
- Selection seed (holdout fit): `42`
- Holdout accuracy: 1.0000
- Holdout macro F1: 1.0000

## Candidate Leaderboard (Repeated Holdout)
- svc_rbf: acc=0.9944±0.0111, macroF1=0.9944±0.0111
- hard_vote_ensemble: acc=0.9833±0.0136, macroF1=0.9833±0.0136
- extratrees: acc=0.9778±0.0111, macroF1=0.9777±0.0111
- rf: acc=0.9333±0.0283, macroF1=0.9331±0.0286

## Leave-One-Re-Out
- Re=100: acc=0.8333, macroF1=0.8222 (n=60)
- Re=200: acc=1.0000, macroF1=1.0000 (n=60)
- Re=300: acc=0.9667, macroF1=0.9666 (n=60)

## Robustness Sweep
- |eps|~0.00412: acc=1.0000 (n=50)
- |eps|~0.01236: acc=1.0000 (n=22)
- |eps|~0.02061: acc=1.0000 (n=28)
- |eps|~0.02885: acc=0.9600 (n=25)
- |eps|~0.03709: acc=1.0000 (n=22)
- |eps|~0.04533: acc=1.0000 (n=33)