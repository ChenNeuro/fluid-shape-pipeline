# OpenFOAM Template (Adapter Scaffold)

This template provides a minimal runnable laminar channel case scaffold. The adapter fills placeholders in:
- `system/controlDict`
- `system/blockMeshDict`
- `0/U`
- `constant/transportProperties`

The adapter also runs `scripts/build_geometry.py`, which reads `case_params.json` and writes
`system/obstacleDict` as a geometry hook for downstream mesh scripts.

To fully enable geometry-specific runs for circle/square/triangle plus lens perturbation (`eps`),
extend this hook and mesh generation setup (e.g. add `snappyHexMeshDict` and obstacle STL generation).

Expected parsed outputs in this project:
- `data/raw/<case_id>/probes.csv`
- `data/raw/<case_id>/metadata.json`
