# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

This is a fluid mechanics research repository that classifies 2D obstacle shapes using wake-field images. The core approach: in a 2D channel flow, can we identify the upstream obstacle shape (circle/triangle/airfoil/diamond/bar) from its wake pattern?

The **wake-field branch** (`wake_field`) is the relevant pipeline for image-based wake classification using distance-based wake crops.

## Common Commands

```bash
# Wake-field pipeline (most relevant for shape classification from wake images)
make wake-dataset CONFIG=configs/wake_field_450.yaml     # Generate dataset + wake frames
make wake-fields CONFIG=configs/wake_field_450.yaml      # Build wake fields from raw data
make wake-train CONFIG=configs/wake_field_450.yaml       # Train multi-scale classifiers
make wake-reconstruct CONFIG=configs/wake_field_450.yaml # Inverse reconstruction

# Or run the full pipeline at once
make wake-pipeline CONFIG=configs/wake_field_450.yaml

# Traditional probe-based pipeline
make dataset CONFIG=configs/exp_450_biggap.yaml
make train CONFIG=configs/exp_450_biggap.yaml
make sota
make reconstruct
make audit CONFIG=configs/exp_450_biggap.yaml AUDIT_N_PERM=60
```

## Architecture

### Wake-Field Pipeline (Image-Based)

The wake-field pipeline operates on velocity field distributions extracted from particle visualization (laser + tracer particles):

1. **Data Generation** (`sim/generate_dataset.py`): Synthetic unsteady wake signals. Writes `wake_frames.npz` containing short video sequences of wake patterns.

2. **Wake Field Extraction** (`extract/build_wake_fields.py`): Converts raw probe data into multi-scale velocity field crops stored in `wake_fields/`.

3. **Multi-Scale Model** (`vision/wake_model.py`): `MultiScaleWakeNet` using ResNet18 encoder with 2 active variants:
   - `dist_single_4ch`: Single downstream crop, 4 channels (ux, uy, speed, vorticity)
   - `dist_multi_4ch`: Multiple downstream crops combined, 4 channels

4. **Training** (`ml/train_wake.py`): Multi-task learning predicting shape, position (dy), perturbation (eps), and Reynolds number.

5. **Inverse Reconstruction** (`ml/reconstruct_wake.py`): Reconstructs obstacle geometry from predicted parameters.

### Traditional Probe-Based Pipeline

- `sim/`: Case generation, solver adapters, OpenFOAM orchestration
- `extract/`: Feature extraction from probe time series (spectral, statistical features)
- `ml/`: Classification (rf/svc/ensemble) and reconstruction models

### Key Data Schema

- Raw data: `data/raw/<case_id>/probes.csv`, `data/raw/<case_id>/metadata.json`
- Wake fields: `data/wake_fields/` with per-case `.npz` files containing multi-scale crops
- Features: `data/features/features.csv`
- Models: `models/sota.pkl`, `models/wake_field_main.pt`, `models/wake_field_single.pt`

## Config Files

- `configs/wake_field_450.yaml`: Main wake-field experiment (450 cases)
- `configs/wake_field_smoke.yaml`: Smoke test with fewer cases
- `configs/exp_450_biggap.yaml`: Traditional probe-based experiment

## Your Research Direction

Based on your description (classifying shapes from wake images at different downstream distances), the `wake_field` pipeline already implements that distance-based approach. The key files to understand/modify are:

- `vision/wake_dataset.py`: How different scales/crops are defined and loaded
- `ml/wake_common.py`: `VARIANTS` dict defines scale combinations
- `vision/wake_model.py`: `MultiScaleWakeNet` architecture
