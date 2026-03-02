# Fluid Camera: CFD + ML End-to-End Pipeline

可复现实验仓库：在 2D 通道流中用出口截面探针速度 `u(t)` 识别上游障碍物形状（`circle/square/triangle`）。

## Visual Preview

![Pipeline overview GIF](reports/pipeline_overview.gif)

后端支持：
- `synthetic`（默认）：合成非定常尾迹信号，保证无 CFD 环境也能完整跑通。
- `openfoam`：OpenFOAM 自动化适配层（模板 + 命令封装 + probes 解析）。

## 1. Environment

- Python 3.11
- Dependencies: `numpy scipy pandas scikit-learn matplotlib pyyaml`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2. One-Command Workflow

### Dataset + Features

```bash
make dataset
```

默认使用 `configs/default.yaml`（小数据集，`3*3*3=27` cases）。

### Train + Reports

```bash
make train
```

### Train SOTA candidate (model search + best model export)

```bash
make sota
```

### Build preview GIF

```bash
make gif
```

### 45-case 扩展实验

```bash
make dataset CONFIG=configs/exp_45.yaml
make train CONFIG=configs/exp_45.yaml
```

### 挑战性 45-case 压测（更强扰动）

```bash
make dataset CONFIG=configs/challenging_45.yaml
make train CONFIG=configs/challenging_45.yaml
```

## 3. Required Outputs

- Raw probe series: `data/raw/<case_id>/probes.csv`
- Case metadata: `data/raw/<case_id>/metadata.json`
- Run manifest: `data/raw/manifest.csv`
- Metadata index: `data/raw/index.csv`
- Features table: `data/features/features.csv`
- Model: `models/baseline.pkl`
- SOTA candidate model (best from model search): `models/sota.pkl`
- Reports:
  - `reports/pipeline_overview.gif`
  - `reports/separability_pca.png`
  - `reports/spectra_examples.png`
  - `reports/confusion_matrix.png`
  - `reports/robustness_sweep.png`
  - `reports/holdout_stability.png`
  - `reports/holdout_repeats.csv`
  - `reports/summary.md`
  - `reports/model_leaderboard.csv`
  - `reports/sota_repeats.csv`
  - `reports/confusion_matrix_sota.png`
  - `reports/robustness_sweep_sota.png`
  - `reports/sota_summary.md`

## 4. Data Schema

### `probes.csv`

Columns:
- `time`
- `u_000 ... u_031` (default `N=32`)

### `metadata.json`

Includes:
- case identifiers: `case_id`, `shape`, `Re`, `dy`, `eps`, `seed`
- geometry fields: `H, d, x0, y0, L_in, L_out`
- probe layout: count, `x`, all `y` positions
- sampling window: `dt`, `n_samples`, `t_start`, `t_end`
- backend and file references

### `features.csv`

Per case row:
- labels/meta: `case_id, shape, Re, dy, eps, seed`
- per-probe: `mean/std/f_peak/a_peak/band_0..band_5`
- cross-probe: adjacent xcorr peak and lag stats
- POD energy ratios: `pod_energy_1..pod_energy_5`

## 5. Modeling and Evaluation

- Baseline model: `RandomForestClassifier`
- Holdout split: stratified by `(shape, Re)`
- Multi-seed stability: repeated holdout across `ml.repeat_seeds`
- Generalization test: Leave-One-Re-Out
- Metrics: `accuracy`, `macro F1`, confusion matrix
- Robustness sweep: `|eps|` bin vs cross-validated accuracy

## 6. Switching Backend

### Synthetic backend (default)

```bash
make dataset SOLVER=synthetic
```

### OpenFOAM backend

```bash
make dataset SOLVER=openfoam
```

OpenFOAM path uses:
- template case: `sim/templates/openfoam_case/`
- adapter: `sim/openfoam_adapter.py`
- geometry hook: `sim/templates/openfoam_case/scripts/build_geometry.py`

If OpenFOAM commands are not in `PATH`, dataset generation fails per-case and logs errors while pipeline remains resumable.

## 7. Project Structure

- `sim/`: case generation, solver adapters, OpenFOAM orchestration, manifests/index
- `extract/`: feature extraction from probe time series
- `ml/`: training, evaluation, report plots, summary writing
- `configs/`: experiment YAMLs
- `data/`: raw data and features
- `reports/`: plots and markdown summary

## 8. Reproducibility and Failure Handling

- fixed random seed in config
- case-level retries in dataset generation
- failures do not stop whole run
- failed cases + errors are captured in:
  - `data/raw/manifest.csv`
  - `reports/summary.md`
- logs:
  - `logs/dataset.log`
  - `logs/features.log`
  - `logs/train.log`

## 9. Extending the Pipeline

- Add new shape:
  - synthetic: add `simulation.synthetic.shape_params` and spatial mode in `sim/synthetic_solver.py`
  - OpenFOAM: extend geometry hook/meshing templates
- Change probe layout: modify `simulation.probes_n`
- Add new features: extend `extract/feature_engineering.py`
- Try new models: extend `ml/train.py`

## 10. Borrowed Architecture Notes

This repository intentionally borrows **architecture/process ideas** from prior open repositories while re-implementing all code in this repo.
See:
- `docs/ARCHITECTURE_REFERENCES.md`
