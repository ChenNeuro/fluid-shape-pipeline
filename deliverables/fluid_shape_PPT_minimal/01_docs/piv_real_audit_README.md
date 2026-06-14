# Real PIV Validation Audit

Updated: 2026-06-07

## Source Data

Source directory:

```text
蓝月传奇/计算结果
```

Parsed structure:

```text
5 shapes x 3 speed levels x 3 sequences = 45 independent sequences
299 CSV velocity fields per sequence
13,455 CSV velocity fields total
```

Shape mapping:

| Source name | Model label |
|---|---|
| 圆 | circle |
| 三角 | triangle |
| 机翼 | airfoil |
| 菱形 | diamond |
| 长方 | bar |

Experiment notes from the field:

```text
water height = 35 cm
test-section/camera height = 35 cm / 2 = 17.5 cm
speed levels = 5 / 10 / 15
flow meter broken, so flow speed must be inferred from PIV boundary velocity
airfoil mount point was around half-chord, not the geometric centroid
```

## CSV Schema

Each CSV contains:

```text
X(mm), Y(mm)
Velocity |V|(mm/s)
Velocity U(mm/s)
Velocity V(mm/s)
Correlation Value
Flag
Rotation Tensor(rad)
Peak Ratio
```

Observed native grid:

```text
57 x 31 vectors = 1767 vectors per CSV
local exported ROI x = 4.15383 - 244.038 mm
local exported ROI y = 7.39902 - 135.908 mm
```

This is smaller than the originally recommended camera ROI, so the real PIV validation is not geometrically identical to the CFD `distD1/2/4` crop setup.

## Inferred Flow Speed

Free-stream speed was estimated from the top/bottom boundary strips of each PIV field.

For the stride-5 validation build:

| Speed level | Selected CSVs | Median boundary U | Median inferred Re |
|---:|---:|---:|---:|
| 5 | 900 | 10.90 mm/s | 869.5 |
| 10 | 900 | 10.34 mm/s | 825.4 |
| 15 | 900 | 10.67 mm/s | 851.4 |

Conclusion: the three speed levels are not reliably separated in the exported PIV velocity fields. For the first validation pass, all real PIV samples were assigned to the nearest CFD training Re class:

```text
Re = 800
```

The original speed level is still preserved as metadata.

### Alternative pump-setting estimate

The field team later noted that the pump/flow-meter relation appears approximately linear:

```text
45 speed level ~= 50 m^3/h
```

If this relation is used, then:

```text
Q(m^3/h) = speed_level * 50 / 45
```

With water height `h = 0.35 m` and tank width `W = 0.40 m`:

| Speed level | Q m^3/h | U mm/s | Re | Nearest trained CFD Re |
|---:|---:|---:|---:|---:|
| 5 | 5.56 | 11.02 | 880 | 800 |
| 10 | 11.11 | 22.05 | 1759 | 1500 |
| 15 | 16.67 | 33.07 | 2639 | 1500, but out of range |

This is physically more separated than the PIV boundary-strip estimate. However, it also means speed levels 10 and 15 are partly outside the CFD training range, because the current CFD model was trained only up to `Re=1500`.

Important implication:

```text
Changing the Re labels does not change zero-shot shape inference, because Re is not a model input.
It only changes the metadata/Re-head target and the physical interpretation.
```

For future fine-tuning with the current trained classes, the least-wrong mapping is:

```text
5 -> Re 800
10 -> Re 1500
15 -> Re 1500 / out-of-distribution high-speed validation
```

For a paper-level treatment, the better solution is to run additional CFD at approximately `Re=1800` and `Re=2600`, or to treat speed level as an experimental domain variable rather than forcing it into the old CFD Re classes.

## Built Validation Set

Primary validation run:

```text
/home/chenyihao/fluid_runs/piv_blueluna_validation_stride5
```

Build command:

```bash
python3 -m scripts.piv.build_real_wake_fields \
  --source-root 蓝月传奇/计算结果 \
  --output-run-dir /home/chenyihao/fluid_runs/piv_blueluna_validation_stride5 \
  --stride 5 \
  --assigned-re 800 \
  --crop-mode model_fractions
```

Output:

```text
2700 wake_field.npz samples
45 independent sequences
60 wake fields per sequence
```

Main index:

```text
/home/chenyihao/fluid_runs/piv_blueluna_validation_stride5/data/wake_fields/index.csv
```

## CFD Model External Validation

Model:

```text
/home/chenyihao/fluid_runs/cfd_final_stable175_tau6_all_jepa_gn/models/wake_field_main_cfd_finetuned.pt
```

Evaluation command:

```bash
python3 -m scripts.piv.evaluate_real_piv \
  --piv-run-dir /home/chenyihao/fluid_runs/piv_blueluna_validation_stride5 \
  --output-dir /home/chenyihao/fluid_runs/piv_blueluna_validation_stride5/reports \
  --batch-size 128
```

Results:

| Level | N | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| frame | 2700 | 0.225 | 0.160 |
| sequence mean probability | 45 | 0.222 | 0.159 |
| sequence majority vote | 45 | 0.244 | 0.172 |

The model mostly predicts only `circle` and `diamond`, so direct CFD-to-PIV transfer currently fails.

Control runs:

| Build | Sequence accuracy | Macro-F1 | Note |
|---|---:|---:|---|
| model_fractions | 0.222 | 0.159 | best of tested direct-transfer variants |
| roi_fractions | 0.156 | 0.113 | worse |
| model_fractions + flip_y | 0.222 | 0.163 | no meaningful improvement |

## PIV-Only Diagnostic

To check whether the real PIV data itself contains shape information, a lightweight Random Forest diagnostic was run on crop statistics:

```text
train = sequences 1 and 2
test = sequence 3
split is by independent sequence, not by random frames
```

Result:

| Split | Train N | Test N | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| seq1/2 -> seq3 | 1800 | 900 | 0.762 | 0.746 |

Conclusion: the PIV data is not useless; it contains usable shape signal. The failure is mainly a CFD-to-real domain gap and crop/ROI mismatch problem.

## PIV JEPA Training

A JEPA wake classifier was then trained directly on real PIV tensors using the same wake-field tensor interface as the CFD/synthetic pipeline:

```text
train = sequences 1 and 2
test = sequence 3
split is by independent experimental sequence, not by random frames
variant = distD_multi_4ch
loss = shape classification only; Re and geometry heads disabled for this real-data pass
```

Training command:

```bash
python3 -m scripts.piv.train_real_piv_jepa \
  --piv-run-dir /home/chenyihao/fluid_runs/piv_blueluna_validation_stride5 \
  --output-run-dir /home/chenyihao/fluid_runs/piv_jepa_seq12_test3_stride5 \
  --train-sequences 1,2 \
  --test-sequences 3 \
  --batch-size 32 \
  --pretrain-epochs 25 \
  --epochs 50 \
  --lr 5e-4 \
  --pretrain-lr 1e-3 \
  --encoder-norm group \
  --re-weight 0.0 \
  --params-weight 0.0 \
  --noise-std 0.015
```

Results:

| Split | Level | N | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|
| train | frame | 1800 | 1.000 | 1.000 |
| train | sequence | 30 | 1.000 | 1.000 |
| test | frame | 900 | 0.871 | 0.870 |
| test | sequence | 15 | 0.867 | 0.870 |

Test sequence confusion matrix:

| True / Pred | airfoil | bar | circle | diamond | triangle |
|---|---:|---:|---:|---:|---:|
| airfoil | 3 | 0 | 0 | 0 | 0 |
| bar | 0 | 3 | 0 | 0 | 0 |
| circle | 0 | 0 | 2 | 0 | 1 |
| diamond | 0 | 0 | 0 | 2 | 1 |
| triangle | 0 | 0 | 0 | 0 | 3 |

By speed level:

| Speed level | Test sequences | Sequence accuracy |
|---:|---:|---:|
| 5 | 5 | 1.000 |
| 10 | 5 | 1.000 |
| 15 | 5 | 0.600 |

Interpretation:

```text
Real PIV -> JEPA works much better than direct CFD -> PIV zero-shot transfer.
The two sequence-level errors both occur at speed level 15.
This suggests the high-speed setting is either physically outside the current CFD range or has stronger measurement/domain shift.
```

Model:

```text
/home/chenyihao/fluid_runs/piv_jepa_seq12_test3_stride5/models/wake_field_main_piv_jepa.pt
```

Reports:

```text
/home/chenyihao/fluid_runs/piv_jepa_seq12_test3_stride5/reports
```

Final all-PIV training run:

```bash
python3 -m scripts.piv.train_real_piv_jepa \
  --piv-run-dir /home/chenyihao/fluid_runs/piv_blueluna_validation_stride5 \
  --output-run-dir /home/chenyihao/fluid_runs/piv_jepa_all_stride5 \
  --train-all \
  --batch-size 32 \
  --pretrain-epochs 25 \
  --epochs 50 \
  --lr 5e-4 \
  --pretrain-lr 1e-3 \
  --encoder-norm group \
  --re-weight 0.0 \
  --params-weight 0.0 \
  --noise-std 0.015
```

Final all-PIV model:

```text
/home/chenyihao/fluid_runs/piv_jepa_all_stride5/models/wake_field_main_piv_jepa.pt
```

The all-PIV run uses all 45 real sequences for deployment-style prediction and should not be reported as an independent generalization test. It reached 100% training frame/sequence accuracy, while the honest sequence-holdout estimate remains the `seq1/2 -> seq3` result above.

## Figures

Generated QC figures:

```text
reports/piv_real_audit/real_piv_input_examples.png
reports/piv_real_audit/real_piv_sequence_confusion_matrix.png
reports/piv_real_audit/real_piv_speed_level_estimated_re.png
reports/piv_real_audit/piv_jepa_sequence_confusion_matrix.png
reports/piv_real_audit/piv_jepa_training_history.png
```

## Recommendation

For the presentation:

```text
Use CFD-only model result as simulation benchmark.
Use real PIV as external validation.
Report that direct CFD-to-real zero-shot transfer is currently poor.
Then show PIV-only diagnostic accuracy around 76%, proving real data contains shape information.
Then show real PIV JEPA sequence-holdout accuracy around 86.7%.
For deployment/testing tomorrow, train a final PIV JEPA on all real sequences after preserving the seq1/2 -> seq3 result as the honest holdout estimate.
For CFD reruns, prioritize Re around the experimental speed levels: approximately 800, 1800, and 2600.
```

Do not claim the CFD-trained model already generalizes to real PIV. The evidence does not support that.
