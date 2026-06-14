# 周一汇报 PPT 页纲草稿

## 1. 标题页

题目：基于 CFD 稳定尾流与 JEPA 表征学习的二维试件形状识别

副标题：从试件设计、OpenFOAM 数据生成到 PIV 外部验证

## 2. 问题定义

要回答的问题：

给定下游 PIV/CFD 尾流速度场，能否识别上游试件形状？

当前五类形状：

circle、triangle、airfoil、diamond、bar

## 3. 实验几何与二维近似

水槽尺寸：

| 方向 | 尺寸 |
|---|---:|
| 长度 x | 0.85 m |
| 水深 y | 0.45 m |
| 宽度 z | 0.40 m |

建模假设：

二维计算域取 x-y 平面，忽略 z 向宽度；真实实验用 PIV 做外部验证三维端部效应。

## 4. 试件设计

统一二维面积：

```text
A = 5000 mm^2
beta_area = A/H^2 = 0.02469
D_eq = 79.8 mm
```

说明：beta_area 是无量纲面积比，不带单位。

名义尺寸表：

| 形状 | 尺寸 |
|---|---|
| Circle | D=79.8 mm |
| Triangle | a=107.5 mm, h=93.1 mm |
| Airfoil | c=228.3 mm, t=32.0 mm |
| Diamond | w=121.4 mm, h=82.4 mm |
| Bar | w=144.3 mm, h=34.6 mm |

## 5. CAD/STL 单位检查

需要强调：

STL 坐标按 mm 读取，代码转换为 m。

如果 SolidWorks/STL 被当成 inch：

```text
长度误差 25.4 倍
面积误差 25.4^2 倍
```

这是之前怀疑“面积过大”的关键风险源。

## 6. CFD 流程

流程图：

```text
STL -> x-y 投影轮廓 -> Gmsh 2D 网格 -> OpenFOAM pimpleFoam -> U 场 -> wake_field.npz
```

边界条件：

inlet uniform U；walls/obstacle noSlip；outlet p=0；frontAndBack empty。

## 7. 为什么要跑稳定尾流

早期 1 秒 CFD 不足以形成成熟尾流。

定义：

```text
tau = t U / D_eq
```

CFD 跑到 tau=10，只取 tau>=6。

一句话：

不是按固定秒数取数据，而是按“扰动已经对流了多少个试件直径”取数据。

## 8. 数据集

参数矩阵：

```text
5 shapes x 5 Re x 7 dy = 175 CFD cases
Re = [300, 500, 800, 1100, 1500]
dy = [-30, -20, -10, 0, 10, 20, 30] mm
```

稳定尾流样本：

```text
tau>=6: 735 wake-field samples
strict audit: train=630, test=105
```

## 9. Wake-field 表示

输入通道：

```text
ux, uy, speed, vorticity
```

多尺度 crop：

```text
distD1.0_full
distD2.0_full
distD4.0_full
```

使用 D_eq 定义距离，而不是 H。

## 10. 模型对比

| 模型 | strict test accuracy | strict test macro-F1 |
|---|---:|---:|
| ResNet18 | 0.448 | 0.339 |
| small CNN | 0.438 | 0.371 |
| JEPA + GroupNorm | 0.981 | 0.981 |

核心结论：

JEPA 自监督预训练显著提升 CFD 小数据场景下的尾流表征。

## 11. 最终模型

最终模型使用全部稳定 CFD 样本训练：

```text
cfd_final_stable175_tau6_all_jepa_gn
model_type = jepa
encoder_norm = group
train_samples = 735
```

外部验证交给真实 PIV。

## 12. PIV 验证计划

PIV 首先作为外部验证集，不参与模型选择。

若每类形状有足够独立序列，再做 secondary experiment：

```text
CFD pretrain -> small PIV fine-tune -> independent PIV test
```

必须按独立视频/实验段划分，不能随机打散帧。

## 13. 风险与下一步

风险：

1. CFD-real gap。
2. 二维近似与端部三维效应。
3. PIV 噪声、反光、粒子密度。
4. CAD/STL 单位误差。

下一步：

1. 明天采集 PIV。
2. 统一 PIV importer 到 wake_field.npz schema。
3. 用 final JEPA 模型做外部验证。
4. 必要时做少量真实数据微调。

## 14. 备选一句话总结

我们用真实试件 STL 构建了 OpenFOAM 稳定尾流数据集，并按对流时间 tau 筛掉启动瞬态；在严格中心摆放 holdout 上，JEPA + GroupNorm 达到约 98% 形状识别准确率，下一步用 PIV 检验 CFD-to-real 泛化能力。

## 15. 可直接使用的 CFD 图片

建议放法：

| 图片 | 放在哪页 |
|---|---|
| `docs/figures/cfd_experiment/01_equal_area_stl_outlines.png` | 第 4 页，试件设计 |
| `docs/figures/cfd_experiment/02_tank_roi_alignment.png` | 第 3 页或第 12 页，实验几何/PIV 对齐 |
| `docs/figures/cfd_experiment/03_shape_wakes_re800_dy0_latest_tau.png` | 第 8 页或单独一页，五类尾流对比 |
| `docs/figures/cfd_experiment/04_circle_re_sweep_latest_tau.png` | 第 7 页，流速/Re 选择 |
| `docs/figures/cfd_experiment/05_circle_crop_alignment_re800.png` | 第 9 页，wake-field crop 定义 |
| `docs/figures/cfd_experiment/06_circle_time_evolution_re800.png` | 第 7 页，稳定尾流筛选依据 |
