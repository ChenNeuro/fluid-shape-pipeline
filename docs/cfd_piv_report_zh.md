# 基于稳定尾流 CFD 与 JEPA 表征学习的二维试件形状识别报告草稿

更新时间：2026-06-06

## 摘要

本项目研究的问题是：在二维水槽流动中，是否可以仅根据下游尾流速度场反推出上游试件形状。为尽量贴近即将开展的 PIV 实验，我们将水槽简化为长度-水深平面的二维通道流，忽略水槽宽度方向，并使用实际 SolidWorks/STL 试件轮廓构建 OpenFOAM CFD 数据集。试件采用等面积设计，名义面积为 5000 mm^2，对应等效圆直径 79.8 mm。CFD 数据按无量纲对流时间 tau=tU/D_eq 跑到 tau=10，并只取 tau>=6 的成熟尾流窗口作为训练数据。模型方面，ResNet 与普通小 CNN 在严格中心位置留出测试上表现不稳定，而 JEPA 自监督预训练加 GroupNorm 编码器在相同数据上取得了明显优势：在 dy != 0 训练、dy = 0 中心摆放测试的内部审计中，形状分类准确率为 98.1%，macro-F1 为 98.1%。最终交付模型使用全部稳定 CFD 尾流样本训练，后续由真实 PIV 数据进行外部验证。

## 1. 实验目标与问题定义

目标不是做一个纯合成数据分类器，而是建立一条可以服务真实 PIV 实验的流程：

1. 根据水槽尺寸与可加工约束设计五类等面积试件。
2. 使用真实 STL 轮廓生成二维通道 CFD 数据。
3. 从 CFD 尾流中提取多下游距离速度场图像。
4. 训练形状识别模型。
5. 用明天采集的 PIV 数据做外部验证，必要时再做少量真实数据微调。

坐标定义如下：

| 方向 | 含义 | 当前取值 |
|---|---|---|
| x | 来流方向/水槽长度方向 | 0 到 0.85 m |
| y | 水深方向/二维通道高度方向 | 0 到 0.45 m |
| z | 水槽宽度方向，当前二维模型忽略 | 0.40 m |

因此，当前 CFD 模型中的 dy 不是来流方向位置，而是试件在水深方向相对中线的偏移。dy=0 表示试件位于水深中线 y=0.225 m。

## 2. 试件设计方法

### 2.1 等面积设计

五类试件包括 circle、triangle、airfoil、diamond、bar。为了使不同形状的尾流差异主要来自形状本身，而不是迎风面积或尺度差异，本项目采用等二维面积设计。目标面积为：

```text
A = 0.005 m^2 = 5000 mm^2
```

水深方向通道高度：

```text
H = 0.45 m
```

面积系数：

```text
beta_area = A / H^2 = 0.005 / 0.45^2 = 0.02469
```

这里的 beta_area 是无量纲面积比，不带单位。它不是物理长度，也不是速度参数。等效圆直径定义为：

```text
D_eq = sqrt(4A/pi) = 0.0798 m = 79.8 mm
```

后续 Reynolds 数统一使用 D_eq 定义：

```text
Re = U D_eq / nu
```

其中水的运动黏度取 nu=1e-6 m^2/s。

### 2.2 名义加工尺寸

由 `manufacturing/design_obstacles.py` 计算得到的名义尺寸如下。内部计算使用 SI 单位，输出加工尺寸使用 mm。

| 形状 | 关键尺寸 1 | 关键尺寸 2 | 面积 |
|---|---:|---:|---:|
| Circle | D = 79.8 mm | - | 5000 mm^2 |
| Triangle | 边长 a = 107.5 mm | 高 h = 93.1 mm | 5000 mm^2 |
| Airfoil | 弦长 c = 228.3 mm | 最大厚度 t = 32.0 mm | 5000 mm^2 |
| Diamond | 横向对角线 w = 121.4 mm | 纵向对角线 h = 82.4 mm | 5000 mm^2 |
| Bar | 长 w = 144.3 mm | 高 h = 34.6 mm | 5000 mm^2 |

### 2.3 二维近似与有限展长风险

当前实验水槽宽度为 0.40 m。若试件在宽度方向近似贯穿流场，则可以将中间区域近似看作二维流动。对于不同形状，宽度与特征尺寸比值如下：

| 形状 | streamwise/H | crossflow/H | span/streamwise | span/crossflow |
|---|---:|---:|---:|---:|
| circle | 17.7% | 17.7% | 5.01 | 5.01 |
| triangle | 20.7% | 23.9% | 4.30 | 3.72 |
| airfoil | 50.7% | 7.1% | 1.75 | 12.51 |
| diamond | 27.0% | 18.3% | 3.29 | 4.86 |
| bar | 32.1% | 7.7% | 2.77 | 11.55 |

需要注意：airfoil 的弦长较长，span/streamwise 只有约 1.75，因此如果试件不是横跨水槽宽度，或者边界端部效应明显，airfoil 会最容易偏离理想二维假设。报告中建议将这一点作为实验误差来源，而不是回避。

### 2.4 CAD/STL 单位约束

本项目 CFD 构建脚本默认 STL 顶点坐标单位为 mm，读取后乘以 0.001 转换为 m。SolidWorks 导出或导入时必须确认单位是 mm。如果 CAD 或切片软件把 STL 当作 inch 处理，长度会差 25.4 倍，面积会差 25.4^2 倍，直接导致试件过大、尾流不再对应二维设计。当前实际输入目录为：

```text
HURRY/solidworks_model_STL/
```

文件命名为 `*_5000mm2.STL`，表示这些模型已经按 5000 mm^2 的等面积目标准备。

## 3. CFD 数据生成流程

### 3.1 几何与网格

CFD 管线直接从 STL 文件读取每类试件的 x-y 投影轮廓，并忽略 z 向展长。脚本使用 STL 投影点的凸包近似二维外轮廓，将试件放置在通道中：

```text
center_x = 0.25 m
center_y = 0.225 m + dy
```

二维通道尺寸：

```text
length L = 0.85 m
height H = 0.45 m
pseudo thickness = 0.01 m
```

网格由 Gmsh 生成，再通过 `gmshToFoam` 转为 OpenFOAM 网格。frontAndBack 边界设置为 empty，从而形成二维 OpenFOAM 求解域。

### 3.2 边界条件与求解器

当前求解设置：

| 项目 | 设置 |
|---|---|
| 求解器 | OpenFOAM `pimpleFoam` |
| 流体 | 水，nu = 1e-6 m^2/s |
| 流态 | laminar |
| inlet | uniform U |
| outlet | p=0, U zeroGradient |
| walls | noSlip |
| obstacle | noSlip |
| frontAndBack | empty |

使用的 Reynolds 数为：

```text
Re = [300, 500, 800, 1100, 1500]
```

对应入口速度：

| Re | U (m/s) | D_eq/U (s) | tau=10 物理时间 (s) |
|---:|---:|---:|---:|
| 300 | 0.00376 | 21.22 | 212.2 |
| 500 | 0.00627 | 12.73 | 127.3 |
| 800 | 0.01003 | 7.96 | 79.6 |
| 1100 | 0.01379 | 5.79 | 57.9 |
| 1500 | 0.01880 | 4.24 | 42.4 |

这个表解释了为什么早期 1 s CFD 虽然跑得很快，但物理上还不够成熟：对 Re=800，1 s 只对应 tau≈0.126；对 Re=300 更只有 tau≈0.047。

### 3.3 参数矩阵

稳定 CFD 数据集使用以下参数矩阵：

```text
5 shapes x 5 Re x 7 dy = 175 CFD cases
```

其中：

```text
dy = [-0.03, -0.02, -0.01, 0.0, 0.01, 0.02, 0.03] m
```

内部审计时，dy=0 作为中心摆放测试集，dy != 0 作为训练集。最终模型训练时，则使用全部稳定 CFD 样本，真实泛化能力交给 PIV 数据验证。

### 3.4 稳定尾流筛选

为了避免将启动瞬态误当作成熟尾流，本项目使用无量纲对流时间：

```text
tau = t U / D_eq
```

CFD 每个 case 跑到 tau=10。训练数据只取：

```text
tau >= 6
```

选择 tau>=6 的理由是：当前 wake crop 包含 1D、2D、4D 三个下游距离。若扰动至少对流 6D，4D 位置处已有足够安全余量形成下游尾流结构，而不会主要反映初始条件传播过程。

实际得到的数据为：

| 数据集 | 样本数 | 说明 |
|---|---:|---|
| `cfd_tank_stable175` | 175 CFD cases | 全部 case 成功 |
| `cfd_tank_stable175_tau6` | 735 wake-field samples | tau 从 6 到 10 |
| strict audit split | train=630, test=105 | dy!=0 train, dy=0 test |

## 4. Wake-field 表示

每个 CFD 末/中间时刻速度场被插值到规则网格，并构造 4 个通道：

```text
ux, uy, speed, vorticity
```

每个样本保存为 `wake_field.npz`，包含：

```text
field_raw
field_norm
crops
scales
channel_names
crop_boxes
```

多尺度 crop 使用下游距离按 D_eq 定义，而不是按通道高度 H 定义：

```text
distD1.0_full
distD2.0_full
distD4.0_full
```

这样做的原因是本实验的主要尺度是试件等效直径。若用 H 定义下游距离，0.5H 或 1H 对本试件来说已经是多个 D_eq，容易把近尾流细节跳过去。

## 5. 模型与训练策略

### 5.1 为什么最终选择 JEPA

我们比较了三类模型：

1. ResNet18 监督训练。
2. small CNN 监督训练。
3. JEPA 自监督预训练 + GroupNorm CNN 编码器 + 监督微调。

ResNet 与 small CNN 在小规模 CFD 数据上受 BatchNorm 或样本分布影响较大；训练 loss 可以很低，但 eval 指标不稳定。JEPA 的优势在于先从 CFD wake crop 本身学习空间表征，再使用标签微调，这与 PIV 迁移场景更匹配。

### 5.2 JEPA 设置

最终审计模型设置：

| 项目 | 设置 |
|---|---|
| backbone | JEPA |
| encoder | lightweight CNN |
| norm | GroupNorm |
| feature_dim | 192 |
| fusion_hidden | 192 |
| dropout | 0.1 |
| pretrain_epochs | 60 |
| finetune_epochs | 80 |
| mask_ratio | 0.3 |
| batch_size | 32 |
| shape loss weight | 1.0 |
| dy/eps loss weight | 0.03 |
| Re loss weight | 0.0 |

Re loss 被关闭，是因为 wake-field 构建时做了 per-case 通道归一化，绝对速度幅值被削弱。当前模型目标应优先是形状分类，而不是从归一化图像中强行恢复 Re。

## 6. 当前结果

### 6.1 内部审计结果

严格审计使用 dy!=0 训练、dy=0 中心摆放测试。这个 split 对实际实验很苛刻，因为模型完全没有见过中心摆放样本，但可以检验模型是否学到了形状尾流结构，而不是只记住位置偏移。

| 模型 | train accuracy | test accuracy | test macro-F1 |
|---|---:|---:|---:|
| ResNet18 | 0.490 | 0.448 | 0.339 |
| small CNN | 0.724 | 0.438 | 0.371 |
| JEPA + GroupNorm | 1.000 | 0.981 | 0.981 |

JEPA + GroupNorm 在严格中心位置 holdout 上显著优于其他模型，因此当前主线应采用 JEPA，而不是继续调 ResNet。

### 6.2 最终 CFD 训练模型

为了最大化利用 CFD 数据，并考虑明天 PIV 才是真正外部验证，最终模型使用全部 tau>=6 稳定 CFD 样本训练：

```text
/home/chenyihao/fluid_runs/cfd_final_stable175_tau6_all_jepa_gn/models/wake_field_main_cfd_finetuned.pt
```

该模型：

```text
model_type = jepa
variant = distD_multi_4ch
encoder_norm = group
train_samples = 735
```

该 final 模型没有内部 test 指标；它的用途是用于明天 PIV 外部验证。内部可信度参考 strict audit JEPA 结果。

## 7. PIV 验证与是否参与训练

建议报告中这样表述真实数据：

1. PIV 数据首先作为外部验证集，不参与模型选择。
2. 若每个形状至少有 3 段独立 PIV 序列，可以做一个 secondary experiment：CFD 预训练 + 少量 PIV 微调。
3. PIV 微调时必须按独立采集序列划分 train/test，不能把同一段视频的相邻帧同时放入 train 和 test。
4. 主结果仍以 CFD-only 模型在 PIV 上的零样本或少样本验证为准，避免真实数据太少导致过拟合。

可以在 PPT 中保守地说：当前阶段已经得到 CFD 内部审计可信的模型；真实 PIV 的角色是验证 CFD-to-real gap。如果 PIV 与 CFD 差异较大，再讨论用少量真实数据做 domain adaptation。

## 8. 目前的主要风险

1. 二维近似风险：实际水槽宽度有限，尤其 airfoil 的 streamwise 尺寸较长，端部三维效应可能更明显。
2. STL/CAD 单位风险：必须确认所有 STL 坐标单位为 mm。
3. CFD-real gap：OpenFOAM 使用理想边界和 laminar 设置，真实 PIV 可能包含自由液面、壁面粗糙、3D 端部效应、粒子噪声和照明误差。
4. Re 识别暂不作为主任务：当前图像归一化削弱绝对速度，Re head 不应作为报告重点。
5. PIV 数据泄漏风险：真实视频帧不能随机打散，必须按独立实验段落划分。

## 9. 可复现实验命令

### 9.1 生成稳定 CFD cases

```bash
cd /mnt/c/Users/chenyihao/Documents/GitHub/fluid-shape-pipeline

python3 -m scripts.cfd.build_openfoam_cases \
  --config configs/cfd_tank_stable175.yaml \
  --stl-dir HURRY/solidworks_model_STL \
  --output-root /home/chenyihao/fluid_runs/cfd_tank_stable175 \
  --openfoam-bashrc /usr/share/openfoam/etc/bashrc
```

### 9.2 运行 OpenFOAM

```bash
python3 -m scripts.cfd.run_openfoam_cases \
  --case-root /home/chenyihao/fluid_runs/cfd_tank_stable175/openfoam_cases \
  --workers 12 \
  --openfoam-bashrc /usr/share/openfoam/etc/bashrc
```

### 9.3 提取稳定尾流样本

```bash
python3 -m scripts.cfd.build_wake_fields_from_openfoam \
  --config configs/cfd_tank_stable175.yaml \
  --case-root /home/chenyihao/fluid_runs/cfd_tank_stable175/openfoam_cases \
  --run-dir /home/chenyihao/fluid_runs/cfd_tank_stable175_tau6 \
  --openfoam-bashrc /usr/share/openfoam/etc/bashrc \
  --time-mode all \
  --tau-min 6 \
  --workers 8
```

### 9.4 JEPA strict audit

```bash
python3 -m ml.finetune_wake_cfd \
  --cfd-run-dir /home/chenyihao/fluid_runs/cfd_tank_stable175_tau6 \
  --output-run-dir /home/chenyihao/fluid_runs/cfd_audit_stable175_tau6_jepa_gn \
  --backbone jepa \
  --pretrain-epochs 60 \
  --epochs 80 \
  --batch-size 32 \
  --lr 0.0005 \
  --pretrain-lr 0.001 \
  --mask-ratio 0.3 \
  --seed 42 \
  --fusion-hidden 192 \
  --dropout 0.1 \
  --encoder-norm group \
  --params-weight 0.03 \
  --re-weight 0.0 \
  --noise-std 0.005
```

### 9.5 JEPA final all-train

```bash
python3 -m ml.finetune_wake_cfd \
  --cfd-run-dir /home/chenyihao/fluid_runs/cfd_tank_stable175_tau6 \
  --output-run-dir /home/chenyihao/fluid_runs/cfd_final_stable175_tau6_all_jepa_gn \
  --backbone jepa \
  --train-all \
  --pretrain-epochs 60 \
  --epochs 80 \
  --batch-size 32 \
  --lr 0.0005 \
  --pretrain-lr 0.001 \
  --mask-ratio 0.3 \
  --seed 42 \
  --fusion-hidden 192 \
  --dropout 0.1 \
  --encoder-norm group \
  --params-weight 0.03 \
  --re-weight 0.0 \
  --noise-std 0.005
```

## 10. 可以放进 PPT 的一句话结论

我们没有把早期瞬态 CFD 当作训练数据，而是按无量纲对流时间筛选成熟尾流；在 tau>=6 的稳定尾流上，JEPA 自监督表征学习显著优于普通监督 CNN，并在严格的中心摆放 holdout 上达到 98.1% 形状识别准确率。下一步将用真实 PIV 数据验证 CFD-to-real 泛化能力。

## 11. CFD 图件索引

以下图片已经由稳定 CFD 数据直接生成，可用于和明天的 PIV 图像做对照。

| 文件 | 建议用途 |
|---|---|
| `docs/figures/cfd_experiment/01_equal_area_stl_outlines.png` | 说明五类试件来自同面积 STL 投影，检查 CAD/STL 公制单位 |
| `docs/figures/cfd_experiment/02_tank_roi_alignment.png` | 说明水槽坐标、试件位置、PIV/CFD 尾流 ROI 和 1D/2D/4D 裁剪起点 |
| `docs/figures/cfd_experiment/03_shape_wakes_re800_dy0_latest_tau.png` | 展示五类试件在 Re=800、中心摆放、稳定尾流下的 speed/U 与无量纲涡量 |
| `docs/figures/cfd_experiment/04_circle_re_sweep_latest_tau.png` | 展示圆柱尾流随 Re=300 到 1500 的变化，辅助选择实验流速 |
| `docs/figures/cfd_experiment/05_circle_crop_alignment_re800.png` | 单独展示 PIV 图像应如何对齐 distD1.0、distD2.0、distD4.0 wake crops |
| `docs/figures/cfd_experiment/06_circle_time_evolution_re800.png` | 展示 tau=7 到 tau=10 的尾流发育过程，说明使用 tau>=6 的理由 |

复现命令：

```bash
python3 -m scripts.cfd.plot_cfd_experiment_figures \
  --index /home/chenyihao/fluid_runs/cfd_tank_stable175_tau6/data/wake_fields/index.csv \
  --config configs/cfd_tank_stable175.yaml \
  --stl-dir HURRY/solidworks_model_STL \
  --output-dir docs/figures/cfd_experiment
```
