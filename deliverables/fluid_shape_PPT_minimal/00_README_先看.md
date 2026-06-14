# Fluid Shape Pipeline PPT Minimal Package

用途：给课程展示/PPT 同学快速取图、取表、取结论。这个包不包含原始 PIV CSV、不包含 OpenFOAM case、不包含大模型 checkpoint。

## 一句话结论

我们用二维水槽尾流速度场识别上游试样形状。真实 PIV 数据按独立实验序列切分：seq1/seq2 训练，seq3 测试。PIV-JEPA 在真实 PIV 上的 sequence-level accuracy 为 86.7%，明显优于 CFD 模型直接迁移到真实 PIV 的约 22.2%，说明 CFD 与真实实验存在 domain gap，但真实 PIV 尾流中包含可学习的形状信息。

## 推荐 PPT 叙事顺序

1. 问题：给定下游尾流速度场，反推上游障碍物形状。
2. PIV 装置：水槽、试样、示踪粒子、激光片光、相机 ROI、速度场导出。
3. 尾流观察：速度亏损、剪切层、涡量、不同 Re 下尾流发展变化。
4. 实验：5 类试件、3 个速度档、3 条独立序列。
5. 物理参数：速度档按线性流量近似映射为 Re≈880/1760/2640。
6. 方法：CFD/synthetic 先验 + wake-field tensor + JEPA classifier。
7. 结果：CFD→PIV zero-shot 差，PIV-only 训练明显提升。
8. 结论：真实 PIV 可识别形状；下一步应补跑更贴近 Re≈1800/2600 的 CFD，并做更稳健的跨域适配。

## 关键数字

- CFD-JEPA 直接验证真实 PIV：sequence accuracy ≈ 22.2%。
- PIV-only Random Forest 诊断：accuracy ≈ 76.2%。
- PIV-JEPA 从头训练：test frame accuracy 87.1%，test sequence accuracy 86.7%，macro-F1≈0.870。
- CFD-JEPA 初始化后 PIV fine-tune：test sequence accuracy 80.0%（当前最好一轮）。
- CFD-JEPA 初始化后 shape-only fine-tune：test sequence accuracy 73.3%，说明弱 Re 监督有帮助。

## 文件夹说明

- `01_docs/`：中文说明、实验记录、PPT 提纲、PIV 审计报告、尾流分析、PIV 搭建流程。
- `02_figures_for_ppt/`：直接可放 PPT 的图。
- `03_results_csv/`：核心指标总表、混淆矩阵、sequence prediction、线性 Re 数据索引。
- `04_stl_specimens/`：五个等面积试样 STL。
- `05_code_snapshot/`：PIV 处理/训练脚本和关键 config 快照。

## 注意

PPT 汇报时不要把全量训练 100% accuracy 当泛化指标。可信泛化结果是 seq1/seq2 训练、seq3 测试的 86.7%。
