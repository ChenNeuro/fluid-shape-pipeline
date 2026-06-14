# PIV CSV 样例：给 AI 画矢量图用

用途：每种障碍物提供 2 个原始 PIV CSV，方便同学尝试让 AI 或绘图脚本仿照千眼狼系统的矢量图/云图效果。

选择规则：
- 五种障碍物：circle / triangle / airfoil / diamond / bar。
- 每类选 `10转速-第1组`。
- 每类 2 帧：第 1 帧和中间帧。

CSV 字段重点：
- `X(mm)`, `Y(mm)`：PIV 网格坐标。
- `Velocity U(mm/s)`：水平方向速度。
- `Velocity V(mm/s)`：竖直方向速度。
- `Velocity |V|(mm/s)`：速度大小。
- `Correlation Value`, `Peak Ratio`, `Flag`：质量控制字段。

画矢量图时最直接的输入是：

```text
X(mm), Y(mm), Velocity U(mm/s), Velocity V(mm/s)
```

如果画云图，可以用：

```text
Velocity |V|(mm/s)
```

注意：这些是原始 PIV 导出的局部 ROI 数据，不是完整水槽尺寸。
