# PIV 原始 CSV 数据包：菱形 / diamond

内容：该障碍物的全部原始 PIV CSV。

目录结构：
- 3 个速度档：5 / 10 / 15 转速
- 3 条独立序列：第1组 / 第2组 / 第3组
- 每个目录约 299 个 CSV

CSV 主要字段：
- X(mm), Y(mm)
- Velocity |V|(mm/s)
- Velocity U(mm/s)
- Velocity V(mm/s)
- Correlation Value
- Flag
- Rotation Tensor(rad)
- Peak Ratio

推荐用途：
- 画 PIV 矢量图：使用 X/Y + Velocity U/V。
- 画速度云图：使用 Velocity |V|。
- 做数据检查：参考 manifest CSV。

注意：这是原始 PIV 导出数据，未做模型归一化、未裁剪为 wake-field tensor。
