# PIV 原始数据按障碍物打包说明

本目录用于给同学分发 PIV 初始 CSV 数据。

## 文件

- `piv_sample_csv_for_vector_ai.zip`：每种障碍物 2 个 CSV 样例，适合快速试画矢量图/云图。
- `piv_raw_circle_圆.zip`：圆形障碍物全部原始 CSV。
- `piv_raw_triangle_三角.zip`：三角形障碍物全部原始 CSV。
- `piv_raw_airfoil_机翼.zip`：机翼障碍物全部原始 CSV。
- `piv_raw_diamond_菱形.zip`：菱形障碍物全部原始 CSV。
- `piv_raw_bar_长方.zip`：长方形障碍物全部原始 CSV。
- `piv_raw_zip_summary.csv`：每个 zip 的目录数、CSV 数、原始体积和压缩体积。

## 推荐给画图同学的最小输入

画矢量图：

```text
X(mm), Y(mm), Velocity U(mm/s), Velocity V(mm/s)
```

画速度云图：

```text
X(mm), Y(mm), Velocity |V|(mm/s)
```

质量控制可参考：

```text
Correlation Value, Peak Ratio, Flag
```

## 注意

这些是 PIV 软件原始导出的 CSV，不是机器学习训练时的归一化 wake-field tensor。
