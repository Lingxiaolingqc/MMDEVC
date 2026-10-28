# 单图评测协议 v1

主维护：A / Qinchuan XIAO；B / Jun Yu Eugene YIP 的真实预测交接与独立复核待完成。适用于 Spring 2027 Week 1 单图实验；视频、边界与正式多方法比较尚未冻结。

## 数据、范围与尺寸

- **Middlebury trainingQ**：15 对原始 Quarter Resolution 图像；Adirondack、Pipes 为 dev，其余 13 对为 evaluation。配置只在 dev 上选择。标定 `Z = f_px × baseline_mm / 1000 / (d_px + doffs_px)`，单位米。all-GT 与 nonoccluded-GT 区域由 GT 和官方 mask 固定，见 [双目基线](stereo_baseline.md) 与 [批量运行](stereo_suite.md)。
- **NYUv2 labeled**：保留官方 MAT 的 `depths`（填补后的深度），单位米；标准 795 train / 654 test，原始 1-based MATLAB index 对应 `nyuv2_0001` 等 ID。三个开发样本只从 train 侧按索引选取不同场景，不改动标准 split，见 `manifests/nyuv2_dev.csv`。
- NYUv2 的 HDF5 `images` 为 `(N,C,W,H)`，转为 `(H,W,C)`；`depths` 为 `(N,W,H)`，转为 `(H,W)`。原图 480×640，不旋转、不镜像、不除以 1000、不把 GT 量化为 PNG。
- NYUv2 固定评测区域：Python 半开区间 `y=45:471, x=41:601`，同时要求 GT 有限且在 **[0.001,10] m（含端点）**。模型可看完整 RGB，评测时才应用 crop。
- 评测入口要求预测与原始 GT 等尺寸；不自动 resize、填洞或裁剪预测深度。模型自己的 resize/postprocess 必须记录。UniDepthV2 使用官方 `infer` 返回原尺寸的 Z depth。

**参考实现边界：** 课程要求参考 [DA V2 metric-depth](https://github.com/DepthAnything/Depth-Anything-V2/tree/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/metric_depth)。该版本提供指标实现，但没有 NYUv2 专用 loader/split/crop。因此本项目采用其指标公式，crop 另外取自 [UniDepth NYUv2 eval_mask](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/datasets/nyuv2.py)；0.001 m 下限为显式项目设置，上限为 NYUv2 10 m。不能称为“DA V2 官方 NYUv2 crop”。标准 split 使用公开的 [splits.mat](https://github.com/ankurhanda/nyuv2-meta-data/blob/master/splits.mat)，来源与 SHA256 写入数据检查元数据。若教师另指定 crop/range，应统一改协议并重算。

## 无效预测与指标

`G` 是固定 GT 区域；`P` 是预测有效 mask（有限且正的米制深度，并满足模型自带的有效性条件）；`V=G∩P`。报告 coverage=`|V|/|G|` 和无效像素数。模型 mask 不得改变 GT 分母。

在 V 上计算以下条件误差，`p` 为预测、`g` 为 GT：

| 指标 | 定义 | 单位 |
| --- | --- | --- |
| AbsRel | mean(abs(p−g)/g) | 无量纲 |
| SqRel | mean((p−g)²/g) | m |
| RMSE | sqrt(mean((p−g)²)) | m |
| RMSE-log | sqrt(mean((ln(p)−ln(g))²)) | 自然对数 |
| δk，k=1,2,3 | mean(max(p/g,g/p) < 1.25^k) | 比例；严格小于 |

另外报告以整个 G 为分母、无效预测算失败的 `delta*_full_gt`。空 V 时条件误差为 null，coverage=0、full-GT δ=0；空 G 直接报错。Stereo 还报告全 G 上 Bad>1/2 px，将无效预测计为错误。深度 RMSE 不能脱离 coverage 单独比较。

批量 Stereo 深度指标采用逐场景宏平均；覆盖率和 Bad 的像素加权汇总明确以 `pooled_` 命名。dev、13 场景 evaluation、包含 dev 的 15 场景描述性汇总分别报告。未实现的 NYUv2 全测试集比较不冒充已完成。

## Direct metric 与 Relative + Alignment

| 输入声明 | 单位 / alignment | 处理与标签 |
| --- | --- | --- |
| `metric_depth` | `m` 或 `mm` / `none` | 只做 mm→m 必要转换；不用 GT 校正；`direct_metric` |
| `relative_depth` | `arbitrary` / `median` | 在 G∩有效正预测上拟合 s=median(g)/median(raw)，p=s×raw；`relative_aligned` |
| `inverse_depth_like` | `arbitrary` / `inverse_scale_shift` | 在 G∩有效预测上最小二乘拟合 a×raw+b≈1/g，再取倒数；a 必须正；非正逆深度输出为无效；`relative_aligned` |

对齐只用于有 GT 的逐图 benchmark 分析，会使用该图 GT；不能拿来证明直接米制能力，也不能直接用于没有 GT 的视频。拟合一次后复用相同转换，不为不同遮挡区域分别拟合。退化、常数或方向错误的逆深度拟合报错。B 需确认模型实际输出方向，并保留未经归一化的 float `.npy` 和模型配置。

## B 可直接调用的入口

在仓库根目录、安装 `.[data,stereo,nyu]` 的环境中执行，输出目录必须是新的：

```sh
# Direct metric，例如 UniDepthV2 的原始相机 Z 深度
python scripts/evaluate_depth.py --dataset nyuv2 --sample nyuv2_0003 --prediction outputs/unidepth_week1/predictions/nyuv2_0003.npy --prediction-type metric_depth --unit m --alignment none --output-dir outputs/eval_metric

# B 的 inverse-depth-like 原始输出；需事先确认表示与同一原图尺寸
python scripts/evaluate_depth.py --dataset nyuv2 --sample nyuv2_0003 --prediction outputs/dav2/nyuv2_0003.npy --prediction-type inverse_depth_like --unit arbitrary --alignment inverse_scale_shift --output-dir outputs/eval_dav2
```

可选 `--mask path.npy` 必须为同尺寸 bool 数组；预测本身的 NaN/Inf/非正 metric depth 仍视为无效。输入 float32/float64、二维 H×W，不接收整数图或伪彩色图。入口保存转换后的副本、有效 mask、指标和变换参数，原始输入不改动。

已检查：GT=prediction、mm→m、自然对数与 strict δ、空预测、NaN/Inf、单位/尺寸/方向错误、两种显式对齐；七指标与上述 DA V2 实现的随机正值对照一致。保存的 Stereo 预测已通过该入口复算。B 的真实预测与双方独立复现仍待交接。

## 后续边界

视频的 scene split、光流公共 mask、边界指标和时序公式留到后续任务。无 GT 视频不报告绝对深度准确率；相机移动引起的真实 Z 变化不能全部称为漂移。光流只建立像素对应，不把历史 Z 转为当前相机坐标系。融合拒绝历史信息的内部 mask 不用于挑选评测像素。
