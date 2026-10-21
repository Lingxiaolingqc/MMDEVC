# Middlebury Quarter Resolution 数据准备与检查

日期：2026-10-20。A / Qinchuan XIAO 的 A01 数据任务。

## 当前结论

15/15 组 trainingQ 场景的左右图、标定、左 GT 视差与 mask 均读取成功，尺寸一致，PFM 有效位置与 mask 对应。已生成 [样本清单](../manifests/middlebury.csv) 和 [GT 检查统计](../results/tables/middlebury_data_check.csv)。10 项回归测试通过；单样本派生数组的保存与读回通过。

这是初次数据读取验证的记录，不是模型结果。当前划分已固定为 Adirondack/Pipes 两个 dev 和其余 13 个 evaluation；相关变体（Motorcycle/MotorcycleE、Piano/PianoL、Playtable/PlaytableP）均保持同侧，见 [批量基线](stereo_suite.md)。NYUv2 的后续准备已完成，见 [数据说明](nyuv2_data.md)。

## 解压位置

在 [官方下载页](https://vision.middlebury.edu/stereo/submit3/) 获取 `MiddEval3-data-Q.zip` 与 `MiddEval3-GT0-Q.zip`。当前路径保留两个解压目录：

```text
data/middlebury/
  MiddEval3-data-Q/MiddEval3/trainingQ/<scene>/
    im0.png
    im1.png
    calib.txt
  MiddEval3-GT0-Q/MiddEval3/trainingQ/<scene>/
    disp0GT.pfm
    mask0nocc.png
```

也支持按官方说明将两包合并到同一个 `MiddEval3/trainingQ/<scene>/` 目录。读取器在 `data_roots.middlebury` 下按场景名配对，不要求移动现有数据；遇到重复场景副本或缺失文件时会报错。只选择 trainingQ，不把 testQ、trainingF 或 trainingH 混入清单。

在 `configs/paths.local.yaml` 中设置 `data_roots.middlebury: data/middlebury`；相对路径相对于仓库根目录。清单路径相对于这个数据根目录。组员若使用不同解压结构，可重建本机清单，但共享仓库应统一层级后再提交，避免无意义的路径差异。

## 复现命令

```sh
python -m pip install -e ".[data]"
python scripts/inspect_middlebury.py --sample Adirondack --save-arrays
python scripts/inspect_middlebury.py --all --write-manifest
python -m unittest discover -s tests -v
```

`--data-root` 可覆盖配置路径；`--no-preview` 跳过绘图。默认创建带 UTC 时间戳的新输出目录；显式 `--output-dir` 必须指向尚不存在的目录，以免覆盖结果。`--write-manifest` 必须与 `--all` 一起使用，只有全部样本通过后才写入清单，并保留已有 split/notes。

## 关键数据约定

- RGB 读取为 `uint8 H×W×3`，通道顺序是 RGB。
- `disp0GT.pfm` 读取为 `float32 H×W` 像素视差；负头字段表示小端，行顺序需要上下翻转。按照官方 SDK 保留原始浮点数，不乘头字段的绝对值。本批 15 个文件的字段均为 `-0.003922`。
- `mask0nocc.png`：`0` 为未知，`128` 为遮挡，`255` 为非遮挡。`nonoccluded = (mask == 255)`；不能使用 `mask > 0` 代替非遮挡区域。
- GT 有效区域包含有真值的遮挡像素，和非遮挡区域分别保存/统计。读取器核对有限视差与 mask 非零位置是否一致。
- `baseline` 单位为毫米；使用对应 Q 分辨率的标定计算 `Z_m = f_px * baseline_mm / (1000 * (d_px + doffs_px))`。不再额外缩放 Q 标定。
- 非有限视差或非正分母对应无效深度，派生深度保存为 NaN；有效零视差在 `doffs > 0` 时仍可对应有限深度。

PFM 处理核对了 [SDK 1.6](https://vision.middlebury.edu/stereo/submit3/zip/MiddEval3-SDK-1.6.zip) 中 `code/imageLib/ImageIO.cpp` 的 `ReadFilePFM`；mask 编码核对 `code/computemask.cpp` 与 `code/evaldisp.cpp`。单位与公式见 [官方标定说明](https://vision.middlebury.edu/stereo/data/scenes2014/)。SDK 仅作为参考读取，未执行或复制进项目源码。

参考 SDK ZIP 的 SHA256：`f9ee9eec52826d00c55ff2805e8751efcb25efc22cbf9c76d79c941f7f370095`。

## 2026-10-20 检查记录

本机 Python 3.13.5、NumPy 2.1.3、Pillow 11.0.0、PyYAML 6.0.2、Matplotlib 3.9.2；仅验证数据处理，不代表官方模型支持该 Python 版本。

运行：`python scripts/inspect_middlebury.py --all --write-manifest --output-dir outputs/middlebury_check_20261020`。

- 15/15 场景通过；GT 有效比例为 89.69%–99.82%，与非遮挡比例分开记录。
- Adirondack：718×496，GT 有效 99.66%，非遮挡 91.29%，GT 派生 Z 深度约 1.573–2.916 m。
- 输出有 `summary.csv`、`metadata.json`、每场景预览和 `contact_sheet.png`。已检查 Adirondack 预览与全场景拼图，RGB/GT 方向对应。
- 元数据保存输入文件 SHA256、代码文件 SHA256、环境与检查范围；运行时包含本分支尚未提交的改动，不能仅凭父提交复现，应使用最终合入的读取器版本。
- 单样本另运行 `--sample Adirondack --save-arrays --no-preview --output-dir outputs/middlebury_sample_20261020`；深度为 float32，mask 为 bool，形状为 496×718，非遮挡 mask 是有效 GT 的子集。

预览是静态 GT 数据诊断，色标按场景范围显示并有单位；不能当作后续视频固定色标对比。原始数据、派生数组和预览均留在被忽略的本地目录，仓库只保存代码、清单与小体积统计。

后续进展：本读取器已用于完整 15 对 StereoSGBM 基线，原始预测和指标复算通过，见 [结果记录](../results/stereo_week1.md)。
