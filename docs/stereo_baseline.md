# StereoSGBM 单样本基线

主责：A / Qinchuan XIAO。第一张开发样本为 Middlebury v3 `trainingQ/Adirondack`，原始尺寸 718 × 496。它已在 manifest 标为 `dev`；其余 14 场景暂为 `unassigned`。本次用于验证端到端计算与输出接口，不代表 15 场景最终精度或正式独立测试。

## 运行

在仓库根目录执行（无需 GPU、PyTorch、模型权重或 opencv-contrib）：

```sh
python -m pip install -e ".[data,stereo]"
python -m unittest discover -s tests -v
python scripts/run_stereo_baseline.py --sample Adirondack
```

数据路径使用 `configs/paths.local.yaml`；也可用 `--data-root "你的 Middlebury 目录"` 覆盖。输出默认自动创建唯一的 `outputs/stereo_Adirondack_<UTC时间>/`。`--output-dir` 可指定新目录，已有目录会报错，避免覆盖历史实验。入口读取 `manifests/middlebury.csv`，只处理选中的一对图。更换场景前须先确定其 `dev/evaluation` 划分。

## 匹配与单位

1. 输入是已校正的左右 RGB 图，按 RGB 顺序转灰度；不缩放、不裁剪。
2. 配置为 `configs/stereo_sgbm.yaml`：block size 5、uniqueness 10、speckle window 100、speckle range 2、pre-filter cap 31；使用 `SGBM_3WAY`，灰度的 `P1=200`、`P2=800`。这些是初始固定参数，未对 Adirondack GT 搜索最优值。
3. 左视差定义为 `dL=xL-xR`，搜索下界 0。只用标定的保守 `ndisp=73` 向上取 16 的倍数得到 80，搜索整数候选为 0–79 px。**不使用 GT 的最小/最大视差或 `vmin/vmax` 调整搜索。**
4. OpenCV 的 int16 输出除以 16 才是像素视差；无效哨兵 `(minDisparity-1)*16` 转为 NaN。零视差不自动判为无效。
5. 交换左右输入再匹配，右视差定义为 `dR=xR-xL`，下界为 `-79`，共 80 个候选。两个匹配器使用相同 uniqueness/speckle 设置以及内置 `disp12MaxDiff=1`。
6. 显式左右检查：在同一行用 `xR=xL-dL`、最近邻 `floor(xR+0.5)` 采样右视差。对应坐标必须在图像内，左右视差均有效，并满足 `abs(dL+dR)<=1 px`。未通过的左视差保留 NaN；不做 WLS 或空洞填充。
7. 用相机光轴 Z 深度公式转换，单位为米：

   `Z = f_px × (baseline_mm / 1000) / (dL + doffs_px)`

   Adirondack：`f=1038.018 px`、`baseline=176.252 mm`、`doffs=53.271 px`。只保留有限且正的 Z，不使用 GT 尺度对齐或深度截断。

这些约定参考 [OpenCV StereoSGBM 参数](https://docs.opencv.org/4.x/d2/d85/classcv_1_1StereoSGBM.html)、[官方 Python 样例的 fixed-point 转换](https://github.com/opencv/opencv/blob/4.x/samples/python/stereo_match.py)、[右匹配器的符号/范围](https://github.com/opencv/opencv_contrib/blob/4.x/modules/ximgproc/src/disparity_filters.cpp) 和 [Middlebury 标定说明](https://vision.middlebury.edu/stereo/data/scenes2014/)。我们的右匹配器保留 uniqueness/speckle 筛选，不等同于 ximgproc 的辅助右匹配器设置。

## 单样本评测口径

分别固定两块 **GT 定义的区域**：`all_gt` 为所有已知 GT（mask 为 128 或 255），`nonoccluded_gt` 只包括 mask 为 255。无额外 crop 或深度区间过滤，预测失效不会改变 GT 区域。

- `coverage = 有效预测像素数 / GT 区域像素数`；另报告无效比例及绝对像素数。
- `bad1/bad2 = (有效预测中绝对视差误差严格大于 1/2 px 的数量 + 无效预测数) / GT 区域像素数`。像素单位对应 Quarter Resolution；这是项目诊断指标，不声称与官方榜单分辨率/容差协议一致。
- 深度指标只在该 GT 区域内存在有效预测的像素上计算，所有字段明确标记 `on_valid`。不能脱离 coverage 比较不同方法。若完全无有效预测，误差为 JSON `null`，bad 比例为 1；空 GT 区域直接报错。

设有效预测为 `p`、GT 为 `g`、该交集大小为 `N`：

| 指标 | 定义 |
| --- | --- |
| AbsRel | `mean(abs(p-g)/g)`，无单位 |
| SqRel | `mean((p-g)^2/g)`，单位 m |
| RMSE | `sqrt(mean((p-g)^2))`，单位 m |
| RMSE-log | `sqrt(mean((ln(p)-ln(g))^2))`，自然对数 |
| δ1/2/3 | `mean(max(p/g,g/p) < 1.25^k)`，严格小于，k=1/2/3 |

此处仅实现 stereo 单样本直接 metric 诊断；多方法正式汇总、NYUv2 crop、relative alignment、边界和时序评测仍待实现与冻结。

## 产物与交接

```text
outputs/<run_id>/
  predictions/Adirondack.npy     # float32 H×W，Z 深度 m，无效 NaN
  masks/Adirondack.npy           # bool H×W，有效预测
  disparities/Adirondack.npy     # float32 H×W，LR 检查后左视差 px
  diagnostics/                  # 原始左右 int16、LR 前视差、残差、固定 GT 区域
  predictions.csv               # 与 A/B 交接协议一致的相对路径索引
  config_resolved.yaml           # 包含实际搜索范围、P1/P2 与 LR 规则
  metrics.json / metrics.csv     # all_gt / nonoccluded_gt 各一条
  metadata.json                 # 代码 SHA、数据/代码/产物哈希、环境、标定、命令、计时
  visualizations/Adirondack.png  # RGB、视差、深度、mask、误差和 LR 残差
  run.log
```

视差图共享 `[0,80] px` 色标，深度图共享 `[0,4] m`，绝对深度误差 `[0,0.2] m`，LR 残差 `[0,2] px`；色标外数值仅显示饱和，不修改原始数组。灰色表示无效/不可用。正式运行应在干净工作树上执行；之后单独提交精选结果与实验索引，保持代码 SHA 可复现。

计时：CPU、OpenCV 单线程、OpenCL 关闭，预热一对再测量一对；包含灰度转换、两个匹配器创建与匹配、显式 LR 检查及深度转换，不包含文件读写、GT 评测和绘图。只作这次运行记录，不视为效率 benchmark。

B 的下一步：获取该分支、安装依赖、放置同版本数据、运行同一命令；核对 shape/dtype/NaN/mask/标定、逐项复算指标，再把复核结论填入实验索引。当前不将 B 的复核记为完成。
