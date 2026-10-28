# NYUv2 标准测试结果：2026-10-28

三种固定 Small 模型均已运行标准 654 个 test ID，并从保存的原始预测重新计算全部指标。UniDepthV2 与 DA V2 Hypersim metric 直接输出米制 Z 深度；DA V2 relative 另作使用 GT 对齐的分析。数据准备、crop、单位与七指标沿用 [评测协议](../docs/evaluation_protocol.md)。

三者在同一 RTX 4060 Laptop 上使用 FP16，实际 encoder 输入均为 434×574，输出为原图 480×640。UniDepth 使用 resolution_level=0，DA 使用 short side=434。这是本项目记录的配置，不宣称与论文其他模型规模或默认分辨率的成绩相同。

## 深度与边界

| 模型 / 评测方式 | AbsRel | RMSE (m) | full-GT δ1 | coverage | 边界 F1 |
| --- | --- | --- | --- | --- | --- |
| UniDepthV2 ViT-S / direct metric | 0.112243 | 0.416660 | 0.893111 | 100% | 0.522902 |
| DA V2 Small Hypersim / direct metric | 0.280108 | 0.815147 | 0.519935 | 100% | 0.505657 |
| DA V2 Small relative / GT aligned | 0.062102 | 0.311352 | 0.953786 | 99.8471% | 0.575487 |

完整七指标、full-GT δ1/2/3、逐图深度/边界记录和资源统计在 [结果目录](nyuv2_test_20261028/)。[summary.csv](nyuv2_test_20261028/summary.csv) 是汇总表；[provenance.json](nyuv2_test_20261028/provenance.json) 记录固定模型、运行配置、源代码及数据清单哈希。

直接米制比较中，本次配置的 UniDepth 在 AbsRel、RMSE 和 δ1 上优于 DA Hypersim。相对模型使用了每张图的 GT 拟合尺度与偏移，不能与 direct metric 两行混为同一准确率排名。

相对模型在 `nyuv2_0650` 得到负的 inverse-depth 拟合比例，无法产生符合已声明方向的对齐深度。该图作为失败保留：coverage=0、full-GT δ=0、条件误差未定义。相对模型的条件误差平均来自 653 张，full-GT 指标和 coverage 仍包含全部 654 张。原始预测未改动，未翻转方向或删除样本。

边界采用本项目明确的 depth-discontinuity 定义：四邻居最大绝对 log-depth 差超过 ln(1.1)，在固定 GT 区域内匹配，距离容差 2 px。表中 precision/recall/F1 按边缘像素汇总；缺失预测仍损失 GT recall。它不是对某篇论文边界指标数值的直接复现。

## 资源与可视化

| 模型 | 平均推理时间 | 吞吐率 | peak allocated / reserved |
| --- | --- | --- | --- |
| UniDepthV2 ViT-S | 43.68 ms | 22.89 fps | 0.393 / 0.445 GiB |
| DA V2 Small metric | 56.83 ms | 17.60 fps | 0.270 / 0.311 GiB |
| DA V2 Small relative | 58.53 ms | 17.08 fps | 0.270 / 0.311 GiB |

预热一个 train 图像后，在 654 张图上逐张同步计时。包括官方预处理、传输、模型与原尺寸恢复，排除文件读取、保存和评测；这是本机一次完整运行的记录。显存为 PyTorch 分配器统计，不含整台电脑其他进程。

[比较图](nyuv2_test_20261028/comparison.png) 按可重复规则选取 UniDepth RMSE 中位样本 `0285`、最大误差样本 `0090`，以及 relative 对齐失败样本 `0650`。固定显示 0–10 m，显示裁剪不影响原始数值或指标。`0090` 的 UniDepth RMSE 为 3.9283 m，说明平均性能较好仍可能出现显著尺度误差。`0650` 画面包含镜面，反射可能影响预测与 GT 的对应解释；仅凭此图不能确定失败原因。

## 复现

在 [CUDA 环境](../environments/cuda.md) 和固定模型资产准备好后，对 `unidepth`、`dav2_metric`、`dav2_relative` 分别执行，始终使用新输出目录：

```sh
python scripts/run_nyuv2.py --model unidepth --subset test --device cuda --precision fp16 --output-dir outputs/reproduce/unidepth
python scripts/verify_nyuv2_run.py --run outputs/reproduce/unidepth
python scripts/evaluate_nyuv2_boundaries.py --run outputs/reproduce/unidepth --output-dir outputs/reproduce/unidepth_boundaries
```

成功标准：654 个唯一标准 test ID，预测/GT 哈希检查通过，所有保存指标复算一致。允许相对对齐失败，但必须明确计数、保留分母。深度推理版本分别记录为 `e3ed7d8`（两个直接米制模型）和 `1429c30`（相对模型）；相关实现文件的完整 SHA256 随结果保存。
