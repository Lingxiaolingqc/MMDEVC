# 模型与视频入口检查：2026-10-28

由助手实现和执行，保留 A/B 原模块主责；成员独立复现待完成。本记录只说明真实小样本检查，不作为 654 张标准测试或自录视频实验成绩。完整配置、源代码/权重/预测哈希、指标及输出目录见 [provenance](depth_models_smoke_provenance.json)。复现命令见 [模型说明](../environments/depth_models.md)。

## 单图

DA V2 relative 和 Hypersim metric 均运行三个 train-side dev ID：`nyuv2_0003`、`nyuv2_0005`、`nyuv2_0010`。输入 encoder 为 434×574、输出 480×640，CPU FP32。采用既定 NYUv2 crop/range，逐图指标等权平均，coverage 均为 100%。原始预测、valid mask、逐图记录和汇总已保存并复算一致。

| 模型与评测方式 | AbsRel | SqRel (m) | RMSE (m) | RMSE-log | δ1 | δ2 | δ3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DA V2 Small relative，GT inverse scale/shift 对齐 | 0.072191 | 0.033619 | 0.272239 | 0.104608 | 0.925257 | 0.990204 | 0.999778 |
| DA V2 Small Hypersim metric，无 GT 对齐 | 0.290626 | 0.331903 | 0.913820 | 0.275392 | 0.463013 | 0.890264 | 0.995937 |

两行使用了不同的尺度信息，不能据此声称 relative 模型的直接米制准确率更好。数据只有三个开发样本，不据此选出最终优胜模型。

新 UniDepth 批量入口另在 `nyuv2_0003` 检查，与 Week 1 原入口的七项指标逐值一致（RMSE 0.2988495998353366 m），保存预测复算通过；该单样本结果不与上表三图平均值排名比较。

## 视频

输入来自固定 VDA 官方 checkout 内的 `assets/example_videos/Tokyo-Walk_rgb.mp4`，只取 24 帧。它是公开示例，不是自录素材，未写入正式视频清单。统一缓存为 RGB 180×320、保留原始 PTS，目标采样 10 fps；实际帧间隔以索引为准。

- DA relative / metric 实际 encoder 为 `[1,3,140,252]`；VDA metric 为 `[32,3,140,252]`。空间尺寸相同，VDA 保留官方时间窗口。
- 三个模型各自保存 24 张原尺寸浮点预测及 mask。relative 单位任意；两种 metric 输出为 m，无 GT 对齐。
- 固定色标比较视频包含 RGB 和三个模型，另保存首、中、末帧预览；重新解码确认 24 帧及原始时间戳一致。
- 输出在 `outputs/video_smoke_20261028/rendered_complete/`，完整视频与原始数组不进 Git。
- 这次 short side=140 仅用于接口检查，低分辨率会使细节粗糙。室外街景也不是 indoor checkpoint 的正式准确率证明；不据此评价时序改进。

## 检查边界

36 项单元测试通过，覆盖原有数据/指标/SGBM 规则、新批量划分及聚合、真实可变帧率视频解码。新代码运行时尚未提交，已保存相关实现 SHA256；下一阶段正式运行需记录已提交版本。

尚未完成：654 张标准测试、真实自录素材与冻结视频设置、边界/时序指标、E0–E5、Demo 和正式报告。当前只验证 CPU 路径；GPU 结果以之后的实机记录为准。
