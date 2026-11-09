# 实验复现入口

从仓库根目录运行。按 [分阶段重跑指南](../docs/rerun_by_part.md) 完成主实验，再按 [补充实验](../docs/supplementary_experiments.md) 和 [扩展实验](../docs/extension_results_20261109.md) 复现控制实验。默认输出见 [output_layout.yaml](../configs/output_layout.yaml)。

| 目录 | 运行与评测 | 分析与绘图 |
| --- | --- | --- |
| [Part 1](part1_baseline/README.md) | Stereo 单样本/全场景、逐帧 DA、课程 Stereo 指标 | Stereo 分布、LR 开关、逐帧视频诊断 |
| [Part 2](part2_models/README.md) | UniDepth、NYU 主评测/边缘指标、VDA offline/streaming | NYU 误差/尺度/边界、DA/VDA、离线/流式比较 |
| [Part 3](part3_improvement/README.md) | 光流、E0–E5、dev 冻结、history/confidence 消融、定向 refinement | 时序/置信度诊断、控制实验、区域统计 |
| [Application](application/README.md) | 同参数 DA/E4 背景虚化及阈值对照 | 应用权重、输出差、匹配帧图 |

## 共用入口

| 文件 | 作用 |
| --- | --- |
| [check_setup.py](check_setup.py) | 基础目录、清单和配置检查 |
| [inspect_middlebury.py](inspect_middlebury.py) | Middlebury 完整性与 manifest |
| [prepare_nyuv2.py](prepare_nyuv2.py) | NYUv2 提取与标准划分 |
| [prepare_video.py](prepare_video.py) | 原始时间戳、视频帧缓存和登记 |
| [prepare_model_assets.py](prepare_model_assets.py) | 固定官方源码与权重准备 |
| [evaluate_depth.py](evaluate_depth.py) | 通用保存预测评测；NYU 正式成绩使用 Part 2 课程协议 |
| [render_depth_video.py](render_depth_video.py) | 相同输入、整段固定色标的多方法视频 |

共用内部模块负责路径、文件读写、推理适配、自动媒体和匹配帧图；它们被各 Part 导入，不是额外实验。共享算法在 `src/p3_depth/`，官方网络在本地 `third_party/`。

## 分析与图表

[分析流程](../docs/analysis_workflow.md) 列出各 Part 的指标复算和图表命令。新增的 [Stereo 选定场景箱型图](part1_baseline/plot_stereo_distributions.py) 和 [NYU 匹配误差图](part2_models/plot_nyuv2_cases.py) 从已保存数据出图，不依赖报告目录、PDF 检查或文档打包工具。

实验中的 `verify_nyuv2_run.py`、`verify_fusion_suite.py` 和 `verify_supplementary_ablations.py` 复算预测、掩码、权重、指标和时间戳，属于结果复现链。它们不会检查报告排版。

DA、VDA、NYU、光流和融合入口支持 `--visualize-only`，补可视化时无需重跑模型。NYU 主结果使用 `part2_models/evaluate_course_nyu.py`；旧七指标仅作诊断，不能替代课程主表。

报告制作、历史迁移、打包工具及旧米制网页演示不属于当前实验代码树。已有来源记录保留原始路径和哈希，按其 Git commit 追溯。
