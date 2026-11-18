# part3_improvement：表格与记录

[总导航](../../../PARTS.md) · [本机媒体](../../../outputs/by_part/part3_improvement/README.md)

## UniDepthV2 Residual 控制实验（2026-11-18）

新增米制单图改进使用冻结的 UniDepthV2 与独立 Residual 网络。[计划](../../../docs/residual_research_plan.md)、[当前进度](../../residual_study_status.md)、[报告章节](../../residual_report_section.md)、[分组筛选](../../residual_study_screening.md)、[三种子复验](../../residual_seed_confirmation.md)、[组合消融](../../residual_combinations.md)、[数据量对照](../../residual_data_size.md)和[推理成本](../../residual_runtime.md)分别记录证据。输出根目录是 `outputs/part3_improvement/residual_study_20261118/`；代码入口见 [Part 3 scripts](../../../scripts/part3_improvement/README.md)。

正式训练和测试已完成，见[最终测试分析](../../residual_final_test.md)和[阅读/复现指南](../../../docs/residual_results_guide.md)。本轮结果与下面的历史 E0–E5 视频结果分开记录。

## 历史 Flow-Guided Fusion

selection/ 是三候选开发结果和冻结证据；dev/、test/ 的 comparison.csv 是 E0–E5。ablations.mp4 展示 RGB 和 E0–E4，E5 原 VDA 在 Part 2 的四栏比较中。failure_cases.csv 和困难案例图对应三个实际观察。

## 本目录文件

- [dev/comparison.csv](dev/comparison.csv)
- [dev/verification.json](dev/verification.json)
- [test/comparison.csv](test/comparison.csv)
- [test/verification.json](test/verification.json)
- [selection/development_comparison.csv](selection/development_comparison.csv)
- [selection/fusion_frozen.yaml](selection/fusion_frozen.yaml)
- [selection/selection.json](selection/selection.json)
- [selection/review.json](selection/review.json)
- [failure_cases.csv](failure_cases.csv)

## 原始预测与运行目录

- [测试 E0–E5 原始结果](../../../outputs/course_video_20261102/course_rollercoaster_70/frozen_suite/)
- [开发 E0–E5 原始结果](../../../outputs/course_video_20261102/dev_tokyo_100/alpha070/)
- [测试光流](../../../outputs/course_video_20261102/course_rollercoaster_70/flow/)

这些原始路径只在准备了数据/输出的机器上存在，不随 Git 下载。分类副本中的元数据仍引用原始运行路径；复算请按照 [课程协议](../../../docs/course_protocol_20261102.md) 操作。副本不是新实验，也不是新的评分结果。
