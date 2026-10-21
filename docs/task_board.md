# 第一批任务

以下按实际运行证据更新；A 的技术检查与 B 的独立复核分别记录，不把单样本结果当全测试集成绩。

| ID | 主责 | 任务 | 完成标准 | 状态 |
| --- | --- | --- | --- | --- |
| C01 | A/B | 复核数据与预测格式 | 一份真实预测按协议交接成功 | A 的命令与格式可用；待 B 提供真实预测 |
| A01 | Qinchuan XIAO | Middlebury + NYUv2 清单与读取规范 | 一组 stereo 和一张 RGB/GT 对齐、单位和划分可核验 | Middlebury 15 场景；NYUv2 标准划分与 3 个 train 开发样本检查通过；完整 MAT 哈希已核验，全量提取/逐值复核进行中 |
| A02 | Qinchuan XIAO | StereoSGBM 单样本 | 视差、米制深度、mask 和标定检查结果齐全 | 已跑通 Adirondack；[运行说明](stereo_baseline.md)；B 交叉复现待完成 |
| A03 | Qinchuan XIAO | 单图评测入口 | GT=prediction 正确性检查及 B 的真实预测复算 | 七指标/单位/mask/对齐检查及真实 GT=prediction、保存结果复算通过；待 B 的真实预测 |
| A04 | Qinchuan XIAO | UniDepthV2 ViT-S 试跑 | 一个样本预测、保存和评测全流程 | 已完成 nyuv2_0003 CPU 真实推理；[结果与来源](../results/unidepth_week1.md)；B 独立复现待完成 |
| A05 | Qinchuan XIAO | StereoSGBM 15 对 | dev 调参后冻结配置；全量结果可复算 | 已完成：2 dev/13 evaluation、最多 3 组参数、15 对保存/复算；[失败例子](../results/stereo_week1.md) |
| B01 | Jun Yu Eugene YIP | DA V2 Small 单图 | 原始浮点预测、表示说明、元数据和命令 | 待开始 |
| B02 | Jun Yu Eugene YIP | 视频采集与清单 | 稳定 clip/frame ID，场景隔离的 dev/test 划分 | 待开始 |
| B03 | Jun Yu Eugene YIP | 逐帧视频推理 | 原始序列、时间戳和固定色标视频 | 待开始 |
| B04 | Jun Yu Eugene YIP | VDA Small metric 试跑 | 短片推理、帧数/单位检查、运行模式记录 | 待开始 |
| C02 | A/B | 周末交叉复现 | 一页周总结、已验证结果、未完成项和下一步 | A 的精选结果已整理；双方交叉复现与共同周总结待完成 |

完整日程和额外工作见 [week1_plan.md](week1_plan.md)。先处理 C01/A01/B01 的依赖，再扩大运行范围。

2026-10-21 的 A 运行记录见 [实验索引](../results/run_index.csv)。视频、融合和完整报告尚未开展；B 行的状态以组员后续提供的运行记录为准。
