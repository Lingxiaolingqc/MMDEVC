# 第一批任务

以下为待执行任务；初始化完成不代表模型已经跑通。可用本表管理两人工作，也可之后同步到 GitHub Issues。

| ID | 主责 | 任务 | 完成标准 | 状态 |
| --- | --- | --- | --- | --- |
| C01 | A/B | 复核数据与预测格式 | 一份真实预测按协议交接成功 | 待开始 |
| A01 | Qinchuan XIAO | Middlebury + NYUv2 清单与读取规范 | 一组 stereo 和一张 RGB/GT 对齐、单位和划分可核验 | 进行中：Middlebury 15 场景检查通过，Adirondack 为 dev；其余划分和 NYUv2 待完成 |
| A02 | Qinchuan XIAO | StereoSGBM 单样本 | 视差、米制深度、mask 和标定检查结果齐全 | 已跑通 Adirondack；[运行说明](stereo_baseline.md)；B 交叉复现待完成 |
| A03 | Qinchuan XIAO | 单图评测入口 | GT=prediction 正确性检查及 B 的真实预测复算 | 部分完成：stereo 单样本与 GT=prediction 检查通过；通用预测入口及 B 预测复算待完成 |
| A04 | Qinchuan XIAO | UniDepthV2 ViT-S 试跑 | 一个样本预测、保存和评测全流程 | 待开始 |
| B01 | Jun Yu Eugene YIP | DA V2 Small 单图 | 原始浮点预测、表示说明、元数据和命令 | 待开始 |
| B02 | Jun Yu Eugene YIP | 视频采集与清单 | 稳定 clip/frame ID，场景隔离的 dev/test 划分 | 待开始 |
| B03 | Jun Yu Eugene YIP | 逐帧视频推理 | 原始序列、时间戳和固定色标视频 | 待开始 |
| B04 | Jun Yu Eugene YIP | VDA Small metric 试跑 | 短片推理、帧数/单位检查、运行模式记录 | 待开始 |
| C02 | A/B | 周末交叉复现 | 一页周总结、已验证结果、未完成项和下一步 | 待开始 |

完整日程和额外工作见 [week1_plan.md](week1_plan.md)。先处理 C01/A01/B01 的依赖，再扩大运行范围。

2026-10-20 数据准备记录见 [Middlebury 数据检查](middlebury_data.md)，StereoSGBM 实现与诊断口径见 [单样本基线](stereo_baseline.md)。单样本通过不代表完整数据集评测或单目模型已完成。
