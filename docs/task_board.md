# 当前任务与成员验收

按新版 Spring 2027 任务说明更新于 2026-11-02。技术实施由助手完成，保留原模块主责；成员独立验收单独记录。

| 事项 | 名义主责 | 技术结果 | 成员待验收 |
| --- | --- | --- | --- |
| Middlebury 15 对与 StereoSGBM | A | GT、标定、原始输出、MAE/有效比例复算通过 | B 在本机复算，理解条件误差与 coverage |
| NYUv2 / UniDepth | A | 795/654 标准划分；新版 654 张评测完成 | B 复核 crop、clip、等权汇总和无 GT 对齐 |
| DA V2 / VDA relative 视频 | B | dev 100 帧、test 70 帧完成；FP16/364 | A 对照输入帧、任意单位和整段范围 |
| ONE Improvement | A 实现，B 实验 | 三候选开发复算；α=0.70 冻结；test E0–E5 复算 | 两人理解 15.93% 是变化指标下降，不是准确率 |
| 效率与困难案例 | B | 实测推理/流/融合开销；三个真实条件 | A 看完整视频，复核观察与限制 |
| Application | B | 相对背景虚化 70 帧导出/复算 | A 检查模糊方向、时间同步和边界光晕 |
| 报告与视频包 | A/B | 源码、结果、导出与打包验证流程已备齐 | 检查最终 PDF、ZIP，决定合并及提交 |

操作入口：[当前协议与命令](course_protocol_20261102.md)、[结果](../results/course_20261102/README.md)、[提交](submission.md)。个人验收指南只保存在本机 outputs/personal_acceptance/。

Week 1 的原任务分解保留在 [week1_plan.md](week1_plan.md)，历史运行见 [project_progress.md](project_progress.md)。自录视频、米制视频比较、额外边界指标和 arXiv 发布不再是新版完成前提。
