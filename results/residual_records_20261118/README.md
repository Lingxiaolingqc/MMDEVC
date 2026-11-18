# Residual 实验记录快照：2026-11-18

这是已完成实验的轻量记录，共 378 份数据文件、7.38 MiB。包含 51 次正式训练、1 次工程 smoke，以及 10 组完整的 654 图像测试。没有启动新训练，也没有修改旧实验。

## 如何阅读

| 目录／文件 | 内容 |
|---|---|
| `runs/` | 第一轮单因素筛选、三个种子确认，及单独标记的 smoke |
| `followups/` | 组合、去除、数据量、分辨率及全开发集重训 |
| `history.json` | 每轮／检查点训练曲线，验证指标及已记录事件 |
| `config.json`、`provenance.json` | 实际配置、种子、划分、源码哈希、环境版本 |
| `validation/` | 验证逐图指标与汇总 |
| `final_test/` | 四个简单基线和六个独立训练模型的完整测试指标 |
| `analysis/` | 匹配对照、种子汇总、场景重采样区间和定性检查记录 |
| `frozen_selection.json` | 测试前冻结的方法、种子、训练时长及源码身份 |
| `run_ledger.json` | 原始运行事件序列，包含失败与恢复记录 |
| `index.json` | 每个导出文件的来源及 SHA-256，用于核对副本 |

原始数据、官方权重、训练权重、逐像素预测、媒体和逐次源码快照继续保留在本机的 `outputs/part3_improvement/residual_study_20261118/` 等原路径，不放入本次 Git 记录。`provenance.json` 仅省略当时的 Git 工作区文件列表，保留源码身份和环境；运行事件由 JSONL 转成 JSON 数组。其余文件为按字节复制。源记录和副本哈希分别列出。

开发：577 fitting / 218 场景不重叠 validation。最终：全 795 开发数据重训、654 test。历史 test 接触已披露。三种子均值是指标均值，不是预测 ensemble；按场景 bootstrap 的区间与种子标准差含义不同。较早阶段记录中的“test 尚未开始”是当时状态，完整结果见 `final_test/`。

当前最佳输入已包含 RGB、预测深度和单一 decoder 初始 latent。不存在已完成的多阶段 decoder 特征融合实验。此快照不把后续讨论当成已批准或已运行的实验。

[中文阶段结论](../residual_stage_summary_20261118.md) · [完整报告](../residual_report_section.md) · [复现指南](../../docs/residual_results_guide.md)
