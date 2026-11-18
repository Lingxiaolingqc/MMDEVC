# Part 3：改进实验

## UniDepthV2 Residual 控制实验（2026-11-18）

计划：[Residual research plan](../../docs/residual_research_plan.md)。进度：[当前实测状态](../../results/residual_study_status.md)。这条米制单图路线与下面的历史视频融合实验分开。

```powershell
python scripts/part3_improvement/prepare_residual_cache.py
python scripts/part3_improvement/run_residual_study.py --action bases
python scripts/part3_improvement/run_residual_study.py --action edges
python scripts/part3_improvement/run_residual_study.py --action train --candidate R0 --smoke
python scripts/part3_improvement/run_residual_study.py --action screen --resume
python scripts/part3_improvement/run_residual_study.py --action confirm --resume
python scripts/part3_improvement/analyze_residual_study.py
```

完整的开发、冻结、重训和测试命令见[复现指南](../../docs/residual_results_guide.md)。`render_refiner_cases.py` 从已保存的测试预测生成固定与排名案例，不进行模型训练。已有 `bases` 结果应跳过，入口会拒绝覆盖。

运行前使用已配置 CUDA 的 Python 环境。`screen` 仅运行记录在配置里的第一轮单因素筛选，不自动选择组合或读取 test。每次训练保存源码、配置、完整曲线、逐图指标、最优权重和预览；已完成运行用 `--resume` 验证后跳过。训练中断则从最近一轮完整 checkpoint 继续，要求源码和配置未变。自动清理仅限已完成运行的冗余 resume checkpoint，删除前后均写清单。默认输出在 `outputs/part3_improvement/residual_study_20261118/`。

`confirm` 使用 `configs/residual_confirmation.json` 中已经确定的三组候选、R0 和三个种子，不自动改选。后续组合、消融、数据量与重训运行使用 `run_residual_jobs.py --jobs <预先保存的配置.json> --resume`；配置明确训练/验证样本、预算、输出形式和选择规则。数据量实验评分固定最终迭代，最终全开发集重训不使用验证/测试选 checkpoint。后续运行同时保留最优与最后迭代权重，并通过同一个本地锁阻止重复训练。

组合配置见 `configs/residual_combinations.json`，后续选择与任务清单见 `configs/residual_followup_selection.json`；具体不可变任务位于 `configs/residual_jobs/`。运行后用 `analyze_residual_followups.py --section combination`（或 `data_size`、`resolution`、`test`）生成对应阶段的匹配对照、场景重采样区间和图表。`benchmark_residual.py --checkpoints <权重路径...> --tag <新名称>` 单独测量含基础模型、必要翻转和修正网络的推理成本，不能与训练同时运行。最终用 `run_residual_final.py` 冻结已选方法后再执行全开发集重训和正式测试。

## 历史 Flow-Guided Fusion 实验

[总导航](../../PARTS.md) · [结果与消融](../../results/by_part/part3_improvement/README.md)

补充实验入口：[历史来源/置信度消融](run_supplementary_ablations.py)、[复算](verify_supplementary_ablations.py)。按 [补充实验指南](../../docs/supplementary_experiments.md)由成员运行观看、助手定量分析和核验。新输出位于 outputs/part3_improvement/supplementary/，原默认 E0–E5 行为不变。

## 代码来源和阅读顺序

本项目由助手实现对 DA V2 relative 输出的后处理；改进方向来自课程任务书。Farneback 光流本身调用 OpenCV，融合和评测逻辑在本仓库。

1. [光流](../../src/p3_depth/temporal/flow.py)：先读 RGB 对应关系与双向检查。
2. [融合](../../src/p3_depth/temporal/fusion.py)：当前 DA 预测 r、历史预测 h，经光流对齐后按 (1-w)r+wh 融合。E4 的 w 会随图像差异、光流残差、预测分歧和边缘强度下降。
3. [时序评测](../../src/p3_depth/evaluation/temporal.py)：所有方法共用 RGB/flow 参考区域；归一化变化不是深度准确率。
4. [选参冻结](../../src/p3_depth/temporal/selection.py)：三个开发候选及冻结凭据。
5. [实验流程](../../scripts/part3_improvement/run_fusion_suite.py)和[保存结果复算](../../scripts/part3_improvement/verify_fusion_suite.py)。

α=0.70 是历史权重上限，不是每个像素都用 70% 历史。第一帧保持原样；历史不可靠时 w=0。当前帧的相对范围只用于置信权重，不把输出逐帧重新归一化。

E0 原 DA；E1 同位置平滑；E2 光流对齐融合；E3 再加可靠性过滤；E4 再加自适应权重；E5 原 VDA 参照。VDA 不经过我们的融合。

## 使用入口

```powershell
python scripts/part3_improvement/prepare_flow.py --help
python scripts/part3_improvement/run_fusion_suite.py --help
python scripts/part3_improvement/verify_fusion_suite.py --help
python scripts/part3_improvement/freeze_fusion_config.py --help
python scripts/render_depth_video.py --help
```

新运行的数据与选参顺序见 [分阶段重跑指南](../../docs/rerun_by_part.md)。默认输出在 outputs/part3_improvement/；光流入口自动生成诊断视频，融合入口自动生成 E0–E4 独立视频以及主比较/消融影片。E5 引用 Part 2 原 VDA。新冻结使用 manifests/videos_current.csv，旧记录仍按原清单验证。历史在 Tokyo 选定 0.70 后用于 rollercoaster，本轮复现保留该设定并重新核查开发证据。

E4 测试变化指标下降 15.93%；E2/E3 的该指标更低，E4 并非数值最优。更低变化可能伴随过度平滑，现有无 GT 视频无法证明准确率或运动延迟改善。具体速度从各次 comparison.csv 读取；estimated_pipeline_fps 是推理+CPU 光流+融合的加和估算，不含视频解码和写盘。对应影片可观察 E1 拖影与 E4 的取舍。
