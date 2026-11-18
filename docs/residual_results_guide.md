# Residual 实验：阅读、复现与记录

本研究在冻结的官方 UniDepthV2 Small 输出上训练一个小型深度修正网络。基础模型不更新；LoRA 不属于这次实验。完整设计见 [研究计划](residual_research_plan.md)，实测进度见 [当前状态](../results/residual_study_status.md)。

## 先看什么

[中文阶段汇总](../results/residual_stage_summary_20261118.md)与[可直接拉取的实验记录](../results/residual_records_20261118/README.md)已保存。后者包含训练曲线、逐图指标和源码身份；大型权重与预测数组保留本地。

1. [报告段落](../results/residual_report_section.md)和[最终测试分析](../results/residual_final_test.md)：研究问题、模型、实验设计、三个种子的结果与成功/失败案例。正式训练和测试已完成。
2. [单因素筛选](../results/residual_study_screening.md)：输入、架构、输出、损失和历史组合，包含失败与无变化结果。
3. [三个种子](../results/residual_seed_confirmation.md) 与 [组合/去除对照](../results/residual_combinations.md)：判断收益是否稳定、组件组合是否还有价值。
4. [数据量/学习曲线](../results/residual_data_size.md)、[分辨率](../results/residual_resolution.md)、[速度](../results/residual_runtime.md)：收益、过拟合和成本。
5. [边界指标解释](../results/residual_boundary_interpretation.md)：边界区域误差降低不等于轮廓被重新恢复。

## 划分和最终方案

NYUv2 官方训练集795张分成577张 fitting和218张 validation，场景不重叠。开发阶段只在577张上优化，在218张上选择设计。官方654张 test 不用于此次设计选择，但早期项目已经查看过它，因此不能称为从未接触过的盲测。

最终锁定 R0 和内部特征 CNN，各跑320112、320113、320114三个种子，修正分辨率240x320。全795张重训固定 R0=2轮、内部特征CNN=76轮，来自三个开发运行最优轮数的中位数；学习率保留80轮余弦周期。最终只用最后一轮权重，不根据 test 挑选轮数或种子。代表性图片固定展示种子320112；三个种子的平均指标不是把预测图做 ensemble。

## 代码入口

- 官方基础模型和适配：`third_party/UniDepth/`、`src/p3_depth/models/runtime.py`。
- 自行实现的 Residual 模型、数据、训练和评测：`src/p3_depth/residual/`。
- 运行/分析入口：`scripts/part3_improvement/*residual*.py`。
- 固定方案：`configs/residual_study.json`、`residual_confirmation.json`、`residual_combinations.json`、`residual_followup_selection.json`、`residual_final_proposal.json`。
- 具体组合、数据量、分辨率任务：`configs/residual_jobs/`。每个任务都包含划分、种子、预算和 checkpoint 选择规则。

使用仓库已配置的 CUDA Python。下面按开发→冻结→重训→测试列出首次运行顺序。训练入口可用 resume 验证并复用完整运行；`bases` 已完成时应跳过，它会拒绝覆盖结果。不要删除记录来绕过配置或源码不一致检查。现有结果已经完整，阅读和重新分析不需要重训。

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
$study = "outputs/part3_improvement/residual_study_20261118"
& $py -X utf8 scripts/part3_improvement/prepare_residual_cache.py
& $py -X utf8 scripts/part3_improvement/run_residual_study.py --action bases
& $py -X utf8 scripts/part3_improvement/run_residual_study.py --action edges
& $py -X utf8 scripts/part3_improvement/run_residual_study.py --action screen --resume
& $py -X utf8 scripts/part3_improvement/run_residual_study.py --action confirm --resume
```

接下来按记录运行组合、数据量、分辨率任务，用 `run_residual_jobs.py --jobs <对应JSON路径...> --resume`；各阶段用 `analyze_residual_followups.py --section combination` / `data_size` / `resolution` 分析。不要将不同训练预算的结果当成控制变量对照。

完成开发证据后，首次冻结使用：

```powershell
& $py -X utf8 scripts/part3_improvement/run_residual_final.py --action freeze --proposal configs/residual_final_proposal.json
```

冻结已存在时跳过这条命令；它有意拒绝覆盖。读取实际冻结文件中的六个任务进行重训：

```powershell
$freeze = Get-Content -LiteralPath "$study/frozen_selection.json" -Raw | ConvertFrom-Json
$jobs = @($freeze.refits | ForEach-Object { "$study/jobs/$($_.run_id).json" })
& $py -X utf8 scripts/part3_improvement/run_residual_jobs.py --jobs $jobs --resume
if ($LASTEXITCODE -ne 0) { throw "Final refit failed" }
& $py -X utf8 scripts/part3_improvement/prepare_residual_cache.py --split test
if ($LASTEXITCODE -ne 0) { throw "Test cache failed" }
& $py -X utf8 scripts/part3_improvement/run_residual_final.py --action evaluate
if ($LASTEXITCODE -ne 0) { throw "Test evaluation failed" }
& $py -X utf8 scripts/part3_improvement/analyze_residual_followups.py --section test
& $py -X utf8 scripts/part3_improvement/render_refiner_cases.py
```

冻结验证的是当前工作目录的数值源码、开发证据、划分和实际任务。换版本或换实验时应使用独立研究目录和明确的新配置，不应覆盖原证据。代码、固定配置和小型分析报告进入Git；数据、官方权重、训练权重和完整生成输出保留本地，分享复现时需同时提供对应数据/权重或按流程重跑。

## 每次训练保留什么

研究输出根目录是 `outputs/part3_improvement/residual_study_20261118/`。

| 位置 | 内容 |
|---|---|
| `runs/<运行名>/` | 单因素与种子确认结果 |
| `followups/<运行名>/` | 组合、数据量、分辨率、全开发集重训 |
| `config.json`、`provenance.json`、`source_at_run/` | 实际设置、源码快照、版本与数据身份 |
| `history.json`、`logs/` | 完整训练曲线和控制台记录 |
| `best.pt`、`final.pt` | 已选择和最后迭代权重；最初筛选阶段保存被选中的权重 |
| `validation/`、`visualizations/` | 开发评测逐图表、总体指标和固定案例 |
| `final_test/<方法>/` | 最终逐图预测数组、指标和固定预览 |
| `analysis/<阶段>/` | 对照表、场景bootstrap区间、图表、解释 |
| `runs.jsonl`、`cleanup.jsonl` | 成功/失败运行以及删除前后的清理记录 |

验收时先确认所读目录和数据划分，再看数值、图片以及记录中的限制。小型工程 smoke 和正确性测试通过不代表正式效果已经提高。冗余恢复点与可重建输入缓存可以在依赖完成后清理；原始数据、官方权重、必要训练权重和唯一结果证据保留。
