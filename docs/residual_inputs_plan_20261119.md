# Residual 输入追加实验：2026-11-19

用户授权直接执行并按阶段表格汇报。上一轮冻结研究不覆盖；本轮不含 LoRA。

## 固定条件

577 fitting / 218 scene-disjoint validation；旧的654 test已查看。本轮是追加探索，validation重复使用。固定冻结UniDepthV2、原始深度基底、CNN、加性输出、AbsRel、240×320、AdamW3e-4、batch8、80轮上限及原早停规则，种子320112初筛，320113/114复验。最终各方法在全795上固定轮数重训，只评价冻结最终迭代。

## 候选

S0=RGB+log原预测+初始latent，旧结果作为参照并做实现一致性检查。六个新候选：I1去RGB；I2加预测深度Sobel；I3加RGB/深度联合Sobel；I4加confidence；M1只用后期decoder特征；M2初始+后期。单分支投影16通道；双分支各8，总预算16。基础模型、主干不更新，投影可训练。

提取官方depth_layer.ups的中间与最后输出，初始latent来自官方infer depth_features。必须检查官方resize/padding及原始预测一致性；不改官方源码、不输入GT特征。原始特征保持本来的空间网格，在投影后对齐到修正网格。NYU固定图像若padding非零，先解决对齐并独立验证，不能悄悄改变参照。

## 事先约定的晋级与停止规则

相对S0同种子AbsRel降低至少1%，F1下降不超过0.002且边界区域误差增加不超过1%，才是主要晋级候选。边界专项允许F1增加至少0.002且AbsRel增加不超过0.2%；它不自动替换准确度主模型。最多两个候选补另两种子；先按合格的AbsRel排序。三种子主候选还需均值条件保持且按场景bootstrap的AbsRel差异上界小于0。区间是探索性，未作多重比较校正。

仅M2主要晋级时尝试一次M3初始/中间/后期，投影6/5/5通道；仅边缘和多阶段各自主要晋级时尝试一次组合。额外条件候选至多两个。进入复验的多分支做同结构置零分支对照；若去RGB胜出，已有完整输入是其对照。无提升保留负结果，不扩展架构/损失/参数搜索。

最终冻结方法、种子、每种子开发最优轮数中位数与80轮余弦周期。仅已确认候选需要新全795重训/654测试；旧S0的三种子最终结果复用。有无收益都给出逐阶段表格、定性失败与时延成本。

## 记录与存储

每次运行保留配置、数据/源码哈希、完整history、最佳和最终权重、AMP事件、逐图指标、定性预览。失败记录保留。结束后仅清理已无依赖的可重建特征缓存和冗余resume；先校验保留权重并记录删除清单。配置/曲线/逐图指标/报告推送Git，权重和大型数组本地保留。此前README和stereo文档的用户改动不混入提交。

## 方法与重跑入口

修正输入网格为240×320。RGB缩放到[-1,1]；深度输入为log(D0)/2。初始/中间/后期特征分别来自官方`infer()['depth_features']`、`pixel_decoder.depth_layer.ups[-2]`、`ups[-1]`，是decoder不同阶段，并非另换ViT主干。各分支在原生网格做可训练1×1投影，再双线性对齐并拼接。固定CNN包含3个32通道3×3卷积+GELU和一个零初始化输出卷积；预测单通道残差，恢复480×640后加到D0，最低截断到0.001m。UniDepthV2始终冻结。Sobel输入只从RGB和预测深度计算，归一化复用旧fitting侧校准，不输入GT边缘。

在仓库根目录、已配置CUDA环境和上一轮数据缓存的条件下：

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/prepare_input_study.py
& $py -X utf8 scripts/part3_improvement/run_input_study.py
& $py -X utf8 scripts/part3_improvement/analyze_input_study.py --section screen --select
```

只有满足上面的预定门槛，才运行条件候选和确认种子。精确入选名单以`screen_selection.json`及后续决策记录为准，不能直接把所有组合都运行一遍后再挑test表现。

```powershell
# NAME替换为决策记录中最多两个方法；已有320112结果不重复训练。
& $py -X utf8 scripts/part3_improvement/run_input_study.py --names NAME --seeds 320113 320114 --phase confirm
& $py -X utf8 scripts/part3_improvement/analyze_input_study.py --section development --decide-confirmation NAME
```

若`confirmation_decision.json`没有保留方法则停止扩展并报告负结果。有保留方法时，用`finalize_input_study.py --action freeze --names NAME`冻结，再依次`--action refit`、`prepare_input_study.py --split test`、`finalize_input_study.py --action evaluate`、`analyze_input_test.py`。冻结后不修改数值实现或配置。

`render_input_study_cases.py --run <运行目录>`生成同样本S0/新候选对照：三个固定样本，以及按逐图AbsRel选出的最大改善、变化中位数、最大退步样本。`benchmark_input_study.py --checkpoints <S0权重> <候选权重> --tag <新标签>`测完整RGB到CPU深度链路；应在训练停止时单独运行。三种子表格汇总的是分别训练模型的指标均值，不是预测ensemble。

输出位于`outputs/part3_improvement/residual_inputs_20261119/`：`jobs/`是不可变运行配置，`followups/`保留各次训练，`stage_cache/`及`derived_inputs/`是可重建缓存，`analysis/`为定量和定性证据，`final_test/`仅在确认通过后出现。已有完整结果会被校验后复用；中断训练用`--resume`，不能覆盖记录。
