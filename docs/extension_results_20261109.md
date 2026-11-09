# Streaming 与一次置信度改进：完成记录

2026-11-09。助手直接运行实验并独立分析。主方法仍为原 E4；此次没有训练模型，也没有增加 metric-video。

## Part 2：官方 offline / streaming

相同 VDA Small 相对深度权重、输入尺寸 364、FP16、RTX 4060 Laptop GPU，原始缓存保持 100/70 帧。每种模式每个片段运行三次，清空 streaming 历史并重放前缀检查。四组重复输出完全一致；新 offline 与原 offline 数组完全一致。

|片段|模式|总时间中位数 s|首次 API 返回 ms|后续逐帧中位/P95 ms|峰值 allocated MiB|T|
|---|---|---:|---:|---:|---:|---:|
|Tokyo|offline|9.585|9584.7|不适用|2386.4|0.002941|
|Tokyo|streaming|5.734|69.1|56.47 / 60.21|685.3|0.003580|
|过山车|offline|5.929|5928.5|不适用|2386.4|0.011590|
|过山车|streaming|4.408|91.5|60.48 / 70.62|682.8|0.012769|

Streaming 的 allocated 峰值降低约 71%，但启动时的颜色变化更明显。主要结构相近，细节合并仍在。后续逐帧延迟未达到过山车 24 fps 所需的 41.67 ms，因此不宣称实时。T 分别升高 21.71% / 10.17%，它是无 GT 的诊断量，不是深度准确率。

Offline 官方接口整段返回；“首次返回”不是其固有首帧运算延迟。输入已在内存，计时不含加载、解码、文件读写和可视化。三次重复按模式分块，未随机化热状态。两种模式使用各自全片固定色标；可视化属于离线审阅，不能当成在线归一化算法。

主要代码：

- `src/p3_depth/models/streaming.py`：封装固定版本官方 streaming 实现、同权重加载和状态重置。
- `scripts/part2_models/run_vda_mode.py`：保存数组、三次计时、逐帧延迟和视频。
- `scripts/part2_models/analyze_vda_modes.py`：核验输入/来源/时间戳、重算指标并生成匹配图。

结果在 `outputs/part2_models/vda_modes/{dev,test}/{clip}/`。`analysis/review.json` 是完成后的独立定性记录；`quantitative_review.json` 保留定量阶段状态，不能单凭其 `visual_pending` 字段判断最终分析未完成。

## Part 3：有边界的一次改进

先对原 E4 分组诊断，再固定一套候选：深度 sigma 从 0.1 放宽到 0.2。比较 original、global、selective。Selective 仅作用于 photo≤0.05、FB≤0.5 px 且不在深度边缘扩张带内的区域。其余 alpha、硬筛选和置信度项相同；掩码由当前原始 DA 与 RGB 光流决定。

|片段|方法|T|CPU 融合中位秒|
|---|---|---:|---:|
|Tokyo|original|0.003439|3.412|
|Tokyo|global|0.003350|3.001|
|Tokyo|selective|0.003372|7.630|
|过山车|original|0.012364|2.166|
|过山车|global|0.011767|2.504|
|过山车|selective|0.012204|5.877|

每组做三次旋转顺序 CPU 计时，输出逐次完全一致；数组、权重、指标和视频时间戳都已复算。Original 与此前 E4 完全一致。Selective 当前计算两套融合和保护掩码，其额外成本是当前实现成本。

Tokyo 的可靠内部深度指数均值仅 0.059，原假设支持较弱。按照预先限定的一次对照完成尝试后，T 仅小幅变化，未见明确视觉收益。因此先冻结保留原 E4 的决定，再使用同一配置运行过山车。过山车亮暗亮变化和小结构合并仍在。没有继续调参。

过山车的可靠区域很多是零值天空；正值内部子集的深度冲突更强，但正值不等于语义前景。分组统计防止全图均值掩盖局部问题。Selective 在保护带外保留原公式，但历史由各自结果累积，因此序列输出不保证逐像素等于原 E4。

主要代码：

- `src/p3_depth/temporal/refinement.py`：新候选；原 `fusion.py` 保持不变。
- `scripts/part3_improvement/diagnose_confidence_regions.py`：先验区域诊断。
- `scripts/part3_improvement/run_confidence_refinement.py`：三组候选、旋转计时、数组/视频/指标和复算。
- `configs/fusion_refinement_20261109.json`：一次候选的设定与停止规则。

结果在 `outputs/part3_improvement/confidence_refinement/{dev,test}/{clip}/`。`diagnosis/` 保存分组结果，`analysis/` 保存独立说明和 review；dev 的 `decision.json` 记录在课程补充运行之前作出的选择。过山车已经看过，不能把它称为新的盲测。

## 查阅与验证

无需重新运行模型。已有结果可复算：

```powershell
$py = '.\.venv-cuda\Scripts\python.exe'
& $py -X utf8 scripts/part3_improvement/run_confidence_refinement.py --clip dev_tokyo_100 --verify-only
& $py -X utf8 scripts/part3_improvement/run_confidence_refinement.py --clip course_rollercoaster_70 --verify-only
```

生成命令会拒绝覆盖同名完成结果。完整基础流程仍见 `docs/rerun_by_part.md`；本扩展依赖其已验证的模型、视频缓存、DA、RGB 光流和原 E0–E5 实验。

新报告源文件在 `report/sample_20261109_extension/`，包含原有 12 份分析及新增 4 份 review 的紧凑证据。正文 8 页、参考文献 1 页。原报告和原提交包保留；新的 PDF、扩展视频包和个人验收指南均在被 Git 忽略的 outputs 内。个人指南不进入提交包，也不 commit/push。

本次通过 4 个新候选测试、4 个相对时序测试、9 个已有补充实验测试，以及实际结果复算。当前实验复算仍使用上面的 `--verify-only` 命令。生成时的报告检查属于历史制作流程，工具可在 Git 版本 d55270f 中追溯，不是重跑实验的依赖。

另一位组员 pull 可得到代码、报告源文件、图片和紧凑统计；数据、权重和大视频需要另行同步。
