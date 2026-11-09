# Project 3 sample report — 2026-11-09

本目录是重新运行和独立分析后的英文 sample report，供两位成员验收。使用课程 Spring 2027 新版要求；正文 8 页，参考文献 1 页。旧报告 `../paper/` 保留为历史版本。

- `main.tex`：报告正文，按 Stereo / DA、metric depth / VDA、融合与控制实验、应用组织。
- `images/`：8 张可追溯图；流程图直接写在 LaTeX 中。
- `data/`：从当前分析复制的紧凑统计与最终 review 记录。review 中的证据位置相对于原分析目录；不是本目录下的图片路径。
- `asset_provenance.json`：原始证据与报告图片的 SHA256。
- `source_audit.md`：课程、模板、文献、结果来源及完成边界。
- `report_provenance.json`：最后的源码、表格、PDF、字体、演示包验证记录，由核验脚本生成。

最终 PDF 与演示包位于仓库根目录的 `outputs/submission_20261109/`。它们不进入 Git。完整中文分析与大图保留在对应实验的 `analysis/` 目录；统一入口为 `outputs/project_analysis_20261109/index.md`。个人验收指南也只保留本机。

## 编译

此目录可独立编译；不需要原始数据或模型：

~~~powershell
latexmk -pdf main.tex
~~~

也可在本目录使用 Tectonic：

~~~powershell
tectonic -X compile --untrusted --keep-logs --keep-intermediates --outdir ../../outputs/submission_20261109/report_build main.tex
~~~

本机编译使用已缓存的官方模板依赖，加了 `--only-cached`。不得用旧 PDF 改名代替重新编译。

修改报告后应重新编译，并逐页检查图表、引用和页数。报告专属验证工具属于生成时流程，可从 Git 版本 `d55270f` 追溯；实验统计与图表使用 [分析流程](../../docs/analysis_workflow.md) 中的入口。

## 结论边界

报告保留负面结果：LR 提高条件准确度但降低覆盖率；时间变化小不代表深度准确；完整置信度和三个 alpha 的视觉差异细微；应用改善主要来自阈值。新增对照为事后分析，既有主配置保持不变。70 帧与任务文字 100 帧的差异已披露。报告未宣称两位成员已经独立验收、完成课程提交或发布 arXiv。

provenance 保留生成时的工具和 README 哈希，属于历史记录；不代表当前说明文档仍具有相同字节。此次工具清理没有修改实验数值或 PDF。
