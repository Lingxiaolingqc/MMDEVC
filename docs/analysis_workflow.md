# 按实验保存分析证据

2026-11-09 执行约定：助手运行剩余实验、独立完成全部定量/定性分析并生成 sample report。成员可以复现命令和补充观察，但成员观察不代替助手对当前源数据的复核。此前手动运行指南中的命令仍可用于复现。

## 组织方式

新增分析保存在对应实验的 analysis/ 下：
- analysis.md：给人阅读的中文问题、设置、数值、视觉发现、结论及局限。
- review.json：助手实际检查范围、证据哈希、结论与不确定性。
- quantitative_review.json：程序计算和来源验证记录；程序不自动宣称完成视觉验收。
- summary.csv、per_pair.csv 等：可复用统计。
- figures/：报告统计图、全景和连续局部证据。

已有 review.json 等记录保持原位置。outputs 被 Git 忽略；完整分析的本地总导航位于 outputs/project_analysis_20261109/index.md。最终报告将择取这些证据，单独提供 CVPR LaTeX 源码和 PDF；个人验收指南只保存在本机。

## 置信度去除结果分析

先完成补充实验及其 verify_supplementary_ablations 复算，再运行分析。默认新建对应实验下的 analysis/，存在则拒绝覆盖。改变 ROI 或重新检查时使用新的 --output-dir。

~~~powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/part3_improvement/analyze_confidence.py --run outputs/part3_improvement/supplementary/dev/dev_tokyo_100/confidence --roi 200 60 390 350
& $py -X utf8 scripts/part3_improvement/analyze_confidence.py --run outputs/part3_improvement/supplementary/test/course_rollercoaster_70/confidence --roi 100 60 450 260
~~~

该脚本读取原始保存预测，不重跑模型或融合。它核对源哈希及复算凭据，从七组数组重算 T、raw L1、coverage、平均历史权重和对 DA 的偏离，并检查四个对照视频的时间戳和固定显示范围。它导出统计图和接触图，但视觉结论必须在实际检查后单独写入 review.json / analysis.md。

全景取样覆盖整段，局部连续窗口定位在与完整 E4 的数值差异较大处；这是一种有说明的诊断选择，不是随机抽样。图中每行记录帧号、时间和方法，同一原始坐标 ROI 应用于所有方法。描述时区分抽样图、连续帧图和完整实时播放。

T 是固定 RGB/flow 公共区域上的预测变化，不是 GT 深度准确率。偏离 DA 也不是误差，因为 DA 不是 GT。逐帧对箱型图用于描述分布，主表是按参考像素加权的汇总；相关帧对不能视作独立实验样本。方法实际融合时间和原推理时间属于加和估计，不能充当同轮端到端 benchmark。

## 报告使用规则

每条主要结论对应当前文件、明确设置、统计和具体图片/帧。既有用户初步观察单列，所有步骤均重新独立核对，包括原本已有简单分析的部分。显示无改善或差别细微的实验也保留。

当前路线中的 LR、历史来源、置信度和应用阈值做控制变量比较；streaming、metric-video 等其他可选路线不会自动扩展为必做实验。冻结主结果不因事后分析改写。
