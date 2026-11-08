# 提交、打包与交叉验收

截至 2026-11-09，本轮实验、逐项分析和 sample report 已整理完成，待两位成员验收。采用 Spring 2027 新版说明，官方 rollercoaster 的完整 70 帧按用户决定使用，并披露与任务文字 100 帧的差异。不要求补拍自录视频，arXiv 为可选项。

## 当前交付

1. CVPR sample report：`report/sample_20261109/` 是源码，`outputs/submission_20261109/MMDEVC_Project3_Sample_Report.pdf` 是本机 PDF。8 页正文、1 页参考文献、14 条引用，包括指定阅读、方法、主结果、控制实验、应用和真实困难案例。
2. 视频包：`outputs/submission_20261109/depth_results.zip`，80 个载荷文件，另有包内说明和 manifest。包含两个片段的主比较、E0–E5 对照和新阈值背景虚化；不包含代码、数据集、权重或个人验收笔记。
3. 分析：各实验 `analysis/` 下的 `analysis.md`、最终 `review.json`、表和图；入口为 `outputs/project_analysis_20261109/index.md`。报告 data 随 Git 保存紧凑证据。
4. 本人验收：`outputs/personal_acceptance/project3_sample_report_20261109.md`，仅本机，不 commit/push。

历史 `outputs/submission_20261102/` 保留。2026-11-07 后的 [按 Part 重跑指南](rerun_by_part.md) 继续作为复现流程；本轮实际交付有独立日期目录，避免旧新结果混用。

## 打包和校验

新计划为 `configs/course_export_plan_sample_20261109.json`。在仓库根目录执行：

~~~powershell
python scripts/package_depth_results.py --plan configs/course_export_plan_sample_20261109.json --output outputs/submission_20261109/depth_results.zip --check-only
~~~

这是只读预检。若需要重新创建，改用尚不存在的输出目录：

~~~powershell
python scripts/package_depth_results.py --plan configs/course_export_plan_sample_20261109.json --output outputs/submission_recheck/depth_results.zip
~~~

预检要求登记片段与源哈希一致、全部主结果复算通过、既有冻结配置匹配、DA/E0–E4 共用固定色标、VDA 使用独立的全段固定色标、应用从原始数组复算通过且时间戳一致、困难案例指向合法区间。创建后逐项回读 CRC 和 SHA256。

这些工程检查不证明测试独立性或画质提升：补充实验和应用阈值是事后分析，报告已有说明。新包采用相同冻结 E4 深度与新的应用阈值，不能把阈值改善归于 E4。

## 成员最终确认

A 复核单位、GT 区域、NYU 协议与结果；B 复核 DA/VDA、视频对照、效率和应用。双方阅读贡献与助手协助表述，检查 PDF 和 ZIP 可在另一台电脑打开，并按本地指南完成必要复算。

工作分支为 `feat/depth-video-pipeline`。最终合并 main、按课程渠道提交 PDF/公开仓库链接/ZIP 由成员决定。助手没有上传 Canvas 或发布 arXiv。个人验收文件不加入任何提交包。
