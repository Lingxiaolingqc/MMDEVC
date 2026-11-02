# 提交、打包与交叉验收

当前以 2026-11-02 收到的新版说明为准。必做视频为官方 rollercoaster，已经按用户确认采用完整 70 帧；不要求补拍自录视频。arXiv 是可选项。

## 交付物

1. 报告：CVPR camera-ready，正文 6–8 页，参考文献另计；两位作者、所有指定阅读、摘要末 GitHub 链接、实际结果/消融/困难案例。源码位于 report/paper/，本地 PDF 位于 outputs/submission_20261102/。
2. GitHub：完整项目代码、环境与复现说明、紧凑结果证据。当前工作分支 feat/depth-video-pipeline，最终合并 main 由成员完成。
3. 视频与应用：outputs/submission_20261102/depth_results.zip。包含 dev/test 比较、E0–E5、背景虚化、时间戳、冻结证据和三个困难案例，不包含原始代码或数据集。
4. 本人验收：outputs/personal_acceptance/ 下的中文指南，仅本机保留，不进 Git，也不进 ZIP。

## 打包与验证

在仓库根目录、项目 Python 环境中执行。已有目标 ZIP 时不要覆盖；使用新的 outputs 子目录。

~~~powershell
python scripts/package_depth_results.py --plan configs/course_export_plan.json --output outputs/submission_20261102/depth_results.zip --check-only
python scripts/package_depth_results.py --plan configs/course_export_plan.json --output outputs/submission_20261102/depth_results.zip
~~~

当前正式 scope 为 course_project_20261031。打包前检查：

- 必须覆盖全部已登记的开发和测试片段，场景/源哈希不得交叉。
- 固定官方 test 的完整帧数和源 SHA256 必须匹配帧数差异记录。
- 每组 E0–E5 都已独立复算，配置等于开发选定参数；test 绑定冻结凭据。
- 原始 DA/E0–E4 同色标，VDA 单独整段固定色标；不混入米制视频。
- 背景虚化为冻结 test 输出；应用值与原始预测复算通过、视频 PTS 一致。
- 三个案例必须指向真实登记片段、合法时间区间和具体观察。
- ZIP 创建后逐项回读核查 SHA256、CRC 和文件数量。

旧 self_recorded_project / pipeline_check_not_final_evaluation 模式仍用于历史记录和兼容性测试，不用于本次新版包。旧示例包不能改名冒充当前提交包。

## 成员最后确认

A 复核单位、GT 区域、新版 NYUv2 数值和融合公式；B 复核两个视频模型、固定范围、三个案例、效率边界和应用。双方各执行一次复算命令，并检查 PDF 与 ZIP 能在另一台电脑打开。助手执行不等于成员已验收。

按课程当前渠道提交 PDF、公开仓库链接和 ZIP。没有替成员合并 main、上传 Canvas 或发布 arXiv，也没有可填写的 arXiv ID。不要把个人验收笔记或模型权重加入提交包。
