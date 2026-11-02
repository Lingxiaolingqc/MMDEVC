# 报告工作区

paper/main.tex 是新版 Spring 2027 Project 3 的完整英文报告源码，使用固定版本官方 CVPR camera-ready 样式及页码。当前 PDF 为 **8 页正文 + 1 页参考文献**；9 页已渲染检查，没有超宽公式/文本或未解析引用，14 条文献全部引用。

本地交付：outputs/submission_20261102/MMDEVC_Project3_Report.pdf。源码随 Git，同目录 images/ 提供论文图，references.bib 覆盖全部指定阅读。模板、结果来源和完成边界见 [source_audit.md](source_audit.md)，精确文件哈希及编译验证见 [paper_provenance.json](paper_provenance.json)。

在 Codex 打开 paper/ 可编译预览；已有 LaTeX 环境可进入 paper/ 执行 latexmk -pdf main.tex，无需 shell escape。本次使用 Tectonic 编译并保留日志、PDF 原件和各页检查图。旧草稿留在 outputs/report_draft_20261028/。

报告包含 Stereo 15 对、UniDepth 654 test、新版 crop/clipping、独立开发后冻结的 relative E0–E5、效率代价、三个实际困难案例和背景虚化应用。官方 70 帧与任务文字 100 帧的差异已明确说明。

作者仍为 Qinchuan XIAO / Jun Yu Eugene YIP；技术实施由助手协助，成员独立验收未代签。当前材料可供成员验收，未上传 Canvas 或发布 arXiv。后续若改源码或结果，必须重新编译并核查 6–8 页正文和图片/表格。
