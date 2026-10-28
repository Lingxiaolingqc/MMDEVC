# 报告工作区

`paper/main.tex` 是英文工作草稿，使用固定版本的官方 CVPR 模板、camera-ready 样式及页码。已经写入真实 StereoSGBM、NYUv2 全量结果、融合方法和应用验证；第一页面明确标注自录视频实验待完成。来源与剩余工作见 [source_audit.md](source_audit.md)。

在 Codex 中打开 `paper/` 可编辑正文并预览 PDF。也可在已有 LaTeX 环境中进入该目录执行 `latexmk -pdf main.tex`；本文不需要 shell escape。主文件包含 `sections/`、`images/`、参考文献和保留原注释的模板文件。编译和页数核验结果记录在 `paper_provenance.json`；正式稿要求仍为 6–8 页正文、参考文献另计。

2026-10-28 草稿已编译为 7 页正文和 1 页参考文献，8 页均已渲染检查。14 条参考文献均有正文引用，没有未解析引用或超宽排版；PDF 字体均已嵌入。页数合格不代表实验已完成，仍须补入自录视频的正式结果后重新核验。

| 部分 | 主责 | 应提供的证据 |
| --- | --- | --- |
| Abstract / Introduction | A/B | 任务、方法、实际结果与仓库链接 |
| Related Work | A/B | 课程阅读列表与方法定位，完整引用 |
| Geometry Baseline | A | 标定、SGBM、误差和失败案例 |
| Metric Depth / Proposed Method | A | 模型配置、流程、公式、限制 |
| Experimental Setup / Results | B；A 复核 | 数据划分、协议、主表、时序与效率、消融 |
| Application / Failure Cases | B；A 复核 | Demo、成功与失败片段索引 |
| Conclusion | A/B | 有证据支持的结论和适用边界 |

每周将已验证结果写入报告；Week 10 做整合与交叉复现。无 GT 视频的稳定性改善不等于绝对准确率提升。课程 PDF 留在课程资料目录；模板文件保留来源和版权说明。本地导出的草稿 PDF 和编译日志在 `outputs/report_draft_20261028/`，不随 Git 上传。
