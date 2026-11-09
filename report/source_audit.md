# 报告来源与完成边界

> 历史来源记录：下面的报告工具和打包配置路径按 Git 版本 `d55270f` 追溯，已移出当前实验代码树。复现实验使用 [分析流程](../docs/analysis_workflow.md)。

核对日期 2026-11-02。当前 paper/ 对应新版要求和真实实验，已编译为 8 页正文+1 页参考文献。名义作者与模块主责保留；助手实现与复算不代表成员独立验收。报告注明 70/100 帧差异、无视频 GT、单场景限制及计算代价。

## 模板

使用 [CVPR 官方 author-kit](https://github.com/cvpr-org/author-kit/tree/CVPR2026-v1(latex)) 的 `CVPR2026-v1(latex)` 标签，按课程要求选 camera-ready 样式并保留页码（`pagenumbers`），未启用 review。模板年份与课程 Spring 2027 版本分别记录，不能把二者混为同一版本。`cvpr.sty` 与 `ieeenat_fullname.bst` 保留原注释和版权说明，只增加末尾换行。

上游 Git blob：`cvpr.sty` 为 `6ce0bd812c7528ac2eb0859a4584e96049543ec4`；参考文献样式为 `261b8c3ccb3fd5c07a88ffc4945c40909bdcf535`。本地文件 SHA256 见 `paper_provenance.json`。

## 文献核对

下列十篇覆盖课程 Recommended Reading List。引用用于说明方法定位，不代表两位成员已逐篇精读；正文没有复制论文实验成绩作为本项目结果。

| 文献 | 正文用途 | 核对来源 |
| --- | --- | --- |
| SGM, TPAMI 2008 | 路径代价聚合与双目匹配；区分原算法和 OpenCV 变体 | [原文](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Hirschmueller2008PAMI.pdf) |
| KITTI, CVPR 2012 | 场景分布和几何基准的背景；本项目未运行 KITTI | [作者原文](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf) |
| DPT, ICCV 2021 | Transformer 特征恢复为密集预测 | [CVF](https://openaccess.thecvf.com/content/ICCV2021/html/Ranftl_Vision_Transformers_for_Dense_Prediction_ICCV_2021_paper.html) |
| Depth Anything, CVPR 2024 | 大规模无标注数据监督 | [作者稿](https://arxiv.org/abs/2401.10891) |
| Depth Anything V2, NeurIPS 2024 | 合成数据、teacher/student、relative/metric 区分 | [作者稿](https://arxiv.org/abs/2406.09414) |
| UniDepth, CVPR 2024 | 相机表示与 metric 3D 预测 | [作者页面](https://lpiccinelli-eth.github.io/pub/unidepth/) |
| Depth Pro, ICLR 2025 | 多尺度、边界与焦距；仅综述，未运行 | [作者机构页面](https://machinelearning.apple.com/research/depth-pro) |
| UniDepthV2, arXiv 2502.20110 | 本项目 ViT-S 模型；引用课程指定 arXiv 文献 | [作者稿](https://arxiv.org/abs/2502.20110) |
| RAFT, ECCV 2020 | 学习式光流作为相关工作；实际选 Farneback | [作者稿](https://arxiv.org/abs/2003.12039) |
| Video Depth Anything, CVPR 2025 | 视频 temporal head 与独立 E5 参照 | [作者稿](https://arxiv.org/abs/2501.12375) |

补充引用：[NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[Middlebury v3](https://vision.middlebury.edu/stereo/submit3/)、[Middlebury 标定单位与转换](https://vision.middlebury.edu/stereo/data/scenes2014/)、[Farneback 作者机构页面](https://www.ida.liu.se/ext/WITAS-ev/Computer_Vision_Technologies/PaperInfo/farneback03.html)、[OpenCV StereoSGBM](https://docs.opencv.org/4.13.0/d2/d85/classcv_1_1StereoSGBM.html)。官方模型仓库链接随文献与环境说明保留。

## 当前结果来源

- Stereo：results/tables/stereo_week1_summary.csv；新版正视差 GT 审计见 results/course_20261102/stereo/。
- NYUv2：results/course_20261102/nyuv2/；复用 654 张原始 UniDepth 输出，采用新版 clipping。效率来自原始推理，不把复算时间当推理时间。
- 视频：results/course_20261102/video/；独立开发选择见 selection/。E0–E4 共用原 DA 范围，E5 使用 VDA 自身范围。
- 图片：scripts/build_course_figures.py 从保存数组生成，源/帧/裁剪见 results/course_20261102/figure_provenance.json。Jadeplant 图保留原始哈希。
- 三个案例：results/course_20261102/failure_cases.csv；包含 1 个 dev 和 2 个 test 条件，不伪造独立测试场景。
- ZIP：outputs/submission_20261102/depth_results.zip，80 payload，CRC/逐项 SHA256 回读检查。
- PDF：outputs/submission_20261102/MMDEVC_Project3_Report.pdf；源码/结果/日志哈希和页数见 paper_provenance.json。

## 待成员完成

阅读并验收报告结论，复算或跨机复现，确认作者贡献及最终版本，合并 main 并提交课程材料。没有代替成员提交 Canvas 或发布 arXiv；arXiv 不是新版必做项。旧 2026-10-28 草稿与原始实验仍保留在历史输出目录。
