# 报告来源与完成边界

核对日期：2026-10-28。`paper/` 是英文工作草稿，保留原两人署名；助手编写与运行不代表组员已经独立验收。第一页面向读者明确标注未完成自录视频实验，不是可直接提交或公开发布的最终稿。

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

## 结果来源

- Stereo 数字：`results/tables/stereo_week1_summary.csv`、`stereo_week1_per_scene.csv`；原始数组已复算。13 场景 evaluation 与包含 dev 的 15 场景统计明确分开。
- NYUv2 主表、边界、效率：`results/nyuv2_test_20261028/summary.csv`；三模型各 654 个 test ID。relative 条件误差贡献 653 图，1 图对齐失败仍保留固定分母。
- 图：复制已检查的 Jadeplant 与 NYUv2 比较图，不另行归一化或修改数据。Demo 预览保留在应用文档中，报告描述其数值与操作验证。
- 视频示例数值：`results/fusion_smoke_20261028.csv` 与 provenance；只用于程序验证。自录 dev/test 尚未采集，不能据此宣布 Improvement 成功。
- 应用：`results/demo_smoke.md`；数值/时间戳复算和浏览器操作检查已完成，成员验收待完成。

## 最终稿前必须补齐

1. 真实自录 dev/test 清单、开发选择证据和冻结配置。
2. E0–E5 正式时序与效率表、固定场景波动、相机运动响应、遮挡/细边缘拖影分析，以及困难条件的片段索引。
3. 正式自录 Demo 与所有规定处理视频，生成并检查 `depth_results.zip`。
4. 根据真实结果重写摘要、时序结果和结论，去除工作草稿声明后重新核对 6–8 页正文和参考文献。
5. 完成成员验收。arXiv 和 Canvas 发布由成员确认执行；没有已生成的 arXiv ID。
