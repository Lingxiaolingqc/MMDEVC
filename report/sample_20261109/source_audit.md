# 来源和完成边界

核对日期：2026-11-09。

## 课程和模板

依据 Projects 目录的 `Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf`，SHA256 为 `13036feb075b86b1b69b5a2ad5c5f6eeb535e8bb804efad8c9fc1307651d2eef`。该文件规定新版 Spring 2027 路线；报告选一项视频融合改进和背景虚化应用。

模板与参考文献样式直接复制仓库已核实的官方 CVPR author-kit 文件，保持版权和文件内容不变。版本、上游 blob 见 [原模板审计](../source_audit.md#模板)；新版本哈希另见本目录 provenance。模板 CVPR 2026 标签不等于课程学期。

## 文献

正文引用课程十篇指定阅读：SGM、KITTI、DPT、Depth Anything、Depth Anything V2、UniDepth、Depth Pro、UniDepthV2、RAFT、Video Depth Anything。另引用 NYUv2、Middlebury、OpenCV 和 Farneback，共 14 条。文献是方法背景和所借用方法的来源，不代表学生已逐篇精读，也不把论文成绩当成本项目实测值。

本轮核对作者论文或官方来源：

- [SGM](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Hirschmueller2008PAMI.pdf)、[KITTI](https://www.cvlibs.net/publications/Geiger2012CVPR.pdf)、[DPT](https://arxiv.org/abs/2103.13413)
- [Depth Anything](https://arxiv.org/abs/2401.10891)、[Depth Anything V2](https://arxiv.org/abs/2406.09414)
- [UniDepth](https://arxiv.org/abs/2403.18913)、[UniDepthV2](https://arxiv.org/abs/2502.20110)、[Depth Pro](https://arxiv.org/abs/2410.02073)
- [RAFT](https://arxiv.org/abs/2003.12039)、[Video Depth Anything](https://arxiv.org/abs/2501.12375)
- [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)、[Middlebury benchmark](https://vision.middlebury.edu/stereo/submit3/)、[Middlebury 标定](https://vision.middlebury.edu/stereo/data/scenes2014/)
- [OpenCV StereoSGBM](https://docs.opencv.org/4.13.0/d2/d85/classcv_1_1StereoSGBM.html)、[Farneback 出版社摘要](https://link.springer.com/chapter/10.1007/3-540-45103-X_50)

## 当前结果与证据

本目录 data 的文件名前缀对应原分析：

| 前缀 | 原分析目录（相对仓库根） |
| --- | --- |
| stereo | outputs/part1_baseline/stereo/all/analysis |
| da | outputs/part1_baseline/dav2_relative/test/course_rollercoaster_70/analysis |
| nyu | outputs/part2_models/unidepth/test/analysis |
| vda | outputs/part2_models/vda_relative/test/course_rollercoaster_70/analysis |
| main_dev | outputs/part3_improvement/dev_candidates/alpha070/dev_tokyo_100/analysis |
| main_course | outputs/part3_improvement/test/course_rollercoaster_70/analysis |
| history_dev / history_course | outputs/part3_improvement/supplementary/{dev/dev_tokyo_100 或 test/course_rollercoaster_70}/history/analysis |
| confidence_dev / confidence_course | 同上，将 history 换为 confidence |
| candidates | outputs/part3_improvement/selection/analysis |
| application | outputs/application_exploratory/course_rollercoaster_70/transition010_030/analysis |

每项最终 `review.json` 记录独立视觉检查范围、结论、局限和源哈希。`quantitative_review.json` 是先完成数值阶段时的记录，里面可能保留 `visual_pending`；是否完成视觉检查以较后的最终 review 为准。用户的初步观察没有被直接当作最终结论。

图由 `scripts/build_sample_report_assets.py` 读取保存的数组或已验证的统计生成。按全段固定范围显示，不逐帧拉伸色标；图表抽样只用于展示，不替代全量统计。NYU 0088 与 0033 分别使用 0–10 m 和 0–2.5 m 的显式行内共享范围。

本次困难片段表位于 `results/sample_20261109/video_failure_cases.csv`。报告还给出 Jadeplant、NYU 0088 和 0033 三类静态困难案例。新演示包使用 `configs/course_export_plan_sample_20261109.json`，其中应用采用事后新阈值，元数据保留实际阈值和方法来源。

## 解释与剩余成员行动

- 15 场景 Stereo 汇总含两个开发场景；另列 13 个评价场景，不能冒充官方隐藏测试。
- NYU 654 test 使用课程裁剪和预测 clipping，不做测试 GT 尺度拟合。
- 官方视频实际 70 帧/24 fps，按用户决定使用完整视频并披露 100 帧差异；这不是任课教师确认。
- 东京仅开发，过山车为课程指定展示片段。后续补充实验与应用阈值明确事后性质，不能包装为新盲测。
- VDA streaming、metric-video、额外训练不是本报告实验。
- 视频没有深度 GT；时序变化、偏离 DA、虚化强度不是准确率或画质指标。
- 视觉核验使用定位到帧号的总览、连续帧和局部图；没有声称完整实时观看全部视频。
- 成员仍需亲自阅读、验收贡献表述和最终版本，再决定合并 main 与课程提交。助手没有代签验收、上传 Canvas 或发布 arXiv。
