# 项目执行与验收证据

## 当前完成状态（2026-11-02）

依据收到的新版说明执行，[完整协议和复现](course_protocol_20261102.md)。原 A/B 主责保留，助手完成双方技术实施；人工验收单列。

- 官方 rollercoaster 完整 70 帧/24 fps，用户确认全段使用；与文字要求 100 帧的差异已写入报告及输入审计。
- Middlebury 15 对的正有限视差 GT、原始预测 MAE/有效比例已复核；保存结果未改写。
- UniDepth 654 张标准 test 已按新版 crop/GT range/pred clip 复算：AbsRel 0.1107756635、RMSE 0.3928173152 m、δ1 0.8947810718。
- DA/VDA relative 在独立 Tokyo dev 100 帧和官方 test 70 帧均完成 FP16/364 推理；VDA 518 OOM 记录保留。
- E0–E5 三组开发候选全部复算，α=0.70 在测试前冻结；最终 69 对 flow/mask、E0–E4 数组/权重与 E0–E5 分数复算通过。
- 固定范围视频、70 帧背景虚化、三个实际困难案例完成；E4 test 变化指标降低 15.93%，加和估算约 3.10 fps，不宣称准确率或速度提升。
- 74 项自动检查均通过：73 项沙箱通过；本地 HTTP 项因网络限制在沙箱外连同 Demo 其余两项通过。打包改动后 4 项打包回归也通过；真实数据负例拒绝错误帧数、缺失模型和改动色标。
- CVPR PDF 为 8 页正文+1 页参考文献，9 页已渲染检查。depth_results.zip 为 80 payload/2 clips，SHA256 和 CRC 回读通过。
- 中文个人验收指南仅本机 outputs/personal_acceptance/ 保存。仍需两位成员验收、决定 main 合并并执行课程提交；arXiv 可选。

当前证据见 [结果索引](../results/course_20261102/README.md)、[报告](../report/README.md)、[打包](submission.md)。下方仅为旧版历史来源。

## 旧版已完成成果及历史范围（2026-10-28）

目标：依据 Spring 2027 Project 3 完成 Baseline、Reproduction、Improvement、应用及可提交材料。模块主责保留原 A/B 分工，由助手执行实现和实验；人工验收须由本人完成。代码分支 `feat/depth-video-pipeline`，起点 `c6c8314`。

## 已有可复用成果

- Middlebury 15 对文件与标定、StereoSGBM 固定配置、原始预测、七指标和视差诊断；见 `results/stereo_week1.md`。
- NYUv2 1449 对及标准 795/654 split；原始数组和源文件一致性已核验。
- UniDepthV2 ViT-S 的单样本 CPU 推理与独立复算；见 `results/unidepth_week1.md`。
- 统一指标与相对对齐协议；见 `docs/evaluation_protocol.md`。

## 必须完成的证据

| 范围 | 主责 | 当前状态 | 完成需要的证据 |
| --- | --- | --- | --- |
| DA V2 Small relative 单图 | B | 已完成 dev 与 654 test；含 1 个明确对齐失败 | 官方权重/源码固定；真实原始预测及独立评测 |
| Part 1 relative 视频诊断 | B | 推理、固定色标、原始相对单位的时序/中位数诊断已在示例上验证 | 自录静止与移动片段；见 `docs/relative_video.md`，不能将相对误差当米制成绩 |
| NYUv2 全标准测试集 | A/B | 三模型各 654 张推理、保存结果复算及边界评测通过 | 见 `results/nyuv2_standard_test.md` 与完整逐图表 |
| 视频素材与统一帧输入 | B | 等待采集 | 真实文件、scene group、预先固定的 dev/test、时间戳 |
| DA V2 metric / VDA metric | B | GPU 同一公开视频 24 帧检查通过；正式实验待自录素材 | 相同帧和分辨率，原始输出、固定色标、延迟/显存/模型大小 |
| 边界与时序评测 | A/B | NYUv2 边界全量完成；公共时序 mask 和固定分母复算通过 | 真实视频指标与困难场景结论待素材 |
| E0–E5 Improvement | A 实现/B 实验 | 六版本和原始数组复算通过；开发选择/冻结入口已实现；参数仍 provisional | 真实 dev 观察与选择、冻结配置、独立 test、消融及失败解释；见 `docs/fusion_selection.md` |
| 应用 Demo | B | 浏览器点选、切帧/方法、播放及 24 帧导出通过；正式素材待采集 | RGB、固定色标深度、点选 Z 深度和时间曲线；见 `results/demo_smoke.md` |
| CVPR 报告与提交包 | A/B | 已有 CVPR 工作草稿；打包入口和示例 ZIP 回读验证通过 | 自录结果、最终报告和正式 depth_results.zip 待补；见 `report/README.md`、`docs/submission.md` |
| 个人验收 | 本人 | 待最终生成 | 本地中文操作指南，含预期结果、通过标准和未完成项；不进 Git |

2026-10-28：已对照课程 PDF 完整六页。必需的边界质量、光流对齐时序指标、逐帧中位数变化、效率、困难场景案例均保留。arXiv 与 Canvas 最终发布由成员确认执行，不自动代发。

用户已确认暂无视频；采集操作见 `docs/video_capture.md`。公开视频只作为明确标记的接口检查，不代替自录测试。RTX 4060 Laptop 8188 MiB 的隔离 CUDA 环境已验证，视频默认 short side=364；详见 `environments/cuda.md`。代码与实验仍处于工作分支，尚未自动合并到 main。
