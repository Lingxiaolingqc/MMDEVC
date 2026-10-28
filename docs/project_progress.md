# 项目执行与验收证据

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
| NYUv2 全标准测试集 | A/B | 三模型各 654 张推理、保存结果复算及边界评测通过 | 见 `results/nyuv2_standard_test.md` 与完整逐图表 |
| 视频素材与统一帧输入 | B | 等待采集 | 真实文件、scene group、预先固定的 dev/test、时间戳 |
| DA V2 metric / VDA metric | B | GPU 同一公开视频 24 帧检查通过；正式实验待自录素材 | 相同帧和分辨率，原始输出、固定色标、延迟/显存/模型大小 |
| 边界与时序评测 | A/B | NYUv2 边界全量完成；公共时序 mask 和固定分母复算通过 | 真实视频指标与困难场景结论待素材 |
| E0–E5 Improvement | A 实现/B 实验 | 原型、六版本比较、原始融合数组复算通过；参数 provisional | dev 调参记录、冻结配置、独立 test、消融及失败解释 |
| 应用 Demo | B | 浏览器点选、切帧/方法、播放及 24 帧导出通过；正式素材待采集 | RGB、固定色标深度、点选 Z 深度和时间曲线；见 `results/demo_smoke.md` |
| CVPR 报告与提交包 | A/B | 待整理 | 6–8 页正文、必读引用、实际图表、可复现 README、depth_results.zip |
| 个人验收 | 本人 | 待最终生成 | 本地中文操作指南，含预期结果、通过标准和未完成项；不进 Git |

2026-10-28：已对照课程 PDF 完整六页。必需的边界质量、光流对齐时序指标、逐帧中位数变化、效率、困难场景案例均保留。arXiv 与 Canvas 最终发布由成员确认执行，不自动代发。

用户已确认暂无视频；采集操作见 `docs/video_capture.md`。公开视频只作为明确标记的接口检查，不代替自录测试。RTX 4060 Laptop 8188 MiB 的隔离 CUDA 环境已验证，视频默认 short side=364；详见 `environments/cuda.md`。代码与实验仍处于工作分支，尚未自动合并到 main。
