# MMDEVC

**Monocular Metric Depth Estimation & Video Consistency**
AIAA3201 Project 3 · Spring 2027 · 按 2026-11-02 收到的新版任务说明执行。

## 当前成果

| 部分 | 实际完成内容 |
| --- | --- |
| StereoSGBM | Middlebury trainQ 全 15 对；有效预测视差 MAE 1.3113 px（场景平均），有效比例 62.65%（GT 像素加权） |
| Metric depth | UniDepthV2 ViT-S，标准 NYUv2 654 test；AbsRel 0.110776、RMSE 0.392817 m、δ1 0.894781 |
| Relative video | DA V2-Small 与 VDA-Small；相同连续帧、FP16、输入短边 364、固定整段色标 |
| ONE Improvement | Flow-Guided Fusion；独立 Tokyo dev 比较三组参数，冻结 α=0.70 后完成 E0–E5 测试 |
| 测试结果 | 官方 rollercoaster 全 70 帧；E4 归一化时序变化指标下降 15.93%，附计算开销与三个实际困难案例 |
| Application | 同参数比较原始 DA 与 E4 的背景虚化；导出全部 70 帧并核验时间戳 |
| 交付 | CVPR 报告源码、视频打包入口、结果与复现记录；成员个人验收及正式提交由成员完成 |

**官方源文件完整只有 70 帧、24 fps，新版文字要求前 100 帧。** 已按项目所有者确认使用全部 70 帧并记录差异，没有重复或插帧；这不是教师已确认的变更。证据见 [输入核查](results/course_input_audit_20261102.json)。

视频没有深度 GT；时序变化降低不能称为准确率提升。E4 的推理+CPU 光流+融合合计估算约 3.10 fps，比原始模型慢。旧版自录视频、米制视频、七指标及边界实验保留为历史/补充结果，不是新版额外必做项。

## 从哪里开始

1. [当前协议与完整复现顺序](docs/course_protocol_20261102.md)：新版要求、数据范围、命令、参数冻结与检查。
2. [结果与限制](results/course_20261102/README.md)：主表、候选、测试、困难案例和原始证据入口。
3. [提交与打包](docs/submission.md)：PDF、GitHub、depth_results.zip 和人工验收边界。
4. [报告工作区](report/README.md)：CVPR camera-ready 源码和编译说明。
5. [当前任务状态](docs/task_board.md)与[协作流程](CONTRIBUTING.md)。

## 团队与工作流

| 成员 | GitHub | 原模块主责 |
| --- | --- | --- |
| A：Qinchuan XIAO | [Lingxiaolingqc](https://github.com/Lingxiaolingqc) | 数据、StereoSGBM、UniDepth、单图评测、融合算法 |
| B：Jun Yu Eugene YIP | [Geney-gy-29](https://github.com/Geney-gy-29) | DA/VDA、视频、时序/效率评测、消融、应用 |

保留名义分工，助手实施双方技术工作；不将助手复算写成人工交叉验收。当前代码在 feat/depth-video-pipeline，main 合并和 Canvas 提交由成员完成。目标仍为 Week 11 之前交付，具体日历截止时间待课程确认。

## 安装与数据

~~~sh
git clone https://github.com/Lingxiaolingqc/MMDEVC.git
cd MMDEVC
git switch feat/depth-video-pipeline
python scripts/check_setup.py
~~~

本机配置从 configs/paths.example.yaml 复制到 configs/paths.local.yaml；已有配置不要覆盖。基础安装与数据检查：

~~~sh
python -m pip install -e ".[data,stereo,nyu,video]"
python -m unittest discover -s tests -v
~~~

基础依赖不包含所有模型运行环境；官方源码、固定权重与 CUDA 安装分别见 [模型环境](environments/README.md)、[CUDA 环境](environments/cuda.md)、[UniDepth 环境](environments/unidepth.md)。本次实际运行机为 Windows 11、RTX 4060 Laptop 8188 MiB。

数据：Middlebury trainQ 15 对、NYUv2 labeled 1449 对（795/654 标准 split），以及固定官方 VDA 示例文件。准备说明见 [Middlebury](docs/middlebury_data.md)、[NYUv2](docs/nyuv2_data.md)及[当前视频流程](docs/course_protocol_20261102.md)。

## 目录

~~~text
configs/       路径示例、模型/光流/融合配置、当前打包计划
docs/          当前协议、复现、协作与历史周计划
environments/  官方源码/权重版本、CUDA 与依赖记录
manifests/     固定图像划分和两个视频源清单
src/p3_depth/  读取、模型适配、评测、融合、应用与打包
scripts/       数据准备、运行、复算、导出入口
results/       可随 Git 同步的表格、元数据、选择与验证证据
report/paper/  CVPR LaTeX 正文、参考文献和图片
data/          本地数据，不进 Git
checkpoints/   本地权重，不进 Git
third_party/   固定官方源码，不进 Git
outputs/       原始预测、视频、PDF、ZIP、个人验收指南，不进 Git
~~~

Git pull 不会下载数据、权重或大体积输出。分享时按记录的 SHA256 核查原件；不要把压缩包、个人验收文件或本机绝对路径配置提交到仓库。原始周计划见 [Week 1](docs/week1_plan.md)和[周计划](docs/weekly_plan.md)，其中旧版任务范围以当前协议为准。
