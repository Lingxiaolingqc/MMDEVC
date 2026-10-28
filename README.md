# MMDEVC

**Monocular Metric Depth Estimation & Video Consistency**

AIAA3201 Project 3 · **Spring 2027 项目说明**（课程版本已由组员确认）

## 当前状态

Week 1 A 的单图链路已实现并实际运行：

- Middlebury 15 对：两个 dev 场景比较三组参数后冻结 StereoSGBM 配置；全量预测、覆盖率、逐场景与汇总指标复算通过，见 [结果与失败案例](results/stereo_week1.md)。
- NYUv2：完整 labeled 文件校验通过，1449 对已提取，保持标准 795/654 split；三个 train 开发样本的方向、单位和 mask 已检查，见 [数据说明](docs/nyuv2_data.md)。
- 统一单图评测：七个深度指标、固定 GT 分母和显式 relative alignment，见 [协议与 B 的调用命令](docs/evaluation_protocol.md)。
- UniDepthV2 ViT-S：`nyuv2_0003` 已完成 CPU 真实推理、原始 Z 深度保存与独立复算，见 [结果](results/unidepth_week1.md) 与 [环境](environments/unidepth.md)。

2026-10-28：UniDepthV2、DA V2 metric 与 relative 均完成 NYUv2 标准 654 张评测、保存预测复算和边界评测，见 [正式单图结果](results/nyuv2_standard_test.md)。E0–E5 融合、公共时序评测区域和固定色标视频已在示例短片验证，见 [时序协议与命令](docs/temporal_protocol.md)。GPU 环境见 [CUDA 说明](environments/cuda.md)。

**自录视频的 dev/test 实验、最终融合参数、Demo 和完整报告仍待完成。** 原 A/B 模块主责保留，由助手实现和运行；两位组员的独立复现尚未完成。素材要求见 [采集说明](docs/video_capture.md)，完整状态见 [项目进度](docs/project_progress.md)。

目标：Week 11 之前提交；Week 9 结束冻结方法，Week 10 完成报告、复现检查和提交，预留 2–3 天缓冲。Week 7–10 是倒排里程碑，具体日历日期尚未填写。

## 团队

| 成员 | GitHub | 主负责 |
| --- | --- | --- |
| A：Qinchuan XIAO | [@Lingxiaolingqc](https://github.com/Lingxiaolingqc) | Middlebury/NYUv2、StereoSGBM、UniDepthV2、单图与边界评测、时序融合算法 |
| B：Jun Yu Eugene YIP | [@Geney-gy-29](https://github.com/Geney-gy-29) | DA V2、Video Depth Anything、视频数据、时序与效率评测、消融实验、Demo |

A 提供算法和配置，B 运行统一实验；两人交叉复现并共同核对报告结论。

## 第一次使用

```sh
git clone https://github.com/Lingxiaolingqc/MMDEVC.git
cd MMDEVC
python scripts/check_setup.py
```

已有本地副本时，使用该副本即可。检查脚本仅依赖 Python 标准库，要求 Python 3.10+；它检查仓库结构、清单表头与示例元数据，不检查 GPU 或模型兼容性。基础包的 Python 要求不代表所有官方模型都支持同一 Python 版本。

准备本机配置（目标已存在时跳过复制，直接编辑）：

```powershell
# Windows PowerShell
Copy-Item configs/paths.example.yaml configs/paths.local.yaml
```

```sh
# Linux/macOS
cp -n configs/paths.example.yaml configs/paths.local.yaml
```

编辑 `configs/paths.local.yaml` 中的数据、权重、第三方代码与输出路径；相对路径约定相对于仓库根目录。该文件不会上传。基础 Python 包可选安装：`python -m pip install -e .`；当前不安装 PyTorch 或模型依赖。

进行 Middlebury 数据读取时，安装数据处理依赖并运行：

```sh
python -m pip install -e ".[data]"
python scripts/inspect_middlebury.py --sample Adirondack
python scripts/inspect_middlebury.py --all --write-manifest
```

先按 [数据准备说明](docs/middlebury_data.md) 放置 Quarter Resolution 输入与 GT；`--write-manifest` 会更新相对路径并保留已有 split/notes。新样本暂记 `unassigned`，正式调参前需确定开发/评测划分。

运行已实现的 StereoSGBM 单样本基线（CPU，无需 GPU）：

```sh
python -m pip install -e ".[data,stereo,nyu]"
python -m unittest discover -s tests -v
python scripts/run_stereo_baseline.py --sample Adirondack --config configs/stereo_sgbm_selected.yaml
```

Adirondack、Pipes 为 `dev`，其余 13 场景为 `evaluation`。单样本输出保存在自动创建的 `outputs/stereo_Adirondack_<时间>/`；全量运行与复算命令见 [批量说明](docs/stereo_suite.md)。深度误差只在有效预测上计算，必须与 coverage 和无效惩罚指标一起报告。

固定配置的 13 场景 evaluation 结果：all-GT 像素加权 coverage **61.26%**，含无效惩罚的 Bad>1 px **46.38%**，逐场景平均有效预测 RMSE **0.1201 m**。全部 15 对含 dev，不能称为完全独立测试。最初 block-size-5 的单样本历史记录保留在 [Adirondack 首次结果](results/stereo_adirondack_20261020.md)。

UniDepthV2 环境与本地权重准备好后，在仓库根目录执行（每次用新输出目录）：

```sh
python scripts/run_unidepth_sample.py --sample nyuv2_0003 --device cpu --output-dir outputs/unidepth_week1
python scripts/evaluate_depth.py --dataset nyuv2 --sample nyuv2_0003 --prediction outputs/unidepth_week1/predictions/nyuv2_0003.npy --prediction-type metric_depth --unit m --alignment none --output-dir outputs/eval_unidepth
```

接下来依次阅读：

1. [协作流程](CONTRIBUTING.md)：分支、PR、实验交接。
2. [第一周详细计划](docs/week1_plan.md)：按 A/B 分工开始任务。
3. [数据与预测格式](docs/data_format.md)：先统一接口。
4. [单图评测协议](docs/evaluation_protocol.md)与[时序协议](docs/temporal_protocol.md)：分清 direct metric、GT-aligned relative 与无 GT 视频指标。
5. [环境与官方模型](environments/README.md)：按模型单独配置环境。

## 研究路线

| 部分 | 计划 |
| --- | --- |
| Part 1 | StereoSGBM + DA V2 Small 逐帧相对预测 |
| Part 2 | UniDepthV2 ViT-S；DA V2 Small indoor metric 与 Video Depth Anything Small metric 的视频比较 |
| Part 3 | 在缓存的 metric depth 上实现光流对齐、遮挡处理和自适应权重融合 |
| Application | RGB/固定色标深度、点选深度读数、变化曲线 |

数据：Middlebury Stereo v3 Quarter Resolution、NYU Depth V2 labeled 与标准划分、自录视频。开发/测试视频按完整片段或场景隔离。具体清单由 A/B 核对后填写。

## 目录

```text
configs/       项目设置、本机路径示例、模型计划与 E0–E5 消融矩阵
docs/          周计划、数据格式、评测协议、实验记录、待补充信息
environments/  固定官方源码/权重、CPU 与 CUDA 环境及锁定依赖
manifests/     Middlebury/NYUv2 固定划分；视频清单等待自录素材
src/p3_depth/  数据读取、模型适配、单图/时序评测、光流融合与可视化
scripts/       数据检查、批量推理、融合、保存结果复算与图表导出
demo/          应用展示计划
tests/         数据方向/划分、SGBM 匹配与坐标、单位、七指标、mask 和显式对齐
results/       可提交的精选表格、图片与实验索引
report/        报告提纲与写作分工
data/          本地数据，仅跟踪 README
checkpoints/   本地权重，仅跟踪 README
outputs/       完整实验输出，仅跟踪 README
third_party/   官方仓库的本地 checkout，仅跟踪 README
```

每次正式实验保存代码 commit、配置、权重标识、数据清单版本、环境、原始浮点预测、指标和日志。实验目录与交接字段见 [data_format.md](docs/data_format.md)。

## 里程碑与下一步

完整安排见 [每周计划](docs/weekly_plan.md)，第一批工作见 [任务清单](docs/task_board.md)。

- A：在自录 dev 视频检查融合的边缘和运动响应，再冻结参数。
- B：按采集清单准备独立 dev/test 视频，完成正式时序实验与展示。
- 共同：人工检查真实模型结果，并由成员独立复现。

[待补充信息](docs/open_questions.md)包括运行系统/GPU、数据共享位置、实际截止日期和协作者访问状态。代码许可待两位作者决定；第三方代码、模型与数据遵循各自的许可。
