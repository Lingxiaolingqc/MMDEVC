# MMDEVC

**Monocular Metric Depth Estimation & Video Consistency**

AIAA3201 Project 3 · **Spring 2027 项目说明**（课程版本已由组员确认）

## 当前状态

仓库已建立项目结构、协作规则、数据格式与实验计划。Middlebury Quarter Resolution 的 15 组训练数据已完成读取检查；StereoSGBM 已在 Adirondack 开发样本上跑通视差、米制深度、左右检查、单样本评测和保存流程。详情见 [数据准备](docs/middlebury_data.md) 与 [StereoSGBM 基线](docs/stereo_baseline.md)。**NYUv2 尚未准备；单目模型、正式多方法评测、时序融合和 Demo 尚未实现或运行。**

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
python -m pip install -e ".[data,stereo]"
python -m unittest discover -s tests -v
python scripts/run_stereo_baseline.py --sample Adirondack
```

Adirondack 已标为 `dev`，其余 14 场景的划分待确认。完整输出保存在自动创建的 `outputs/stereo_Adirondack_<时间>/`，参数、有效区域与指标口径见 [基线说明](docs/stereo_baseline.md)。

已验证的首次输出见 [Adirondack 结果记录](results/stereo_adirondack_20261020.md)：全 GT 区域覆盖率 73.28%，有效预测内深度 RMSE 0.1015 m。报告包含无效惩罚指标与固定色标对比图。

接下来依次阅读：

1. [协作流程](CONTRIBUTING.md)：分支、PR、实验交接。
2. [第一周详细计划](docs/week1_plan.md)：按 A/B 分工开始任务。
3. [数据与预测格式](docs/data_format.md)：先统一接口。
4. [评测协议草案](docs/evaluation_protocol.md)：标有待确认的项不能当作正式协议。
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
environments/  官方代码来源及后续验证过的环境记录
manifests/     Middlebury 15 场景清单；NYUv2/视频仍只有表头
src/p3_depth/  Middlebury 读取器、StereoSGBM、单样本评测与可视化；融合待实现
scripts/       check_setup.py、inspect_middlebury.py、run_stereo_baseline.py
demo/          应用展示计划
tests/         Middlebury 读取、SGBM 坐标/单位与真实合成匹配、指标和 mask 检查
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

- A：完成 NYUv2 小样本清单与读取规范；确定剩余 Middlebury 场景的划分，接入单目深度预测与比较。
- B：先跑通 DA V2 Small 单图原始预测，准备短视频。
- 共同：A 的评测能够读取 B 的预测，并由另一人独立复算。

[待补充信息](docs/open_questions.md)包括运行系统/GPU、数据共享位置、实际截止日期和协作者访问状态。代码许可待两位作者决定；第三方代码、模型与数据遵循各自的许可。
