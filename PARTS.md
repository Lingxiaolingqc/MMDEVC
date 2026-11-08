# 按 Part 阅读代码与结果

本导航更新于 2026-11-08，对应 2026-11-02 收到的新版 Project 3。主线重跑已有输出；后续按 [补充实验指南](docs/supplementary_experiments.md)运行 LR 分析、边缘评价和两组消融。代码验证不等于正式实验或成员视觉验收。旧输出已归档，暂不删除。

## 新运行入口

按照 [分阶段重跑指南](docs/rerun_by_part.md)执行和检查，不要一次运行全部阶段。默认输出规则在 [output_layout.yaml](configs/output_layout.yaml)：

| 部分 | 新原始输出和可视化位置 |
| --- | --- |
| Part 1 | `outputs/part1_baseline/stereo/`、`outputs/part1_baseline/dav2_relative/` |
| Part 2 | `outputs/part2_models/unidepth/`、`outputs/part2_models/vda_relative/` |
| Part 3 | `outputs/part3_improvement/flow/`、`dev_candidates/`、`selection/`、`test/` |
| Application | `outputs/application/{clip}/` |
| Delivery | `outputs/submission/` |

DA/VDA 各自目录的 `visualizations/comparison.mp4` 只展示 RGB 与该模型；多方法比较由 Part 3 生成。NYU 每张图都有 PNG，`visualizations/index.html` 提供浏览页。新输出直接按 Part 保存，不再复制一层 `outputs/by_part/`。

## 历史结果和阅读材料

| 课程部分 | 可运行入口和源码说明 | 表格、指标、元数据（Git 同步） | 图片、视频和提交文件（本机） |
| --- | --- | --- | --- |
| Part 1：StereoSGBM + 逐帧 DA V2 | [代码入口](scripts/part1_baseline/README.md) | [结果](results/by_part/part1_baseline/README.md) | [媒体](outputs/by_part/part1_baseline/README.md) |
| Part 2：UniDepthV2 + VDA | [代码入口](scripts/part2_models/README.md) | [结果](results/by_part/part2_models/README.md) | [媒体](outputs/by_part/part2_models/README.md) |
| Part 3：光流融合与消融 | [代码入口](scripts/part3_improvement/README.md) | [结果](results/by_part/part3_improvement/README.md) | [媒体](outputs/by_part/part3_improvement/README.md) |
| Application：背景虚化 | [代码入口](scripts/application/README.md) | [结果](results/by_part/application/README.md) | [演示](outputs/by_part/application/README.md) |
| Delivery：报告与 ZIP | [提交说明](docs/submission.md) | [验证记录](results/by_part/delivery/README.md) | [交付文件](outputs/by_part/delivery/README.md) |

从仓库根目录运行，例如：
```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py scripts/part1_baseline/run_stereo_suite.py --help
& $py scripts/part2_models/run_video.py --help
& $py scripts/part3_improvement/run_fusion_suite.py --help
& $py scripts/application/render_blur_demo.py --help
```

实验脚本实际位于 `scripts/part1_baseline/`、`scripts/part2_models/`、`scripts/part3_improvement/` 和 `scripts/application/`；公共数据准备、绘图、打包工具留在 `scripts/`。完整入口清单见 [scripts/README.md](scripts/README.md)。共享算法在 `src/p3_depth/`，官方网络在 `third_party/`，不按 Part 复制算法代码。

2026-11-07 整理了脚本路径，旧 `parts/` 转发层已移除。历史结果内的代码路径和哈希仍描述当时运行，按记录的 Git commit 追溯；新运行会记录新路径与哈希。已有实验、冻结凭据、报告和结果副本均保留。旧版本的未完成推理不能跨本次代码版本 `--resume`，应使用新输出目录。

## 哪些是官方的，哪些是本项目实现的

| 内容 | 来源 | 本项目工作 |
| --- | --- | --- |
| StereoSGBM 匹配内核 | OpenCV | 调用、参数选择、额外左右一致性检查、标定转换、评测与可视化 |
| DA V2 网络和权重 | Depth Anything V2 作者 | 固定版本/权重、格式适配、逐帧推理、保存原始输出 |
| UniDepthV2 网络和权重 | UniDepth 作者 | RGB-only 调用、NYUv2 读取、按课程 crop/clip/指标评测 |
| VDA 网络、时间窗口和拼接 | Video Depth Anything 作者 | 准备相同视频帧、调用官方离线接口、计时/显存与结果保存 |
| Farneback 光流内核 | OpenCV | 在连续 RGB 帧上调用，缓存并复用光流 |
| E1–E4 融合、可靠性过滤、自适应权重、公共评测区域 | 本项目由助手编写 | 实现课程给出的改进方向，进行冻结、控制消融与复算 |
| 背景虚化、图表、视频导出、打包验证 | 本项目由助手编写，使用 OpenCV/NumPy 等库 | 从保存的预测生成应用与交付材料 |
| CVPR 模板 | 官方 author-kit | 保留模板来源，撰写本项目方法、结果和限制 |

“本项目实现”不表示原创发明了 SGBM、光流或融合思想。三种网络都使用官方预训练权重，本项目没有训练/微调网络。官方版本见 [model_sources.yaml](environments/model_sources.yaml)，权重记录见 [model_assets.json](environments/model_assets.json)。

## 历史分类副本检查

```powershell
# 本机：复制精选表格/图/视频，核对每个副本与原件的 SHA256
& $py scripts/organize_part_results.py
& $py scripts/organize_part_results.py --check-only

# B 只有 Git 内容、没有本机 outputs 时，检查小型结果即可
& $py scripts/organize_part_results.py --tables-only --check-only
```

此工具仅针对历史结果；新运行不需要执行。清单是 [configs/part_results.json](configs/part_results.json)。分类目录是方便阅读的副本，历史原件以 [旧路径协议](docs/course_protocol_20261102.md)指定的运行目录为准。副本若与原件不同，整理脚本报错，不覆盖或删除任何一方。原始数组、数据集、权重和个人验收指南没有复制进这组导航目录，也不会随 Git 提交。

Git 提供按 Part 整理的 scripts/ 和 results/by_part/；outputs/by_part/ 的媒体只在本机，和其他大体积输出一样需要另行同步。
