# 脚本入口

实验入口按课程 Part 放入子目录；根目录保留共享辅助工具。参数与执行顺序见 [当前复现说明](../docs/course_protocol_20261102.md)，结果见 [总导航](../PARTS.md)。命令从仓库根目录运行，新输出使用新目录。

```text
scripts/
├── part1_baseline/        StereoSGBM、逐帧 DA V2
├── part2_models/          UniDepth / NYUv2、视频 VDA
├── part3_improvement/     光流、融合、消融、选参冻结
├── application/           背景虚化、旧米制交互演示
├── _common.py            路径、读写、哈希、清单工具
├── _video_inference.py   DA/VDA 共用推理与保存流程
└── *.py                  数据准备、通用评测、绘图、打包
```

## Part 1 · Baseline

| 入口 | 用途 |
| --- | --- |
| [run_stereo_baseline.py](part1_baseline/run_stereo_baseline.py) | 单场景 StereoSGBM |
| [run_stereo_suite.py](part1_baseline/run_stereo_suite.py) | dev 选参、15 场景运行、保存结果复算 |
| [audit_course_stereo.py](part1_baseline/audit_course_stereo.py) | 新版 Stereo MAE / coverage 审计 |
| [run_framewise.py](part1_baseline/run_framewise.py) | 逐帧 DA V2；默认 dav2_relative |

## Part 2 · Models

| 入口 | 用途 |
| --- | --- |
| [run_nyuv2.py](part2_models/run_nyuv2.py) | 静态图像批量推理；课程使用 --model unidepth |
| [evaluate_course_nyu.py](part2_models/evaluate_course_nyu.py) | 新版 NYUv2 主评测 |
| [run_video.py](part2_models/run_video.py) | VDA 视频推理；默认 vda_relative |
| [run_unidepth_sample.py](part2_models/run_unidepth_sample.py) | 单张 UniDepth 冒烟检查 |
| [verify_nyuv2_run.py](part2_models/verify_nyuv2_run.py) | 已保存 NYUv2 旧七指标复算 |
| [summarize_nyuv2.py](part2_models/summarize_nyuv2.py) | 旧单图对比表与图 |
| [evaluate_nyuv2_boundaries.py](part2_models/evaluate_nyuv2_boundaries.py) | 补充边界评测 |

旧七指标和边界分析保留为补充，不混入新版 NYUv2 主表。

## Part 3 · Improvement

| 入口 | 用途 |
| --- | --- |
| [prepare_flow.py](part3_improvement/prepare_flow.py) | RGB 公共光流与可靠区域 |
| [run_fusion_suite.py](part3_improvement/run_fusion_suite.py) | E0–E5 融合与消融 |
| [verify_fusion_suite.py](part3_improvement/verify_fusion_suite.py) | 保存数组复算；复查冻结结果用 --verification-output 新文件 |
| [freeze_fusion_config.py](part3_improvement/freeze_fusion_config.py) | 绑定 dev 证据与冻结参数 |
| [evaluate_relative_video.py](part3_improvement/evaluate_relative_video.py) | 补充相对视频诊断 |

## Application

| 入口 | 用途 |
| --- | --- |
| [render_blur_demo.py](application/render_blur_demo.py) | 本次正式背景虚化应用 |
| [render_demo.py](application/render_demo.py) | 旧米制应用录制 |
| [serve_demo.py](application/serve_demo.py) | 旧米制点选交互网页 |

## 共享辅助工具

| 入口 | 用途 |
| --- | --- |
| [check_setup.py](check_setup.py) | 基础结构检查，仅需 Python 标准库 |
| [inspect_middlebury.py](inspect_middlebury.py) | Middlebury 数据完整性、清单 |
| [prepare_nyuv2.py](prepare_nyuv2.py) | NYUv2 提取与清单 |
| [prepare_video.py](prepare_video.py) | 视频帧缓存与登记 |
| [prepare_model_assets.py](prepare_model_assets.py) | 固定权重下载与校验 |
| [evaluate_depth.py](evaluate_depth.py) | 两种图像数据集的通用保存预测评测 |
| [render_depth_video.py](render_depth_video.py) | 多方法视频比较 |
| [build_course_figures.py](build_course_figures.py) | 报告图表 |
| [package_depth_results.py](package_depth_results.py) | ZIP 打包与预检 |
| [organize_part_results.py](organize_part_results.py) | 结果分类副本与哈希检查 |

以下两个文件供入口导入，不直接作为实验命令执行：

- [_common.py](_common.py)：共用文件操作，不依赖某个实验入口。
- [_video_inference.py](_video_inference.py)：DA/VDA 共用模型调用、计时、保存；对应入口各自限制模型类型。

`src/p3_depth/` 放共享算法和模型适配器，`third_party/` 放官方网络。没有另建一套 `parts/` 转发入口。历史输出的代码路径/哈希按记录中的 Git commit 追溯；当前命令以本页为准。
