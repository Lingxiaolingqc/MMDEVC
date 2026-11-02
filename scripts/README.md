# 命令入口

当前课程的参数与执行顺序见 [新版复现说明](../docs/course_protocol_20261102.md)。所有命令从仓库根目录运行；新输出不得覆盖已有结果。

| 用途 | 入口 |
| --- | --- |
| 仓库结构检查 | check_setup.py |
| Middlebury 读取 / SGBM 单样本 / 全量与复算 | inspect_middlebury.py / run_stereo_baseline.py / run_stereo_suite.py |
| 新版 Stereo GT / MAE / coverage 审计 | audit_course_stereo.py |
| NYUv2 提取 / 推理 / 新版评测 | prepare_nyuv2.py / run_nyuv2.py / evaluate_course_nyu.py |
| 固定模型权重 | prepare_model_assets.py；UniDepth 见独立环境说明 |
| 视频缓存 / relative 模型 / 公共光流 | prepare_video.py / run_video_models.py / prepare_flow.py |
| E0–E5 / 原数组复算 / 开发冻结 | run_fusion_suite.py / verify_fusion_suite.py / freeze_fusion_config.py |
| 固定范围比较 / 背景虚化 | render_depth_video.py / render_blur_demo.py |
| 报告图 / 正式视频 ZIP | build_course_figures.py / package_depth_results.py |

其他单图七指标、边界、相对对齐和米制点选 Demo 入口保留，用于历史实验与补充分析。它们的旧协议数字不能混入新版 NYUv2 主表。check_setup 只检查基础结构，不保证 GPU、数据或实验结果正确。
