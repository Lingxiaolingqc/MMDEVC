# Part 1：Geometry & Frame-Wise Depth

[总导航](../../PARTS.md) · [表格结果](../../results/by_part/part1_baseline/README.md)

## StereoSGBM：传统密集匹配

输入是已校正的左右 RGB 图像。对应点原则上在同一行：算法枚举水平位移，比较邻域图像信息，沿多个方向累计匹配代价，并以 P1/P2 惩罚不合理的视差跳变，得到密集视差。OpenCV 使用 Birchfield–Tomasi 类匹配代价与块支持，本项目是灰度、block size=3、MODE_SGBM_3WAY。

这里没有 SIFT 特征提取，也没有神经网络权重。SIFT 通常检测少量关键点并比较描述子；SGBM 目标是在整幅图上计算稠密视差。弱纹理、重复纹理和遮挡都会使匹配含糊。

- [主要实现：predict_stereo](../../src/p3_depth/models/stereo_sgbm.py)：助手写的 OpenCV 调用与结果筛选；真正匹配内核是 cv2.StereoSGBM_create(...).compute(...)。
- [标定与读取](../../src/p3_depth/data/middlebury.py)：视差先除以 16，再用 Z=f×B/[1000×(d+doffs)] 得米。
- [批量流程](../../scripts/part1_baseline/run_stereo_suite.py)、[单样本流程](../../scripts/part1_baseline/run_stereo_baseline.py)、[新版审计](../../scripts/part1_baseline/audit_course_stereo.py)。
- [最终配置](../../configs/stereo_sgbm_selected.yaml)。
- [OpenCV 官方算法说明](https://docs.opencv.org/4.13.0/d2/d85/classcv_1_1StereoSGBM.html)。

## DA V2-Small：逐帧神经网络

结构：RGB → DINOv2 ViT-S/14 编码器 → 第 3/6/9/12 层特征 → DPT 深度解码器融合/上采样 → H×W 单通道浮点预测。

ViT-S 是 Small Vision Transformer，/14 指 14×14 patch。网络输出 relative inverse-depth-like 数值，通常越大越近；它不是米，不能直接取倒数得到米。运行入口保存原始数组后自动着色并导出本模型视频。视频的每一帧独立运行该图像模型。

- [官方网络](../../third_party/Depth-Anything-V2/depth_anything_v2/dpt.py)：DepthAnythingV2 / DPTHead；需按环境说明准备 third_party。
- [官方固定版本网页](https://github.com/DepthAnything/Depth-Anything-V2/blob/a561b849ebae10a6f5ef49e26c83cbbcd36c71bf/depth_anything_v2/dpt.py)。
- [项目适配器](../../src/p3_depth/models/runtime.py)：ImageModel，加载官方类/权重、RGB/BGR 接口转换、官方归一化/resize、恢复输出尺寸。
- [逐帧运行与保存](../../scripts/part1_baseline/run_framewise.py)。
- [模型论文](https://arxiv.org/html/2406.09414v1)：相对输出与米制模型是不同设置。

## 使用入口

```powershell
python scripts/part1_baseline/run_stereo_suite.py --help
python scripts/part1_baseline/run_stereo_baseline.py --help
python scripts/part1_baseline/audit_course_stereo.py --help
python scripts/part1_baseline/run_framewise.py --help
```

run_framewise.py 默认使用 dav2_relative，只允许 DA 系列；共同推理/计时/保存逻辑在 [公共实现](../_video_inference.py)。新运行按 [分阶段指南](../../docs/rerun_by_part.md)，默认保存至 outputs/part1_baseline/，含独立视频，不依赖 Part 2/3。

历史 Stereo 主结果为 15 场景有效预测 MAE 1.3113 px、GT 像素合并 coverage 62.65%。两项汇总方式不同，不能把低条件误差解释成整个画面都准确。新流程的 DA 独立视频属于 Part 1，多方法比较视频由 Part 3 生成。

## 精选误差分布图

[plot_stereo_distributions.py](plot_stereo_distributions.py) 从已有 Stereo 分析 CSV 生成选定场景的视差/深度箱型图和 coverage 标签。用 `--samples` 指定场景，默认输出到对应分析目录的 `selected_figures/`；不依赖报告制作工具。完整复算及用法见 [分析流程](../../docs/analysis_workflow.md)。
