# Part 1：相对视频诊断

Depth Anything V2 Small relative 输出 inverse-depth-like 相对值，近处通常较大，单位不是米。该流程直接测量其原始输出变化，不取倒数冒充真实深度，也不使用逐帧 GT 对齐。

先按 [模型入口](../environments/depth_models.md) 准备缓存，并按 [时序协议](temporal_protocol.md) 准备同一缓存的光流。对于每个自录片段执行以下命令，将 `dev_static` 替换为实际 clip ID：

```sh
python scripts/run_video_models.py --model dav2_relative --cache outputs/videos/dev_static/cache --device cuda --precision fp16 --input-size 364 --output-dir outputs/videos/dev_static/dav2_relative
python scripts/evaluate_relative_video.py --cache outputs/videos/dev_static/cache --run outputs/videos/dev_static/dav2_relative --flow outputs/videos/dev_static/flow --output-dir outputs/videos/dev_static/relative_diagnostics
python scripts/render_depth_video.py --cache outputs/videos/dev_static/cache --run outputs/videos/dev_static/dav2_relative --output-dir outputs/videos/dev_static/relative_rendered
```

输出包含原始相对单位的逐帧中位数、flow-aligned L1、每秒变化量、预测覆盖率、固定相机静止场景的像素标准差，以及诊断曲线。只有 `scene_type=static` 且相机和物体确实静止时，标准差才能作为静止闪烁诊断。

固定公共区域由 RGB 与光流确定。额外的归一化诊断用整段有效值的 99% 分位减去 1% 分位作为唯一跨度，将绝对差除以该跨度后截断到 1；无效预测对按 1 计入固定分母。跨度为零或没有有效值时，归一化结果为空。这个跨度只用于诊断和固定色标，不改变保存的原始预测，更不逐帧归一化。

移动相机片段应同时查看 RGB、相对深度视频、中位数曲线和光流误差。相机靠近目标、物体进入画面、可见范围改变都可能改变中位数。无 GT 的这些变化只能说明相对输出如何变化，不能单独证明物理尺度漂移，更不能证明米制准确率。

相对诊断与 E0–E5 米制实验分别报告。各自保留单位、范围与覆盖率，不能把两类 L1 放在同一排名中。已有程序验证见 [24 帧检查记录](../results/relative_video_smoke.md)；自录静止、移动及遮挡诊断仍待素材。
