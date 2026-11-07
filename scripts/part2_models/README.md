# Part 2：Metric & Temporally Consistent Depth

[总导航](../../PARTS.md) · [表格结果](../../results/by_part/part2_models/README.md)

## UniDepthV2 ViT-S：为什么单图能输出米

它也是 ViT 网络：RGB → DINOv2 ViT-S/14 特征 → CameraHead 估计相机内参 → 为像素生成射线 → DepthHead 根据图像和射线预测三维距离。官方代码用点 P=radius×ray，再取 Z 分量作为 result['depth']；项目使用的正是这个以米表示的相机轴向深度。

米制能力来自官方训练的度量监督，以及学到的物体、场景和相机统计规律。本项目只加载官方权重，并未训练模型。预测相机参数有助于解释图像几何，但不能单独消除尺度歧义：投影 u=fX/Z+cx 中同时把 X、Z 放大相同倍数，图像位置不变。因此输出单位是米，并不保证每张图都测准，仍需要 NYUv2 真值评测。

DA V2 与 UniDepthV2 都有 ViT 编码器；输出含义的区别来自完整模型结构、训练目标、数据及 checkpoint，并非“只有 UniDepth 才是 ViT”。当前视频使用 DA relative checkpoint；DA 官方也有另一套 metric checkpoint，但不是本次视频配置。

- [官方 UniDepthV2](../../third_party/UniDepth/unidepth/models/unidepthv2/unidepthv2.py)：infer / rays 与 Z-depth。
- [官方解码器](../../third_party/UniDepth/unidepth/models/unidepthv2/decoder.py)：CameraHead / DepthHead。
- [项目适配器](../../src/p3_depth/models/runtime.py)：ImageModel，只传 RGB，不传测试 GT。
- [NYUv2 运行](../../scripts/part2_models/run_nyuv2.py)、[课程指标实现](../../src/p3_depth/evaluation/course.py)、[新版重算](../../scripts/part2_models/evaluate_course_nyu.py)。
- [官方论文](https://arxiv.org/html/2502.20110v1)、[固定源码网页](https://github.com/lpiccinelli-eth/UniDepth/blob/8d8cfe4c7ee15297099983607febf0d4f32eb3d6/unidepth/models/unidepthv2/unidepthv2.py)。

## VDA-Small：多帧相对深度与窗口

项目把整段帧缓存交给官方 infer_video_depth；官方内部每次 forward 输入 [1,32,3,H,W]，输出 [1,32,H,W]。DINOv2 先提取每帧特征，带 temporal attention 的时空预测头在 32 个帧槽之间交换信息。这个 32 是时间窗口长度，不是 32 个互不相关的训练样本。

当前使用 relative checkpoint。窗口、关键帧、补齐、窗口间尺度/偏移对齐和拼接全部来自官方实现；我们的 DA 光流融合没有改动 VDA。

固定版本常量为 INFER_LEN=32、OVERLAP=10、KEYFRAMES=[0,12,24,25,26,27,28,29,30,31]、INTERP_LEN=8。每次推进 22 个槽。后续窗口前 10 槽取上一窗口实际输入的这些关键帧，组成两个尺度锚点和八个连续重叠槽；并非简单重用前窗最后 10 帧。

下面是按固定源码推导并用纯索引模拟核对的 70 帧输入，帧编号从 0 开始，末帧为 69：

| 窗口 | 两个锚点 | 八个重叠槽 | 新帧槽 |
| --- | --- | --- | --- |
| W1 | — | — | 首个完整输入是 0–31 |
| W2 | 0、12 | 24–31 | 32–53 |
| W3 | 0、34 | 46–53 | 54–69，复制末帧补 6 槽 |
| W4 | 0、56 | 68、69，复制末帧补 6 槽 | 复制末帧填 22 槽 |

循环起点来自 range(0,70,22)，所以是 4 窗。最后一窗仍会参与末尾共同帧的结果融合，不能因为第三窗已看到最后真实帧就说只有三窗。

每个后续窗口：
1. 用两个锚点预测估计窗口间 scale-and-shift；这是预测对预测的对齐，没有使用 GT。
2. 对八个共同时间点的深度预测交叉渐变，后窗权重为 0,1/7,…,1。
3. 追加 22 个新输出槽。长度依次 32→54→76→98，最后裁回真实 70 帧。

因此，“内部复制末帧凑满窗口”和“没有把课程素材伪造为 100 帧”是两件事。课程输入、评分和导出视频始终只有 70 个真实时间点。预热另有一次单帧调用，内部补成 32 槽；上表不含预热。

- [官方窗口实现](../../third_party/Video-Depth-Anything/video_depth_anything/video_depth.py)：infer_video_depth。
- [官方时空预测头](../../third_party/Video-Depth-Anything/video_depth_anything/dpt_temporal.py)。
- [项目适配器](../../src/p3_depth/models/video.py)：VideoDepthModel，固定官方版本、加载权重、调用离线推理、检查输出长度。
- [视频运行/资源记录](../../scripts/part2_models/run_video.py)、[统一视频输入](../../scripts/prepare_video.py)。
- [固定窗口源码网页](https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/video_depth_anything/video_depth.py#L91-L162)。
- [官方拼接函数](https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/utils/util.py#L16-L74)、[论文 Architecture](https://arxiv.org/html/2501.12375v4#S3.SS1)。

## 使用入口

```powershell
python scripts/part2_models/run_nyuv2.py --model unidepth --help
python scripts/part2_models/evaluate_course_nyu.py --help
python scripts/part2_models/run_video.py --help
python scripts/prepare_video.py --help
python scripts/render_depth_video.py --help
```

run_nyuv2.py 默认使用 unidepth，自动生成逐图 PNG 和浏览页；run_video.py 默认使用 vda_relative，只允许 VDA 系列，共用 [推理实现](../_video_inference.py)，自动生成 VDA 独立视频。默认输出在 outputs/part2_models/；参数详情和顺序见 [分阶段重跑指南](../../docs/rerun_by_part.md)。两模型实际视频 encoder 尺寸为 364×644，输出为 360×640，FP16。不是把输入压成 364×364。
