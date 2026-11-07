# Part 3：Flow-Guided Fusion

[总导航](../../PARTS.md) · [结果与消融](../../results/by_part/part3_improvement/README.md)

## 代码来源和阅读顺序

本项目由助手实现对 DA V2 relative 输出的后处理；改进方向来自课程任务书。Farneback 光流本身调用 OpenCV，融合和评测逻辑在本仓库。

1. [光流](../../src/p3_depth/temporal/flow.py)：先读 RGB 对应关系与双向检查。
2. [融合](../../src/p3_depth/temporal/fusion.py)：当前 DA 预测 r、历史预测 h，经光流对齐后按 (1-w)r+wh 融合。E4 的 w 会随图像差异、光流残差、预测分歧和边缘强度下降。
3. [时序评测](../../src/p3_depth/evaluation/temporal.py)：所有方法共用 RGB/flow 参考区域；归一化变化不是深度准确率。
4. [选参冻结](../../src/p3_depth/temporal/selection.py)：三个开发候选及冻结凭据。
5. [实验流程](../../scripts/part3_improvement/run_fusion_suite.py)和[保存结果复算](../../scripts/part3_improvement/verify_fusion_suite.py)。

α=0.70 是历史权重上限，不是每个像素都用 70% 历史。第一帧保持原样；历史不可靠时 w=0。当前帧的相对范围只用于置信权重，不把输出逐帧重新归一化。

E0 原 DA；E1 同位置平滑；E2 光流对齐融合；E3 再加可靠性过滤；E4 再加自适应权重；E5 原 VDA 参照。VDA 不经过我们的融合。

## 使用入口

```powershell
python scripts/part3_improvement/prepare_flow.py --help
python scripts/part3_improvement/run_fusion_suite.py --help
python scripts/part3_improvement/verify_fusion_suite.py --help
python scripts/part3_improvement/freeze_fusion_config.py --help
python scripts/render_depth_video.py --help
```

新运行的数据与选参顺序见 [分阶段重跑指南](../../docs/rerun_by_part.md)。默认输出在 outputs/part3_improvement/；光流入口自动生成诊断视频，融合入口自动生成 E0–E4 独立视频以及主比较/消融影片。E5 引用 Part 2 原 VDA。新冻结使用 manifests/videos_current.csv，旧记录仍按原清单验证。历史在 Tokyo 选定 0.70 后用于 rollercoaster，本轮复现保留该设定并重新核查开发证据。

E4 测试变化指标下降 15.93%；E2/E3 的该指标更低，E4 并非数值最优。更低变化可能伴随过度平滑，现有无 GT 视频无法证明准确率或运动延迟改善。3.10 fps 是推理+CPU 光流+融合的加和估算，不含视频解码和写盘。对应影片可观察 E1 拖影与 E4 的取舍。
