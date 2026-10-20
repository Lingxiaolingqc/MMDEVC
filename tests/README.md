# 关键验证计划

当前有 10 项 Middlebury 回归测试：PFM 字节序与上下方向、非单位头字段、截断文件、标定/单位/主点差、无效深度、mask 编码、分包配对、缺失/重复文件、尺寸错误、手工划分保留。

安装 `.[data]` 后运行 `python -m unittest discover -s tests -v`。测试使用自行生成的极小样本，不下载数据或依赖 GPU；临时文件位于被忽略的 `outputs/test-temp/`。实际 15 场景检查由 `scripts/inspect_middlebury.py --all` 完成。

模型和精度评测尚未实现，因此当前没有对应测试。

后续优先检查：标定与单位换算、prediction=GT 的指标、无效/空区域处理、原始预测读写、视频帧对应、warping 边界。只编写能发现实际计算或接口错误的检查。
