# 环境与官方实现

基础包没有强制第三方运行依赖；`scripts/check_setup.py` 可直接用 Python 3.10+ 执行。StereoSGBM/NYUv2/单图评测使用 `python -m pip install -e ".[data,stereo,nyu]"`，不需要 GPU 或 opencv-contrib。UniDepthV2 ViT-S 已完成 CPU 真实单样本推理，见 [已验证环境与命令](unidepth.md)。DA V2 relative / metric 与 VDA metric 的 CPU 小样本流程也已验证，批量和视频命令见 [depth_models.md](depth_models.md)；CUDA 三模型标准测试和 VDA 预检已完成，见 [CUDA 环境](cuda.md)。

Middlebury 读取与诊断使用可选依赖：`python -m pip install -e ".[data]"`（NumPy、Pillow、PyYAML、Matplotlib）。2026-10-20 本机数据检查通过的版本记录在 [数据检查说明](../docs/middlebury_data.md)；该记录不代表模型环境兼容性。

先按官方说明为 DA V2、UniDepthV2、VDA 建立各自兼容的环境。根据选定电脑的系统、驱动及官方实现选择 Python/PyTorch/CUDA 组合；不预先假定所有模型可放在同一环境。

| 方法 | 官方来源 | 主责 |
| --- | --- | --- |
| DA V2 relative / metric | [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)，metric 使用仓库的 `metric_depth/` 说明 | B |
| UniDepthV2 | [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) | A |
| Video Depth Anything | [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) | B |

每个模型第一次跑通后：

1. 在 `model_sources.yaml` 记录实际 checkout 的完整 commit SHA、checkpoint 来源/哈希与已验证状态。
2. 在本目录添加该模型的安装步骤、Python/PyTorch/CUDA 版本及环境导出文件；去掉个人绝对路径和访问令牌。
3. 写清输入输出及原始预测保存命令，完成一次小样本检查。
4. 通过统一的磁盘预测格式对接评测；先保证交接正确，再考虑合并环境。

`configs/models.example.yaml` 仅表示模型选择计划，不是可执行推理配置。`null` 表示尚未确认，不能被正式运行静默接受。
