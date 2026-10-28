# 已验证 CUDA 环境

2026-10-28：Windows、Python 3.13.5、RTX 4060 Laptop 8188 MiB、驱动 580.88。独立 `.venv-cuda` 已完成三个单图模型各 654 张 NYUv2 评测及 VDA 24 帧离线推理。包版本见 [环境快照](cuda_environment.json)，CPU 历史环境保持独立。

## 安装

在仓库根目录使用 Python 3.13，先安装匹配的 GPU PyTorch，再安装锁定依赖和项目：

```powershell
python -m venv .venv-cuda
$py = '.\.venv-cuda\Scripts\python.exe'
& $py -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r environments/cuda-dependencies.txt --index-url https://pypi.org/simple
& $py -m pip install -e .
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name())"
```

最后一条应显示 `2.10.0+cu128`、`True` 和本机 GPU 名称。若机器全局 pip 配置包含不可用的镜像，可在当前 PowerShell 设置 `$env:PIP_CONFIG_FILE='nul'` 后重试，不必修改全局配置。Windows 默认临时目录若触发权限问题，可将当前进程的 TEMP/TMP 指向仓库内已存在的 `outputs/test-temp`。

模型源码与权重按 [模型准备说明](depth_models.md) 下载。推理使用 `--device cuda --precision fp16`；模型自身预处理和最终原尺寸输出保留。没有安装 xFormers 或上游训练专用算子；这套推理路径已实际验证。

## 显存预检

同一个 24 帧示例、VDA Small metric、FP16、官方 32 帧离线窗口：

| short side / 实际 H×W | PyTorch peak allocated | PyTorch peak reserved | 整段吞吐率 |
| --- | --- | --- | --- |
| 280 / 280×504 | 1.09 GiB | 1.30 GiB | 20.22 fps |
| 364 / 364×644 | 2.31 GiB | 3.09 GiB | 9.06 fps |
| 518 / 518×924 | 7.81 GiB | 11.58 GiB | 0.92 fps |

这是资源预检，不是自录视频最终性能表。allocated 为张量峰值，reserved 为分配器保留量；均不是桌面、驱动和其他进程在内的整卡占用。518 运行时 reserved 超过物理显存容量，吞吐率明显下降，因此视频入口默认 short side=364。选择依据是内存与输入尺寸，不是正式测试成绩；正式 dev 阶段仍需检查真实片段。

NYUv2 的实际 encoder 输入为 434×574。三种单图模型的 allocated 峰值约 0.27–0.39 GiB；完整时间和显存见 [标准测试结果](../results/nyuv2_standard_test.md)。这些数字不能外推成任意分辨率或批大小的显存需求。
