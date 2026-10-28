# UniDepthV2 ViT-S 单样本复现

已验证 Windows 11、Python 3.13.5、CPU float32、8 threads、batch=1。使用官方代码和完整 ViT-S 权重，strict load 成功，34,184,323 个参数；没有训练、GT 尺度对齐或替代模型。CPU PyTorch 为 `2.13.0+cpu`，torchvision 为 `0.28.0+cpu`。本页保留 CPU 单样本记录；新增 RTX 4060 Laptop 的 CUDA 654 张评测见 [CUDA 环境](cuda.md)和[标准测试结果](../results/nyuv2_standard_test.md)。

源代码/权重完整 SHA 见 [model_sources.yaml](model_sources.yaml)，实际包版本与输入尺寸见 [运行元数据](../results/unidepth_week1_provenance.json)。本机 `.venv` 设置 `include-system-site-packages=true`，复用了已有 NumPy/PyTorch 等依赖，只在项目环境补充 timm/einops/wandb/safetensors；未重装验证一个全新独立环境。

## 准备

在仓库根目录的模型环境中安装项目数据依赖及官方推理需要的包；PyTorch/torchvision 应先按所选 CPU/CUDA 环境安装匹配版本。本次记录版本的 CPU 安装入口为 PyTorch 官方 CPU index，另列的依赖来自 PyPI：

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.13.0 torchvision==0.28.0
python -m pip install -e ".[data,stereo,nyu]"
python -m pip install timm==1.0.22 einops==0.8.1 wandb==0.22.3 safetensors==0.6.2 huggingface-hub==1.29.0
git clone https://github.com/lpiccinelli-eth/UniDepth.git third_party/UniDepth
git -C third_party/UniDepth checkout 8d8cfe4c7ee15297099983607febf0d4f32eb3d6
```

下载以下固定版本的两个公开文件到 `checkpoints/unidepth-v2-vits14/`（已有文件不必重复下载）：

- [config.json](https://huggingface.co/lpiccinelli/unidepth-v2-vits14/resolve/038c238f06c87b6c2f5b3749fd51fbf442b1f218/config.json)
- [model.safetensors](https://huggingface.co/lpiccinelli/unidepth-v2-vits14/resolve/038c238f06c87b6c2f5b3749fd51fbf442b1f218/model.safetensors)，136,777,580 bytes。

Windows 可直接用浏览器或 `curl.exe -L --retry 5 -C - -o <目标文件> <上面的固定版本链接>` 下载，以避免 Hugging Face 深层缓存文件名触发长路径问题。入口会校验权重 SHA256 和官方源码 SHA/干净状态；不匹配时停止。

准备 NYUv2 labeled 与 split，按 [数据说明](../docs/nyuv2_data.md) 建立本地数据和清单。全量 MAT 不用于模型输入，模型只接收指定 RGB；GT 只交给独立评测函数。

## 运行与复算

```sh
python scripts/run_unidepth_sample.py --sample nyuv2_0003 --device cpu --resolution-level 0 --threads 8 --output-dir outputs/unidepth_week1
python scripts/evaluate_depth.py --dataset nyuv2 --sample nyuv2_0003 --prediction outputs/unidepth_week1/predictions/nyuv2_0003.npy --mask outputs/unidepth_week1/masks/nyuv2_0003.npy --prediction-type metric_depth --unit m --alignment none --output-dir outputs/unidepth_eval_again
```

每次使用新输出目录。脚本只接受 `manifests/nyuv2_dev.csv` 中训练侧样本。CPU 推理保持 float32；`--device cuda` 只有在当前环境 `torch.cuda.is_available()` 为真时才可用。模型官方 CUDA 路径使用 float16 autocast，需单独验证数值与显存，不能照搬 CPU 记录。

配置 level=0 时，此图实际编码器输入为 **434×574**（官方 resize 取 14 的倍数），输出恢复 **480×640**。输入是完整 RGB，使用官方 ImageNet normalization；未提供 GT 内参，内参由模型预测。保存官方 `depth` 的相机 Z 分量，不能替换成 `radius`。GT 和预测固定显示 0–10 m，误差固定显示 0–1 m；指标不按显示上限裁剪。

输出包含 `predictions/<sample>.npy`、`masks/<sample>.npy`、`predictions.csv`、`metadata.json`、解析配置、可视化和 `evaluation/`。推理计时包含官方 infer 的预处理、内部 resize、设备传输和前向；排除加载权重、磁盘读写和评测，预热一次后计时一次。

上游可能提示 CUDA、xFormers、KNN、NystromAttention 或训练 patch 算子不可用。本次 ViT-S CPU infer 已在这些提示下成功；项目单图指标用 NumPy，不调用上游点云 KNN 评测或训练损失。真正失败应看 traceback/非零退出码，而不能仅凭这些提示认定模型失败。入口关闭 wandb 记录和 Hugging Face 联网下载。

若 Windows 沙箱的默认 TEMP 触发退出清理权限错误，可在 PowerShell 当前进程把 `TEMP`、`TMP` 指向已存在的 `outputs/test-temp` 再运行；不需要改系统环境变量。
