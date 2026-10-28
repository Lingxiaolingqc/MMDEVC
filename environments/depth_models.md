# DA V2、VDA 与 NYUv2 批量入口

2026-10-28：DA V2 Small relative / Hypersim metric 已分别完成三个 NYUv2 train-side dev 样本；VDA Small metric 已完成官方公开视频的 24 帧 CPU 接口检查。三个视频模型用同一 RGB/时间缓存，保存原始浮点结果。验证记录见 [模型检查结果](../results/depth_models_smoke.md)。这些结果不代表正式测试集或自录视频实验。

## 环境和固定资产

已验证 CPU 环境为 Windows、Python 3.13.5、torch 2.13.0+cpu、torchvision 0.28.0+cpu、8 threads。依赖复用了已有 `.venv`，不是全新安装验证。CUDA 环境另建于 `.venv-cuda`，其状态以运行记录为准。

安装匹配的 PyTorch/torchvision 后，在仓库根目录执行：

```sh
python -m pip install -e ".[data,stereo,nyu,video]"
python -m pip install timm==1.0.22 einops==0.8.1 wandb==0.22.3 safetensors==0.6.2 huggingface-hub==1.29.0 easydict==1.13
git clone https://github.com/DepthAnything/Depth-Anything-V2.git third_party/Depth-Anything-V2
git -C third_party/Depth-Anything-V2 checkout a561b849ebae10a6f5ef49e26c83cbbcd36c71bf
git clone https://github.com/DepthAnything/Video-Depth-Anything.git third_party/Video-Depth-Anything
git -C third_party/Video-Depth-Anything checkout 4f5ae23172ba60fd7bc11ef671cca678842c7072
python scripts/prepare_model_assets.py --model all
```

已有官方 checkout 时跳过 clone。源码 commit、Hugging Face revision、文件大小和 SHA256 固定在 `model_assets.json`；下载及推理入口均核对哈希。权重、官方 checkout 和完整输出不进 Git。自定义 checkpoint_root 时向下载脚本传 `--checkpoint-root`，并保持与本机配置一致。

UniDepth 的源码、config 和权重沿用 [UniDepth 说明](unidepth.md)。DA relative 和 metric 的上游包同名，因此在不同 Python 进程运行；不能在同一进程中先后加载两套同名包。

## 单图开发和全测试集

```sh
python scripts/run_nyuv2.py --model dav2_relative --subset dev --device cpu --output-dir outputs/dav2_relative_dev
python scripts/verify_nyuv2_run.py --run outputs/dav2_relative_dev
python scripts/run_nyuv2.py --model dav2_metric --subset dev --device cpu --output-dir outputs/dav2_metric_dev
python scripts/verify_nyuv2_run.py --run outputs/dav2_metric_dev
python scripts/run_nyuv2.py --model unidepth --subset test --device cpu --output-dir outputs/unidepth_test
python scripts/verify_nyuv2_run.py --run outputs/unidepth_test
```

`--subset test` 不带 `--limit` 才是完整 654 张标准测试。每次用新目录；中断续跑需加 `--resume`，且配置、相关实现哈希、环境、清单和设备必须一致。每张已完成记录还会核对 RGB、GT、预测和 mask 的哈希。复算入口只读取已保存数组，不重新推理。

DA 默认 short side 434；开发样本实际 encoder 输入 **434×574**，与 UniDepth `resolution_level=0` 一致；输出都是原图 480×640。保持各模型官方预处理，记录实际尺寸，不宣称不同官方 resize 实现逐像素完全相同。

- DA relative 保存 inverse-depth-like 原值，单位任意；只有单图评测副本使用 GT 拟合 inverse scale/shift。这组结果须标 `relative_aligned`，不能当作直接米制成绩。
- DA Hypersim metric 与 UniDepth 保存相机 Z 深度，单位 m，不用 GT 对齐。GT 仅用于之后的评测。
- CUDA 运行使用 `--device cuda --precision fp16`；CPU 为 FP32。UniDepth 官方 CUDA infer 使用 FP16 autocast；DA 的最终原图插值以 FP32 完成。正式效率比较必须采用同一设备、精度和实际输入尺寸。

## 视频缓存、推理与可视化

自录要求见 [采集清单](../docs/video_capture.md)。登记一段真实视频的例子：

```sh
python scripts/prepare_video.py --video data/videos/raw/dev_static.mp4 --clip-id dev_static --scene-group room_a --split dev --scene-type static --output-dir outputs/videos/dev_static/cache --register
python scripts/run_video_models.py --model dav2_metric --cache outputs/videos/dev_static/cache --device cpu --precision fp32 --output-dir outputs/videos/dev_static/dav2_metric
python scripts/run_video_models.py --model vda_metric --cache outputs/videos/dev_static/cache --device cpu --precision fp32 --output-dir outputs/videos/dev_static/vda_metric
python scripts/render_depth_video.py --cache outputs/videos/dev_static/cache --run outputs/videos/dev_static/dav2_metric --run outputs/videos/dev_static/vda_metric --output-dir outputs/videos/dev_static/rendered
```

登记会保存原文件、检查重复 clip/hash 和跨 split 的 scene group，并写入 `manifests/videos.csv`。默认最长边 640、目标 10 fps；保留实际 PTS，实际相邻帧间隔可能不同于 0.1 秒。两模型只使用同一缓存，不各自抽帧。视频模型参数接收目标帧率，不能据此假称输入时间间隔精确恒定。

视频默认模型 short side 280；这是待实机预检的起始配置，不是已冻结的正式实验配置。VDA 保留官方 offline 32-frame window、10-frame overlap、padding、keyframe 与跨窗口插值。metric 模式不做尺度拟合。宽高比超过上游阈值时 VDA 会改输入尺寸，因此比较前必须检查元数据的实际 encoder H/W 是否与 DA 相同。VDA 是离线视频处理，吞吐率不能称为实时端到端延迟。

输出 `depths.npy`、`valid.npy` 和 `metadata.json`。渲染保留选中帧的时间戳，并通过重新解码验证；米制固定显示 0–10 m，相对值固定使用整段 1%–99% 分位范围。显示裁剪不修改原始深度；相对值与米制颜色含义不同。`preview.png` 展示首、中、末帧。

## 接口检查命令

公开视频仅检查程序，不登记为自录素材，不参与调参或正式成绩：

```sh
python scripts/prepare_video.py --video third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4 --clip-id upstream_smoke --scene-group upstream_demo --split smoke --scene-type walking --max-long-edge 320 --max-frames 24 --output-dir outputs/video_check/cache
python scripts/run_video_models.py --model vda_metric --cache outputs/video_check/cache --device cpu --precision fp32 --input-size 140 --output-dir outputs/video_check/vda_metric
python -m unittest discover -s tests -v
```

低分辨率检查只验证模型、帧数、保存与可视化流程。正式精度、边界和时序结论必须来自约定分辨率的正式实验。Windows 默认 TEMP 若触发退出清理权限错误，可在当前 PowerShell 将 TEMP/TMP 指向已存在的 `outputs/test-temp`，不修改系统设置。
