# 点选深度 Demo

主责：B / Jun Yu Eugene YIP。助手已实现并运行；组员人工验收待完成。

本地浏览器同时显示 RGB、固定色标米制深度、所选像素的原始 Z 深度和各方法的时间曲线。可以播放、拖动帧、切换方法和点击两幅画面；服务只监听 `127.0.0.1`，读取已有预测，不需要再次运行模型。

Z 是沿相机前方轴的深度，不是相机中心到该点的欧氏距离。时间曲线始终取同一个画面坐标，不跟踪物体。数值不受显示色标上限裁剪；无效值留空，曲线断开。只有原始、未用 GT 对齐的米制预测可以进入此应用。

## 启动

在仓库根目录、已安装视频处理依赖的环境执行。下面使用本机已生成的 24 帧上游示例素材；在新电脑先按 [时序协议](../docs/temporal_protocol.md) 生成对应缓存和结果。示例只验证程序，正式交付必须换成自录视频。

```powershell
$py = '.\.venv-cuda\Scripts\python.exe'
& $py scripts/serve_demo.py `
  --cache outputs/video_smoke_20261028/cache `
  --run outputs/video_smoke_20261028/fusion_364/E0 `
  --run outputs/video_smoke_20261028/fusion_364/E4 `
  --run outputs/video_smoke_20261028/vda_gpu_364 `
  --port 8765
```

浏览器打开终端打印的本地地址。默认端口已被使用时换一个端口，或传 `--port 0` 由系统分配。终端按 Ctrl+C 停止服务。可传 `--session-record outputs/<新目录>/session.json` 保存启动时的输入与代码哈希。

## 导出演示视频

```powershell
& $py scripts/render_demo.py `
  --cache outputs/video_smoke_20261028/cache `
  --run outputs/video_smoke_20261028/fusion_364/E0 `
  --run outputs/video_smoke_20261028/fusion_364/E4 `
  --run outputs/video_smoke_20261028/vda_gpu_364 `
  --method E4 --x 160 --y 90 `
  --output-dir outputs/demo_export
```

每次使用新的输出目录。坐标属于缓存图像，省略时取中心点。输出 `demo.mp4`、`preview.png`、`point_trace.csv` 和元数据；自动重解码核验帧数与原时间戳，并将 CSV 数值与原始预测核对。视频由程序绘制相同应用信息，不能称为浏览器操作录屏。

## 已验证与待完成

2026-10-28：三个 Demo 测试与全仓库 51 个测试通过。浏览器实际点选、切帧、方法切换及播放到末帧通过；读数与保存数组一致。24 帧 MP4 导出和时间戳检查通过，预览已检查。详见 [验证记录](../results/demo_smoke.md)。

当前素材来自上游示例，E4 参数仍 provisional。自录视频到位后，需在冻结后的正式测试结果上重复导出，并由成员验收应用；当前不能将此视频作为已完成的自录项目实验。
