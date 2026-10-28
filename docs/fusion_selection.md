# 开发选择与测试前冻结

当前是可执行流程，尚未选出正式参数。自录素材仍待提供；示例视频不进入正式选择。主责保持 A 实现、B 实验；助手可以运行和观察，记录 `reviewer_kind: assistant`，不代替成员验收。

## 1. 固定素材划分，缓存网络输出

按 [采集说明](video_capture.md) 录制并用 `prepare_video.py --register` 登记全部 dev/test。缓存路径和源视频 SHA256 写入 `manifests/videos.csv`，同一场景不能跨 split。正式选择要求所有候选覆盖全部已登记 dev；test 内容不用于选参数。

以下命令约定每段缓存位于 `outputs/videos/<clip_id>/cache`，DA/VDA 输出分别为 `dav2_metric`、`vda_metric`，光流为 `flow`。模型生成见 [环境与推理说明](../environments/depth_models.md)，光流生成见 [时序协议](temporal_protocol.md)。先准备 dev 输出，test 融合评测必须等冻结之后。

## 2. 只比较三个小候选

| 候选 | 配置 | history_weight |
| --- | --- | --- |
| alpha050 | `configs/fusion_history050.yaml` | 0.50 |
| alpha070 | `configs/fusion_provisional.yaml` | 0.70 |
| alpha085 | `configs/fusion_history085.yaml` | 0.85 |

其余参数完全相同。三个数是待评估的有限候选，不是已证明最优值。复用相同网络和光流缓存，不重跑深度模型，也不做大规模网格搜索。每次保存 E0–E5 全套结果；E0、E5 在候选间应一致。

在仓库根目录的 PowerShell 中运行：

```powershell
$py = '.\.venv-cuda\Scripts\python.exe'
$devClips = @(Import-Csv manifests/videos.csv | Where-Object split -eq 'dev')
if ($devClips.Count -eq 0) { throw '请先登记真实 dev/test 素材' }
$candidates = [ordered]@{
  alpha050 = 'configs/fusion_history050.yaml'
  alpha070 = 'configs/fusion_provisional.yaml'
  alpha085 = 'configs/fusion_history085.yaml'
}
foreach ($clip in $devClips) {
  $base = 'outputs/videos/' + $clip.clip_id
  foreach ($name in $candidates.Keys) {
    $suite = "$base/$name"
    & $py scripts/run_fusion_suite.py --cache "$base/cache" --baseline "$base/dav2_metric" --reference-vda "$base/vda_metric" --flow "$base/flow" --config $candidates[$name] --output-dir $suite
    if ($LASTEXITCODE -ne 0) { throw "融合失败：$suite" }
    & $py scripts/verify_fusion_suite.py --suite $suite --cache "$base/cache" --baseline "$base/dav2_metric" --reference-vda "$base/vda_metric" --flow "$base/flow"
    if ($LASTEXITCODE -ne 0) { throw "复算失败：$suite" }
  }
}
```

输出目录必须是新的。复算成功后生成 `verification.json`，记录六版本结果、输入元数据和复算程序的哈希；它只是程序核验记录。

## 3. 看数值，也看响应与拖影

每个 dev 对比三候选的 `comparison.csv`，并通过 Demo 或固定色标视频检查：静止闪烁、前后移动是否延迟、遮挡后是否留下旧物体、细边缘是否被抹平。记录具体时间段/像素；不适用的项目写明理由。不能只按最小 temporal L1 选，因为停滞的预测也可能有较小变化。

将 `fusion_selection_review.example.json` 复制到 `outputs/video_selection/review.json`。每个候选下填入所有 dev ID，值的结构如下（示意内容不能当作实际观察）：

```json
{
  "dev_static": {
    "stability": "待填写观察时间段、对比与数值",
    "motion_response": "不适用：固定相机和静止场景；运动响应在相应 dev 片段观察",
    "occlusion_and_edges": "待填写细边缘或遮挡现象及位置"
  }
}
```

完成实际观察后填写 `selected_candidate`、`rationale`，并将 `status` 改为 `complete`。理由应说明稳定性、响应与边缘之间的取舍。若均无改善，保留负结果，选定一个已评估配置完成消融，不修改 test 来制造提升。

## 4. 从已验证的开发结果生成冻结配置

继续使用上面 PowerShell 的变量：

```powershell
$reviewPath = 'outputs/video_selection/review.json'
$selected = (Get-Content -Raw $reviewPath | ConvertFrom-Json).selected_candidate
$freezeArgs = @('--selected', $selected, '--review', $reviewPath, '--output-dir', 'outputs/video_selection/frozen')
foreach ($name in $candidates.Keys) {
  foreach ($clip in $devClips) {
    $freezeArgs += @('--candidate', "$name=outputs/videos/$($clip.clip_id)/$name")
  }
}
& $py scripts/freeze_fusion_config.py @freezeArgs
if ($LASTEXITCODE -ne 0) { throw '冻结失败，先解决显示的证据问题' }
```

成功生成 `selection.json`、逐片逐候选的 `development_comparison.csv` 和 `fusion_frozen.yaml`。参数直接取自所选候选，不手动重填。不会自动宣布某个方法胜出，也不会自动标记成员已审核。

## 5. 冻结后运行独立 test

每个 test 的 `run_fusion_suite.py` 使用 `--config outputs/video_selection/frozen/fusion_frozen.yaml`，然后同样执行复算和导出。入口会重新核对清单、片段归属、开发结果/观察记录与参数。清单、开发证据或参数发生变化时，先停止并查明原因；不能通过补写证据把已看过的 test 伪装成未使用测试。

测试结果按原定规则全部报告，包括没有收益和明显失败的片段。冻结证据、逐片表与观察记录随后精选到 `results/`；完整预测仍留在本地。当前没有真实视频，因此没有生成正式冻结配置、开发选择结论或最终测试结果。
