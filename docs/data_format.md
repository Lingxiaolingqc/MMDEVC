# 数据与预测格式 v0.1

本协议是第一周共同使用的接口约定；保存/读取实现尚未编写。A/B 接入时发现问题，应通过 PR 修改本文并说明兼容性。

## 数据清单

`manifests/` 中 CSV 使用 UTF-8，路径统一用 `/`，相对于 `paths.local.yaml` 的对应 `data_roots`。不提交本机绝对路径。

- `middlebury.csv`：每行一组左右图、左 GT disparity 与标定；`split` 使用 `dev` 或 `evaluation`，标记调参场景。完整 15 对汇总不能全部称为独立测试。
- `nyuv2.csv`：RGB/GT 一一对应；记录标准划分文件的来源和版本。正式划分保持原始语义，开发样本从允许的训练侧选择。
- `videos.csv`：每行一个片段；`split` 使用 `dev` 或 `test`；同一场景的关联片段使用相同 `scene_group`，整个 group 只属于一侧。
- `frames.example.csv`：抽帧后生成的逐帧索引表头；实际索引放在对应本地数据目录。

`sample_id`、`clip_id` 在各自清单中唯一。视频使用真实解码帧编号，起始为 0；抽帧后保留原帧编号。`timestamp_s` 相对于片段起点，以秒表示；变帧率视频使用实际时间戳，不简单用 frame/fps 替代。尺寸字段均按 `height,width`。

## 原始预测

- 每张图/每帧保存二维 `float32` `.npy`，数组为 `H × W`。
- `prediction_type` 取 `metric_depth`、`relative_depth` 或 `inverse_depth_like`；`metric_depth` 统一转换为相机光轴 Z 深度，单位 `m`。
- 对相对表示，`unit` 为 `arbitrary`，必须记录数值方向及转换方式；不能直接称为米或直接与米制深度混合。
- 无效像素存 `NaN`，另存同尺寸 `bool` 有效预测 mask。无效预测比例必须统计；评测协议决定其惩罚/报告方式，不能自行缩小评测区域。
- 保存适配器输出、未用 GT 对齐的预测；若生成对齐结果，写到另一个目录并记录参数及来源，不覆盖原始预测。
- 保存输出尺寸与 resize/crop 变换。评测阶段的重采样遵循统一协议。
- 显示图/视频不替代原始数组。同一片段及同组对比使用固定的显示范围并记录端点；相对与 metric 的显示单位分别标注。

## 每次运行的目录

```text
outputs/<run_id>/
  config_resolved.yaml
  metadata.json
  predictions.csv
  predictions/<sample_id>.npy
  masks/<sample_id>.npy
  metrics.json
  visualizations/
  run.log
```

视频预测可分组为 `predictions/<clip_id>/frame_000000.npy`。`predictions.csv` 每行对应一个实际输出，包含 `sample_id,clip_id,frame_index,timestamp_s,prediction_path,mask_path`，所有文件路径相对于 run 目录。

`run_id` 每次唯一，建议日期 + 方法 + 数据划分 + 序号；重跑创建新目录。正式运行工作区保持干净。示例元数据见 [metadata.example.json](../configs/metadata.example.json)：`null` 表示待填写，示例不能作为有效实验记录。

必须记录：代码 SHA、数据清单版本/哈希、checkpoint 来源及哈希、官方代码 SHA、Python/依赖环境、设备、输入/输出尺寸、预测表示/单位/方向、GT 对齐方式、有效像素规则、运行模式、计时范围和复现命令。视频的帧数、时间戳和索引应一致。

## 交接

A/B 首次交接只需一张图或一段短片的原始数组、mask、metadata、索引与命令。另一人先验证形状、帧对应关系、单位和有限值，再运行评测。
