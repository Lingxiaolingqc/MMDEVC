# 精选结果

已完成 [Adirondack StereoSGBM 首次开发基线](stereo_adirondack_20261020.md)：原始视差、米制深度、有效 mask、误差与覆盖率均已保存。精选 CSV 在 `tables/`，预览图在 `figures/`，参数与元数据在 `runs/`；B 的独立复现待完成。

`tables/middlebury_data_check.csv` 是此前的 15 场景 GT 数据检查统计，来源与运行说明见 [Middlebury 数据准备](../docs/middlebury_data.md)；它不是模型精度表。只提交需要共享的小体积结果。

每个结果必须在 `run_index.csv` 记录代码 commit、配置、原始输出的非敏感共享位置与复核人。完整预测和视频放在本地 `outputs/` 或约定的共享存储。
