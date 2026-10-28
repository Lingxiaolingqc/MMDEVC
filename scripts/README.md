# 命令入口

当前可运行：

- `python scripts/check_setup.py`：检查基础结构、CSV 表头与示例元数据。
- `python scripts/inspect_middlebury.py --sample Adirondack`：检查一个场景并生成预览。
- `python scripts/inspect_middlebury.py --all --write-manifest`：检查 15 组训练数据，更新清单并保留手工填写的 split/notes。
- 添加 `--save-arrays` 可保存 GT 派生的米制深度及布尔 mask；`--no-preview` 可跳过画图。

这两类检查**不代表模型环境或实验已通过**。Middlebury 依赖、路径约定与输出见 [数据说明](../docs/middlebury_data.md)。

后续按任务实现 `prepare_data.py`、`run_inference.py`、`run_refinement.py`、`evaluate.py`、`run_ablations.py`、`summarize_results.py`。业务逻辑放在 `src/p3_depth/`，这里负责配置、调用和保存；入口未实现前不在 README 声称可运行。
