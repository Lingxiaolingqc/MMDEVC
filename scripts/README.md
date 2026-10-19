# 命令入口

当前可运行：`python scripts/check_setup.py`，检查基础结构、CSV 表头与示例元数据，**不代表模型环境或实验已通过**。

后续按任务实现 `prepare_data.py`、`run_inference.py`、`run_refinement.py`、`evaluate.py`、`run_ablations.py`、`summarize_results.py`。业务逻辑放在 `src/p3_depth/`，这里负责配置、调用和保存；入口未实现前不在 README 声称可运行。
