# 两人协作与实验交接

## 访问与分工

- A：Qinchuan XIAO（`@Lingxiaolingqc`）。
- B：Jun Yu Eugene YIP（`@Geney-gy-29`）。
- 仓库所有者在 GitHub 的 Settings → Collaborators 中添加 B；B 接受后才能直接推送分支。用户名已记录，邀请/接受状态尚未验证。
- 模块主责见 README。共享格式与评测规则的变更由另一人复核。

## 每个任务的流程

1. 在 `docs/task_board.md` 选择任务，写清目标、输入、输出和验收标准。
2. 保存当前工作，从最新 `main` 建一个任务分支。
3. 实现一个可检查的功能，先跑小样本。
4. 提交代码、配置、运行说明与小体积结果；查看暂存 diff，确认没有本机路径或大文件。
5. 推送分支，创建 PR；另一人检查并按说明复跑。
6. 合并后更新任务清单，下一任务从最新 `main` 建新分支。

```sh
git switch main
git pull --ff-only
git switch -c feat/stereo-baseline
# 编辑、运行检查、git add 本次需要提交的具体文件
git diff --cached
git commit -m "Add single-pair stereo baseline"
git push -u origin feat/stereo-baseline
```

分支示例：`feat/depth-evaluation`、`feat/da-v2-inference`、`feat/video-pipeline`、`feat/flow-fusion`。

`main` 保存已检查过的代码或清楚标注状态的文档。未完成工作可使用 Draft PR。发生冲突时保留双方修改意图，解决后重新运行相关检查，不用强制推送覆盖对方提交。

## PR 至少说明

- 完成的功能或修复的问题。
- 运行命令、环境与输入样本。
- 验证结果及输出所在位置。
- 尚未验证的部分和已知问题。

## GPU 实验

1. 固定代码 commit、模型来源/权重、数据清单和参数；正式结果使用干净的工作区。
2. 在独立环境中先跑小样本，再运行完整清单。
3. 每次运行使用新的 `run_id`，保留 `config_resolved.yaml`、`metadata.json`、预测索引、原始预测、指标与日志。
4. 运行期间不修改该实验使用的代码或配置；共用电脑时使用独立 checkout 与输出目录。
5. 将精选结果放到 `results/`，在 `results/run_index.csv` 记录来源，并更新实验日志。

完整数据、权重、原始预测与视频使用约定的共享存储；仓库中记录共享资料的相对位置或非敏感标识，不提交含访问令牌的下载链接。

## 每周同步

- 周初：确认各自最重要的三项任务及依赖。
- 周中：交付命令、配置和样例，检查接口。
- 周末：按结果验收，填写周总结，把已验证结论写进报告。

A 的算法由 B 跑实验检查；B 的评测与图表由 A 复核。实验失败和未提升的结果也应记录。
