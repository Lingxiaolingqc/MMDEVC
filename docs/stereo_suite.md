# Middlebury 15 场景基线

开发场景在调参前固定为 Adirondack、Pipes；其余 13 场景为 evaluation。Motorcycle/MotorcycleE、Piano/PianoL、Playtable/PlaytableP 均在同一侧。官方 trainingQ 是有 GT 的实验数据，本项目的 evaluation 表示没有用于参数选择，不是官方隐藏测试集。

只比较 block size 5、3、7 共 3 组配置，其他参数沿用单样本基线；P1/P2 随 block size 按现有公式变化。按两个开发场景的全 GT Bad>1px（包括无效预测惩罚）均值选择；并列时优先接近 5，再取较小值。调参过程中不加载 evaluation 场景。

```sh
python scripts/run_stereo_suite.py --phase tune --output-dir outputs/stereo_tuning_week1
# 把该运行的 selected.yaml 保存为 configs/stereo_sgbm_selected.yaml 并提交后，再做固定配置运行。
python scripts/run_stereo_suite.py --phase all --output-dir outputs/stereo_all_week1
python scripts/run_stereo_suite.py --phase verify --output-dir outputs/stereo_all_week1
```

每场景保留与 [单样本基线](stereo_baseline.md) 相同的原始预测、mask、视差、指标、图和元数据。`per_scene.csv` 含 15×2 行；`summary.csv/json` 分别汇总 dev、evaluation、包含开发场景的全部 15 对。深度误差按场景取宏平均，覆盖率与 Bad 同时报告宏平均或明确标记的像素加权结果；零覆盖场景数单列，不把空结果当零误差。

`verify` 从保存数组重新计算全部指标，核验数据和产物哈希、合并表及汇总。完整原始输出留在 `outputs/`，已验证的精选表格与三个失败案例见 [Week 1 结果](../results/stereo_week1.md)。完整 15 对汇总不得称为全部独立测试。
