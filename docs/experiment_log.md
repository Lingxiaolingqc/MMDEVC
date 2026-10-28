# 实验记录

已验证记录索引见 [run_index.csv](../results/run_index.csv)。完整原始预测、配置和元数据仅保存在本地 `outputs/`。

- `stereo_all_week1_20261021`：dev 两场景比较三组参数后冻结 block size 3；全部 15 对保存与复算通过。见 [结果与失败例子](../results/stereo_week1.md)。
- `unidepth_nyuv2_0003_week1_20261021`：官方 ViT-S、CPU、train 开发样本、无 GT 对齐；原始保存结果指标复算一致，GT=prediction 检查通过。见 [单图结果](../results/unidepth_week1.md)。
- 独立 B 复核、B 的 DA V2 真实预测交接、视频与融合实验均待完成。

以下模板供后续每次运行填写；不能把示例当结果。

## <run_id>：<实验目的>

- 执行人 / 日期：
- 代码 commit / 工作区是否干净：
- 数据清单版本 / 开发或测试：
- 模型 / 官方代码 commit / checkpoint 与哈希：
- 环境 / 设备 / 输入尺寸 / 运行模式：
- 命令 / 完整配置位置：
- 输出表示 / 单位 / 对齐规则 / 评测协议：
- 原始结果位置 / 指标 / 有效覆盖率 / 效率：
- 代表性 sample_id 或 clip_id + frame_index：
- 观察与结论（区分已验证与推测）：
- 失败 / 待核查项 / 下一步：
- 交叉复核人及结果：
