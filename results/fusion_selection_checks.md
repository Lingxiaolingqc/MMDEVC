# 开发选择与冻结流程检查

日期：2026-10-28。当前没有自录视频，本记录不代表正式参数已经选出。

- 完整回归：`python -m unittest discover -s tests -v`，58 项通过，含本机 HTTP Demo。日志保留在本地 `outputs/fusion_selection_tests_20261028.log`。
- 新增 7 项选择检查使用临时小型文件夹，验证独立 test 可通过；开发片段不能冒充 test；provisional/参数变更被拒绝；场景或原文件重复、漏评 dev、候选输入不一致被拒绝；示例素材不能当正式 dev；冻结后更改结果、观察或清单被拒绝；未复算/未完成观察记录被拒绝。
- 测试中的文件和观察文字均为明确标注的测试夹具，不是实拍素材、模型实验或成员评价。
- 官方示例的既有 24 帧结果重新通过 `verify_fusion_suite.py`：23 对 RGB 光流/公共 mask、E0–E4 浮点融合与权重、E0–E5 指标均复算一致。核验记录在本地 `outputs/video_smoke_20261028/fusion_364/verification.json`；scope 仍为 `pipeline_check_not_final_evaluation`。
- `freeze_fusion_config.py --help` 可运行。正式冻结要求已登记的 dev/test、全部候选在全部 dev 上的核验结果，以及包含稳定性、运动响应、遮挡/边缘的观察记录；参数从选定候选结果提取。

正式选择、冻结配置与测试效果仍待自录素材。程序校验只能约束记录与文件的一致性，无法替代对场景归属和观察内容的真实确认。操作见 [开发选择与测试前冻结](../docs/fusion_selection.md)。
