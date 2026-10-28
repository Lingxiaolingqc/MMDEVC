# 导出、打包与最终交付

2026-10-28：图像实验、视频程序、应用和报告草稿已经有运行证据。自录视频尚未提供；正式 `depth_results.zip`、最终视频结论与完整个人验收指南尚未生成。下面是已实现的打包流程，不能把示例测试包当作正式提交。

## 正式视频完成后的顺序

1. 按 [采集说明](video_capture.md) 登记全部 dev/test，按 [相对视频诊断](relative_video.md) 处理 Part 1，并运行 DA metric、VDA 和公共光流。
2. 按 [开发选择](fusion_selection.md) 比较三个小候选，记录真实观察，冻结后运行全部独立 test。每段保存 E0–E5 结果并完成 `verify_fusion_suite.py`。
3. 对每段导出相对视频，以及包含 E0–E5 的米制视频。可以分成多个比较视频，避免七列挤在一个画面里；同一片段的所有米制视频采用同一色标，例如 0–10 m。
4. 在至少一个正式 test 上导出 [应用 Demo](../demo/README.md)：RGB、深度、点选 Z 读数和时间曲线。观察者检查 RGB/深度是否同步、数值与数组是否一致、无效值是否留空。读数是预测，不能凭 Demo 宣称绝对距离正确。
5. 写出困难条件索引，更新报告实际结论，最后生成并检查压缩包。

`results/video_failure_cases.csv` 由实际观察生成，表头为：

```csv
category,clip_id,start_s,end_s,observation
```

category 包含 `mirror`、`glass`、`sky`、`motion_blur`、`thin_object`。时间使用缓存保留的原始相对时间戳，观察描述应定位画面区域和具体表现。若某条件没有造成明显失败，也如实写出观察；不能为了填表虚构失败。缺少某类内容时补录诊断片段，并先确定其用途和 split，再运行模型。

## 打包计划

复制 `configs/depth_results.example.json` 到本地 `outputs/depth_results_plan.json`。`clips` 必须列出 `manifests/videos.csv` 中每一个真实片段。每条记录的结构如下，路径应替换为实际产物；dev 的 `suite` 使用所选候选，test 使用冻结配置生成的目录。

```json
{
  "clip_id": "test_static",
  "cache": "outputs/videos/test_static/cache",
  "suite": "outputs/videos/test_static/frozen_suite",
  "renders": {
    "metric_baselines": "outputs/videos/test_static/metric_baselines",
    "fusion_ablations": "outputs/videos/test_static/fusion_ablations",
    "relative_baseline": "outputs/videos/test_static/relative_rendered"
  },
  "relative_diagnostics": "outputs/videos/test_static/relative_diagnostics",
  "demo": "outputs/videos/test_static/demo"
}
```

`renders` 的标签用于压缩包目录；每个目录必须是 `render_depth_video.py` 的完整导出。米制视频合起来包含 E0、E1、E2、E3、E4 和 `vda_metric`，相对视频包含 `dav2_relative`。`demo` 对普通片段可省略，但至少一个 test 必须提供 `render_demo.py` 的完整导出。`frozen_config` 指向生成的冻结配置，`failure_cases` 指向真实困难条件索引。

在仓库根目录运行：

```powershell
$py = '.\.venv-cuda\Scripts\python.exe'
& $py scripts/package_depth_results.py --plan outputs/depth_results_plan.json --output outputs/submission/depth_results.zip --check-only
if ($LASTEXITCODE -ne 0) { throw '先修复打包检查报告的问题' }
& $py scripts/package_depth_results.py --plan outputs/depth_results_plan.json --output outputs/submission/depth_results.zip
```

入口检查完整片段覆盖、冻结证据、所选版本、预测/时间索引对应关系、固定米制色标、相对诊断和应用导出。文件清单是显式白名单，不扫描整个仓库；数据、权重、完整浮点数组和个人验收文件不打包。已有目标文件不会被覆盖。

打包后重新读取 ZIP，检查 CRC、条目数量和每个载荷文件的 SHA256；旁边生成 `depth_results.verification.json`。这些检查证明压缩包与指定产物一致，不能代替对场景身份、观察描述、报告结论或成员验收的确认。

```text
depth_results.zip
  README.txt
  manifest.json
  clips/<clip_id>/           视频、预览、时间索引、指标与模型来源
  application/<clip_id>/     RGB/深度/点选应用视频和原始数值曲线
  evidence/                 视频划分、冻结配置、开发选择和困难条件记录
```

## 已验证的示例测试包

本机已有完整示例导出时可运行：

```sh
python scripts/package_depth_results.py --plan configs/depth_results_smoke.json --output outputs/submission_check/depth_results_smoke.zip
```

这个包只验证导出与打包，允许示例素材和 provisional 参数；包内、包名及核验记录均标明范围，程序拒绝把它命名为正式的 `depth_results.zip`。真实测试记录见 [打包检查](../results/submission_packaging_checks.md)。

## 最终提交由成员确认

Spring 2027 说明要求 CVPR camera-ready 报告正文 6–8 页、参考文献另计；摘要末尾包含公开 GitHub 链接，引用全部指定阅读，报告实际消融与分类失败案例。最终报告需上传 arXiv，并在 Canvas 提交第一页写明 arXiv ID。代码保留在公开 GitHub，不另向 Canvas 上传原始代码文件；视频和应用输出作为 `depth_results.zip`。

仓库公开状态已于 2026-10-28 通过 GitHub 元数据确认；本次工作仍在功能分支。最终检查时需确认提交链接指向完整代码版本、README 可复现、PDF 与 ZIP 相互一致。没有训练步骤，因为本项目运行预训练模型并做后处理；需报告源码、权重、依赖和推理/评测命令。

助手不会替成员声称已完成独立验收。最终中文个人验收指南将放在 `outputs/personal_acceptance/`，不 commit、不 push、不装进提交包。arXiv 与 Canvas 发布以及 main 合并仍由成员确认。
