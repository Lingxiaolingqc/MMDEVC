# 补充实验：按 Part 运行，由助手分析、成员观看

2026-11-08 制定。前提是已完成 [主线重跑](rerun_by_part.md)：15 个 Middlebury 场景、654 张 NYUv2、Tokyo 与指定过山车的 DA/VDA、光流、alpha070 E0–E5，以及保存结果复算。补充入口读取这些已有结果，不加载深度模型。

一次只执行一个阶段。成员运行命令并播放图片/视频；助手读取保存的数组、表格、元数据完成全部定量分析。成员给出视觉观察后，助手定位到同一帧/时间区间，检查 RGB、预测、色标、有效区域和局部图，再记录“成员观察、已核验证据、尚未确定的解释”。视觉变化不自动等同于深度准确率改善。

本页是可共享的运行协议。个人验收笔记、视觉观察记录和大输出保存在被 Git 忽略的 outputs 内，不随代码提交。原主实验、冻结配置及源数组保留；补充实验均标为事后分析，不成为新的盲测或自动替换主结果。

## 0. 进入仓库

从 MMDEVC 仓库根目录的 PowerShell 执行：

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py --version
```

这些步骤使用 CPU 后处理与已有数组。无需重跑 DA、VDA 或 UniDepth。新输出目录必须不存在；若报错，停下交给助手定位，不清空旧目录或手动改哈希。

## 1. Part 1：自定义左右一致性检查

```powershell
& $py -X utf8 scripts/part1_baseline/analyze_stereo_lr.py
if ($LASTEXITCODE -ne 0) { throw "LR analysis failed" }
```

默认输出 `outputs/part1_baseline/stereo/lr_analysis/`：

- `summary.csv`：all_15_descriptive、dev、evaluation 的检查前后汇总；MAE 按场景等权，coverage 按像素汇总。
- `per_scene.csv`：15 场景 × 两种 GT 区域 × 自定义 LR 开/关，共 60 行。
- `rejections.csv`：每个区域被排除的像素数量、其中误差 >1px 的数量/比例、无法比较右图和残差超阈值的数量。比例的分母是有 GT 的被排除像素。
- `visualizations/{scene}.png`：RGB、GT、检查前、检查后、新增无效区域、被排除预测的误差类别；每张可放大查看。
- `rejected_masks/` 与 `metadata.json`：可复核的拒绝区域、来源、哈希与定义。

先把终端结果发给助手，由助手看表并选择需要观看的场景。成员只需描述哪个结构被排除、缺失是否明显。助手会核验是否确由额外 LR 检查造成，而不是匹配器本来就没有输出。

对照固定 OpenCV 内部 `disp12MaxDiff` 和全部匹配结果，仅切换自定义过滤。检查后 MAE 是保留像素的条件误差，必须与有效预测比例同看。红色代表被排除但原误差 ≤1px，并不表示数学上完全正确；绿色代表被排除且原误差 >1px。

## 2. Part 2：当前 UniDepth 的 NYUv2 边缘评价

```powershell
& $py -X utf8 scripts/part2_models/evaluate_nyuv2_boundaries.py --protocol course_clipped
if ($LASTEXITCODE -ne 0) { throw "Boundary evaluation failed" }
Invoke-Item "outputs/part2_models/unidepth/test/boundary_course_clipped/visualizations/index.html"
```

使用全部 654 张当前预测，固定课程 crop、GT 范围和预测 clipping，不做 GT 尺度拟合。默认不带 `--protocol` 仍保留旧版 legacy 行为，以免旧命令静默改变含义。

默认输出 `outputs/part2_models/unidepth/test/boundary_course_clipped/`，包含 `per_image.csv`、`summary.json`、`metadata.json` 及图片浏览页。六张预览按最低 F1 和中位 F1 附近确定性选择，不代表随机抽样；助手会读全体指标而不是只看六张。

边缘定义为四邻域最大绝对 log-depth 差 > log(1.1)，在固定 GT 域腐蚀一像素后比较，允许欧氏距离 2px 的匹配。Precision/Recall 按全部边缘像素汇总，另报 macro F1。这是本项目补充诊断定义，不是课程的 AbsRel/RMSE/δ1，也不宣称复现某论文的边缘分数。

成员观看 RGB 上叠加的 GT 绿边和预测红边，描述漏边、多余边或位置偏移。助手核对对应指标、裁剪范围与原始预测。

## 3. Part 3：历史来源对照

E1 同位置平均；E2 光流对齐；E3 再加硬过滤；E4 再加四项软置信度。每种方法分别比较上一帧原始预测（pairwise）和上一帧融合结果（recursive）。保持相同参数与计算规则；E4 的实际像素权重允许随历史内容变化。

### 3A. Tokyo 开发视频

```powershell
$clip = "dev_tokyo_100"
$split = "dev"
$reference = "outputs/part3_improvement/dev_candidates/alpha070/$clip"
$cache = "outputs/shared/video_cache/$clip"
$da = "outputs/part1_baseline/dav2_relative/$split/$clip"
$flow = "outputs/part3_improvement/flow/$split/$clip"
$out = "outputs/part3_improvement/supplementary/$split/$clip/history"
& $py -X utf8 scripts/part3_improvement/run_supplementary_ablations.py --cache $cache --baseline $da --flow $flow --reference-suite $reference --experiment history
if ($LASTEXITCODE -ne 0) { throw "History controls failed" }
& $py -X utf8 scripts/part3_improvement/verify_supplementary_ablations.py --run $out
if ($LASTEXITCODE -ne 0) { throw "History verification failed" }
```

输出 `comparison.csv` 有九行：原 E0、四种原 recursive、四种新 pairwise。已有参考预测只读复用。四种新配置各自保存 `depths.npy`、`valid.npy`、`history_weight.npy`、指标、逐对帧表和元数据。

助手先分析表格，再指导逐个观看：

```powershell
Invoke-Item "$out/visualizations/E1_pairwise/comparison.mp4"
```

之后把 E1 换为 E2、E3、E4，逐个打开。每部视频三列：RGB / 原 recursive / 新 pairwise，使用相同 DA 整段色标。观察同一移动物体的闪烁、重复轮廓、遮挡后残留；记录时间区间或画面顶部帧号，暂不凭单个指标选赢家。

### 3B. 指定过山车视频

完成开发视频观察后，重设变量并执行：

```powershell
$clip = "course_rollercoaster_70"
$split = "test"
$reference = "outputs/part3_improvement/test/$clip"
$cache = "outputs/shared/video_cache/$clip"
$da = "outputs/part1_baseline/dav2_relative/$split/$clip"
$flow = "outputs/part3_improvement/flow/$split/$clip"
$out = "outputs/part3_improvement/supplementary/$split/$clip/history"
& $py -X utf8 scripts/part3_improvement/run_supplementary_ablations.py --cache $cache --baseline $da --flow $flow --reference-suite $reference --experiment history
if ($LASTEXITCODE -ne 0) { throw "History controls failed" }
& $py -X utf8 scripts/part3_improvement/verify_supplementary_ablations.py --run $out
if ($LASTEXITCODE -ne 0) { throw "History verification failed" }
```

同样先由助手分析，再逐个观看四部影片。指定文件只有 70 帧，保留原始帧率和 69 对时序比较；课程要求与文件长度的差异沿用已有记录。这个片段已被观察，新增对照属于事后分析。

## 4. Part 3：E4 置信度逐项去除

固定 recursive，不与历史来源做所有组合。四个新配置：`E4_no_photo`、`E4_no_flow`、`E4_no_depth`、`E4_no_edge`。去掉的是 soft confidence 的相应指数项；RGB 和前后向光流的硬过滤仍保留，包括 `no_photo` 和 `no_flow`。

先只执行第 3A 代码块中的变量赋值行（从 `$clip` 到 `$out`），不重复运行 history 命令，然后执行下面命令。观看并核验后，再只执行第 3B 的变量赋值行，对指定视频执行同一命令：

```powershell
$out = "outputs/part3_improvement/supplementary/$split/$clip/confidence"
& $py -X utf8 scripts/part3_improvement/run_supplementary_ablations.py --cache $cache --baseline $da --flow $flow --reference-suite $reference --experiment confidence
if ($LASTEXITCODE -ne 0) { throw "Confidence controls failed" }
& $py -X utf8 scripts/part3_improvement/verify_supplementary_ablations.py --run $out
if ($LASTEXITCODE -ne 0) { throw "Confidence verification failed" }
```

七行对照表：原 E0、E3、完整 E4，以及四种新配置。四部影片在 `visualizations/{配置名}/comparison.mp4`，每部四列 RGB / E3 / 完整 E4 / 去掉一项。助手分析指标和权重后，成员观看哪些区域改变，助手再验证具体证据。某项没有显示收益也如实记录。

两组对照的平均历史权重、历史使用比例、偏离 DA 的幅度和归一化时序变化均是诊断量，不是深度准确率或画质分数。公共评测区域固定于 RGB/flow；区域外的遮挡与轮廓问题仍要看影片。原参考与新配置不是同一轮性能 benchmark，速度是已有模型时间、光流时间和各自融合时间相加的估计。

## 5. Application：观看已生成的阈值对照

```powershell
Invoke-Item "outputs/application_exploratory/course_rollercoaster_70/transition010_030/settings_comparison.mp4"
```

上排旧阈值、下排新阈值；左列 DA、右列 E4。目标是保留乘客、车厢、轨道和支架，虚化远背景。成员描述误虚化、背景效果及时间变化；助手核对相同预测/参数下的比较。只有观看后再决定提交演示版本。若该文件尚未生成，按 [应用参数对照](application_settings.md)运行。

## 6. 助手完成定量总结与视觉核验

成员反馈可以只写“配置名、约第几秒/帧、画面位置、看到什么”，无需自行计算指标或证明原因。不确定的观察照样保留。

助手逐项检查来源与复算结果，读取全量表格，按同场景同区域做比较，再选取用户提及的连续帧/局部区域核对。观察与机制解释分开记录；没有充分证据就保留“未确定”。个人观察/验收记录放在 outputs 中，不自动写入报告。

之后才确定三例真实困难案例、当前效率表、报告图和结论。报告与 ZIP 仍需依照实际验收结果更新，新增消融不会自动进入现有提交计划。主结果继续保留 E4 原冻结配置；若提出新方法，另立开发选择与实验记录。

## 中断与复查

融合数组和指标已保存、只有视频导出失败时，原补充运行命令可加 `--visualize-only`。它校验输入后只补视频，不重新融合；已完整的可视化校验后复用。不完整的单个视频目录应先由助手检查，不能为补视频删除预测。

补充复算默认写新 `verification.json`，已有凭据不覆盖。需要再次复查时指定新的 `--verification-output`。LR/边缘分析重跑使用新的 `--output-dir`，保留原统计。不要连续运行整份指南；每阶段经助手分析和成员观看后再进入下一阶段。
