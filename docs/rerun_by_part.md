# 按 Part 重跑：成员操作，助手逐步指导

这份说明对应新的自动可视化和默认输出目录。**完整重跑尚未执行**。实际操作时每次只完成一个阶段，把终端结果和预览交给助手检查，再进入下一阶段；不要一次粘贴整页。每条命令报错后先停下，不删除结果或手改哈希。

2026-11-07 已将旧 outputs 与当时受 Git 跟踪的复现材料归档至仓库外的 `../MMDEVC_archives/outputs_before_20261107.zip`，8,307 个文件逐项校验通过，旁边的 `.verification.json` 记录 ZIP 哈希。旧输出仍原地保留。新结果验收、报告更新和提交包检查完成后，再单独核对旧目录清单并清理。

## 0. 环境检查

在 **MMDEVC 仓库根目录**打开 PowerShell，先只执行：

```powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -X utf8 scripts/check_setup.py
& $py -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA GPU')"
```

本机已有数据、源码、权重和 CUDA 环境；换电脑时先按 [环境说明](../environments/cuda.md)准备，不复制 Windows venv。后续命令假定 `configs/paths.local.yaml` 的 `output_root: outputs`，三个数据根目录也已经填写正确。若使用别的输出根目录，视频清单和打包计划必须一起调整，不能只改一处。

新目录规则集中在 [output_layout.yaml](../configs/output_layout.yaml)，显式 `--output-dir` 仍优先。每个模型的 raw 数组保持原值；着色、裁剪显示和拼图使用副本。模型计时不包含可视化耗时。

```text
outputs/
├── shared/video_cache/{clip}/                 相同 RGB 帧与原始时间戳
├── part1_baseline/
│   ├── stereo/samples/Adirondack/             单场景检查
│   ├── stereo/all/{scene}/                    15 场景原始数组、指标、PNG
│   ├── stereo/course_evaluation/              课程 MAE 与 coverage
│   └── dav2_relative/{dev|test}/{clip}/        DA 原始数组与独立视频
├── part2_models/
│   ├── unidepth/{dev|test}/                   NYU 逐图数组、PNG、浏览页
│   │   └── course_evaluation/                 test 的课程主指标
│   └── vda_relative/{dev|test}/{clip}/         VDA 原始数组与独立视频
├── part3_improvement/
│   ├── flow/{dev|test}/{clip}/                光流、可靠区域及视频
│   ├── dev_candidates/{alpha050|070|085}/{clip}/
│   ├── selection/                            开发证据与冻结配置
│   └── test/{clip}/                          E0–E5 指标和对比视频
├── application/{clip}/                       背景虚化演示
└── submission/                               图表、案例索引、最终 ZIP/PDF
```

模型目录内的 `visualizations/comparison.mp4` 是 **RGB + 该模型独立预测**，`preview.png` 是首/中/末帧预览。Part 1 的命令不会加载 VDA 或运行融合。NYU 的 `visualizations/index.html` 可以浏览该次全部单图输出。

## 1. Part 1：StereoSGBM

先只跑一个场景：

```powershell
& $py -X utf8 scripts/part1_baseline/run_stereo_baseline.py --sample Adirondack
```

检查 `outputs/part1_baseline/stereo/samples/Adirondack/visualizations/Adirondack.png` 及同目录的指标。确认后再运行完整 15 场景及复算：

```powershell
& $py -X utf8 scripts/part1_baseline/run_stereo_suite.py --phase all
& $py -X utf8 scripts/part1_baseline/run_stereo_suite.py --phase verify
& $py -X utf8 scripts/part1_baseline/audit_course_stereo.py
```

默认使用已选定的 `configs/stereo_sgbm_selected.yaml`，不重新用 evaluation 场景调参。每个场景自动生成八面板 PNG；课程主表在 `stereo/course_evaluation/`。输出已存在时命令拒绝覆盖。

## 2. Part 2：UniDepth / NYUv2

先跑 3 张 train-side dev 图像，确认输入、输出和显存：

```powershell
& $py -X utf8 scripts/part2_models/run_nyuv2.py --model unidepth --subset dev --device cuda --precision fp16
```

打开 `outputs/part2_models/unidepth/dev/visualizations/index.html`。每张 PNG 包含 RGB、GT、预测及课程区域的绝对误差，深度显示范围固定 0–10 m。PNG 的显示范围不会截断保存的原始 `.npy`。

确认后，完整运行 654 张 test：

```powershell
& $py -X utf8 scripts/part2_models/run_nyuv2.py --model unidepth --subset test --device cuda --precision fp16
& $py -X utf8 scripts/part2_models/evaluate_course_nyu.py
```

预期有 654 份预测、记录和 PNG；课程主成绩在 `outputs/part2_models/unidepth/test/course_evaluation/summary.json`。`run_nyuv2.py` 的历史七指标作为补充保留，报告使用 `course_evaluation` 的 AbsRel、RMSE、δ1。NYU 全部可视化在推理完成后自动生成，因此中途看不到全部 PNG 属于正常现象。

## 3. 准备正式视频输入

新运行使用 [videos_current.csv](../manifests/videos_current.csv)，旧 `videos.csv` 保留以验证历史冻结记录。两个清单使用相同原视频、split 和源哈希，只有缓存位置不同。**这次是相同素材的复现，不是新的未见测试集。**

本机已有 `data/videos/raw/dev_tokyo_100.mp4` 和 `course_rollercoaster_70.mp4`，不需要下载或拍摄新视频。准备两个缓存：

```powershell
$source = "https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/assets/example_videos"
& $py -X utf8 scripts/prepare_video.py --video data/videos/raw/dev_tokyo_100.mp4 --clip-id dev_tokyo_100 --scene-group tokyo_walk --split dev --scene-type camera_motion --original-fps --source-kind public_example --source-url "$source/Tokyo-Walk_rgb.mp4" --expected-frames 100 --max-frames 100 --max-long-edge 640 --registered
& $py -X utf8 scripts/prepare_video.py --video data/videos/raw/course_rollercoaster_70.mp4 --clip-id course_rollercoaster_70 --scene-group davis_rollercoaster --split test --scene-type camera_motion --original-fps --source-kind course_example --source-url "$source/davis_rollercoaster.mp4" --expected-frames 70 --max-frames 100 --max-long-edge 640 --registered
```

`--registered` 核对现有登记，不再追加清单。预期 Tokyo 100 帧 / 15 fps；rollercoaster 全部 70 帧 / 24 fps；不重复帧伪造 100 帧。与课程“前 100 帧”的差异继续在报告说明。此阶段只解码 test 原视频，不进行选参。

## 4. Part 1 / Part 2：开发视频的独立模型输出

```powershell
$dev = "dev_tokyo_100"
$cache = "outputs/shared/video_cache/$dev"
$da = "outputs/part1_baseline/dav2_relative/dev/$dev"
$vda = "outputs/part2_models/vda_relative/dev/$dev"
$flow = "outputs/part3_improvement/flow/dev/$dev"
& $py -X utf8 scripts/part1_baseline/run_framewise.py --cache $cache --input-size 364 --device cuda --precision fp16
& $py -X utf8 scripts/part2_models/run_video.py --cache $cache --input-size 364 --device cuda --precision fp16
```

两条命令各自生成 `depths.npy`、`valid.npy`、元数据、独立 MP4 和 PNG。先播放 `$da/visualizations/comparison.mp4`，再播放 `$vda/visualizations/comparison.mp4`。两个模型都输出相对深度，颜色不能解释为米；各片段色标固定，不逐帧重新拉伸。

这里保持原实验 input-size=364、FP16，避免把结构整理和模型设置变化混在一起。

## 5. Part 3：开发实验和复算

```powershell
& $py -X utf8 scripts/part3_improvement/prepare_flow.py --cache $cache
foreach ($alpha in @("050", "070", "085")) {
    $suite = "outputs/part3_improvement/dev_candidates/alpha$alpha/$dev"
    & $py -X utf8 scripts/part3_improvement/run_fusion_suite.py --cache $cache --baseline $da --reference-vda $vda --flow $flow --config "configs/fusion_relative$alpha.yaml"
    if ($LASTEXITCODE -ne 0) { throw "Suite failed: $alpha" }
    & $py -X utf8 scripts/part3_improvement/verify_fusion_suite.py --suite $suite --cache $cache --baseline $da --reference-vda $vda --flow $flow
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $alpha" }
}
```

每个候选目录包含 E0–E4 数组与独立视频、E0–E5 指标，另有：

- `visualizations/primary/comparison.mp4`：RGB / DA / E4 / VDA。
- `visualizations/ablations/comparison.mp4`：RGB / E0 / E1 / E2 / E3 / E4。
- E5 直接引用 Part 2 的原始 VDA，不再复制一份大数组或影片。

光流目录也自动生成 `visualizations/flow.mp4`，展示 RGB、光流方向/幅度和公共可靠区域。首帧没有上一帧，用灰色表示。

**在此停下，让助手带你看三个候选的视频和 comparison.csv。** 我们记录稳定性、运动响应、遮挡/边缘，而不是仅选最低时序变化值。原方案选 α=0.70；本轮主要核实同设置复现，不利用已经看过的 test 来重新选择参数。

助手会根据实际检查整理 `outputs/part3_improvement/development_review.json`，字段参照 [review 模板](fusion_selection_review.example.json)。未检查时保持 pending，不复制旧观察冒充本次验收。review 不放进尚未创建的 `selection/`，否则冻结命令会因目录已存在而停止。

review 完成且确认保留 alpha070 后，执行：

```powershell
$candidates = "outputs/part3_improvement/dev_candidates"
& $py -X utf8 scripts/part3_improvement/freeze_fusion_config.py --candidate "alpha050=$candidates/alpha050/$dev" --candidate "alpha070=$candidates/alpha070/$dev" --candidate "alpha085=$candidates/alpha085/$dev" --selected alpha070 --review outputs/part3_improvement/development_review.json
```

预期生成 `outputs/part3_improvement/selection/fusion_frozen.yaml` 和对应证据。若选择改变，先同步打包计划中的 dev suite，再运行 test；不能只手改 frozen 状态。

## 6. 冻结后运行 test 和应用

```powershell
$clip = "course_rollercoaster_70"
$cache = "outputs/shared/video_cache/$clip"
$da = "outputs/part1_baseline/dav2_relative/test/$clip"
$vda = "outputs/part2_models/vda_relative/test/$clip"
$flow = "outputs/part3_improvement/flow/test/$clip"
$suite = "outputs/part3_improvement/test/$clip"
& $py -X utf8 scripts/part1_baseline/run_framewise.py --cache $cache --input-size 364 --device cuda --precision fp16
& $py -X utf8 scripts/part2_models/run_video.py --cache $cache --input-size 364 --device cuda --precision fp16
& $py -X utf8 scripts/part3_improvement/prepare_flow.py --cache $cache
& $py -X utf8 scripts/part3_improvement/run_fusion_suite.py --cache $cache --baseline $da --reference-vda $vda --flow $flow
& $py -X utf8 scripts/part3_improvement/verify_fusion_suite.py --suite $suite --cache $cache --baseline $da --reference-vda $vda --flow $flow
& $py -X utf8 scripts/application/render_blur_demo.py --cache $cache --baseline $da --improved "$suite/E4"
```

test 默认读取第 5 步生成的冻结配置，检查开发证据后运行。影片应有 70 帧；时序评测有 69 对。应用视频在 `outputs/application/course_rollercoaster_70/demo.mp4`。不要根据这一步的结果反过来修改候选选择。

## 7. 报告、打包和最后清理

先由助手陪你查看新影片、指标以及三个困难案例，记录当前真实观察至 `outputs/submission/failure_cases.csv`。可沿用历史案例的片段位置，但观察内容需要对照新输出；新打包计划故意不自动引用旧案例结论。

之后执行：

```powershell
& $py -X utf8 scripts/build_course_figures.py
& $py -X utf8 scripts/package_depth_results.py --check-only
& $py -X utf8 scripts/package_depth_results.py
```

图表在 `outputs/submission/figures/`，视频包在 `outputs/submission/depth_results.zip`。`--check-only` 先核对新来源、冻结证据和输出文件；通过后才写 ZIP。报告 PDF 仍需助手根据新图表、指标、硬件与代码版本更新并重新编译，不能把旧 PDF 改名当作新结果。

全部通过后，我们再列出旧目录与新目录的对应关系，核对归档仍可读，然后清理旧 outputs、旧分类副本及这次小样本检查目录。数据集、checkpoint、源码、个人验收文档不属于输出清理对象；个人验收文档也不进入 Git 或提交 ZIP。

## 只补可视化与中断处理

若原始数组已完整保存、仅可视化失败，可以沿用相同参数并增加 `--visualize-only`：

```powershell
& $py -X utf8 scripts/part1_baseline/run_framewise.py --cache outputs/shared/video_cache/dev_tokyo_100 --visualize-only
& $py -X utf8 scripts/part2_models/run_video.py --cache outputs/shared/video_cache/dev_tokyo_100 --visualize-only
& $py -X utf8 scripts/part2_models/run_nyuv2.py --model unidepth --subset test --visualize-only
```

这些模式不加载模型，只读取已保存预测。光流和融合入口也支持同参数加 `--visualize-only`。已经完整的视频会校验来源与文件哈希；不完整的可视化目录先交给助手诊断，不能为修复图片而删除 raw 数组。NYU 可以重建 PNG/浏览页，原始预测不变。

NYU 推理中断可在**同一代码版本、配置、设备、样本和输入**下原命令加 `--resume`。旧代码版本的未完成输出不能跨本次改造直接续跑。其他入口默认拒绝覆盖已有运行；不要自行加覆盖参数或清空整个 outputs。

旧结果的复查仍使用 [历史协议](course_protocol_20261102.md)中的显式路径与 `configs/course_export_plan.json`。本页是新运行入口，旧 `organize_part_results.py` 仅用于历史副本核验，新流程不再额外复制一层 `outputs/by_part/`。
