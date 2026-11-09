# 按 Part 重跑：成员操作，助手逐步指导

主线已有输出后的 LR、边缘评价、历史来源与置信度消融，使用 [补充实验指南](supplementary_experiments.md)。该指南约定成员运行/观看，助手承担全部定量分析并核验视觉观察。

这份说明对应新的自动可视化和默认输出目录。**按课程 5.1 → 5.2 → 5.3 的顺序逐项执行和验收**。实际操作时每次只完成一个阶段，把终端结果和预览交给助手检查，再进入下一阶段；不要一次粘贴整页。每条命令报错后先停下，不删除结果或手改哈希。

2026-11-07 已将旧 outputs 与当时受 Git 跟踪的复现材料归档至仓库外的 `../MMDEVC_archives/outputs_before_20261107.zip`，8,307 个文件逐项校验通过，旁边的 `.verification.json` 记录 ZIP 哈希。旧输出仍原地保留。新结果验收、报告更新和提交包检查完成后，再单独核对旧目录清单并清理。


顺序固定为：StereoSGBM → 指定视频逐帧 DA V2 → UniDepth / NYUv2 → 指定视频 VDA 与 DA 对比 → Part 3 光流融合 → 应用 → 报告与提交。额外训练建议不进入当前验收流程。已完成的步骤保留结果，核对后直接复用，不因顺序调整重新推理。

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

## 2. Part 1：指定视频逐帧 DA V2

课程 Part 1 的 Stereo 后面紧接 Monocular Frame Baseline。先准备指定视频，再逐帧运行 DA V2-Small relative；UniDepth 在下一步。

新运行使用 [videos_current.csv](../manifests/videos_current.csv)，旧 videos.csv 保留以验证历史冻结记录。源文件和 split 不变，只有缓存路径改变。本轮是同一素材的复现，不是新的未见测试集。

本机已有原视频。先建立共享缓存：

```powershell
$source = "https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/assets/example_videos"
& $py -X utf8 scripts/prepare_video.py --video data/videos/raw/course_rollercoaster_70.mp4 --clip-id course_rollercoaster_70 --scene-group davis_rollercoaster --split test --scene-type camera_motion --original-fps --source-kind course_example --source-url "$source/davis_rollercoaster.mp4" --expected-frames 70 --max-frames 100 --max-long-edge 640 --registered
```

预期为 70 个真实帧、原始 24 fps；不重复或插帧。官方完整片段不足任务书的 100 帧，继续采用用户已确认的全部 70 帧，并在报告说明。`--registered` 核对既有登记，不追加清单。

缓存检查通过后运行：

```powershell
$clip = "course_rollercoaster_70"
$cache = "outputs/shared/video_cache/$clip"
& $py -X utf8 scripts/part1_baseline/run_framewise.py --model dav2_relative --cache $cache --input-size 364 --device cuda --precision fp16
```

打开 `outputs/part1_baseline/dav2_relative/test/course_rollercoaster_70/visualizations/comparison.mp4`，检查 RGB 与 DA 相对预测、结构、边缘和闪烁；同目录有 preview.png。原始数组和有效区域在模型目录的 depths.npy、valid.npy。现在不运行 VDA 或融合。

这里的 test 是指定视频在项目清单中的角色。固定预训练基线可以先运行和展示；Part 3 改进的参数仍只在开发片段确定，冻结后才应用到指定片段。

## 3. Part 2：UniDepth / NYUv2

若此前已经运行成功，先让助手检查已有结果；不要再次执行同一输出目录的推理命令。

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

## 4. Part 2：指定视频 VDA 与基线对比

在课程 Part 2 的单图 metric 评测之后运行 Video Depth Anything。复用 Part 1 的同一缓存及 DA 输出：

```powershell
$clip = "course_rollercoaster_70"
$cache = "outputs/shared/video_cache/$clip"
$da = "outputs/part1_baseline/dav2_relative/test/$clip"
$vda = "outputs/part2_models/vda_relative/test/$clip"
& $py -X utf8 scripts/part2_models/run_video.py --model vda_relative --cache $cache --input-size 364 --device cuda --precision fp16
& $py -X utf8 scripts/render_depth_video.py --cache $cache --run $da --run $vda --output-dir "outputs/part2_models/video_comparison/$clip"
```

VDA 目录自动包含独立视频；最后一条生成 RGB / DA / VDA 三列比较，尚不包含 Part 3。两个模型同帧、同尺寸、FP16；各自整段固定色标，颜色不代表共享的米制尺度。实际分辨率和资源由元数据记录。

先验收 frame-wise 与 video-based 的差异，再进入 Part 3；不把额外训练建议插入这里。

## 5. Part 3：开发、消融和冻结

Parts 1 和 2 已验收后，开始课程 Direction A。Tokyo 是既有方案的独立开发素材，用于冻结融合参数，不替代指定的 rollercoaster。此处补充运行开发素材上的两个固定模型，以供 Part 3 候选复用。

```powershell
$source = "https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/assets/example_videos"
& $py -X utf8 scripts/prepare_video.py --video data/videos/raw/dev_tokyo_100.mp4 --clip-id dev_tokyo_100 --scene-group tokyo_walk --split dev --scene-type camera_motion --original-fps --source-kind public_example --source-url "$source/Tokyo-Walk_rgb.mp4" --expected-frames 100 --max-frames 100 --max-long-edge 640 --registered
```

预期为前 100 帧、原始 15 fps。接着生成开发素材的模型预测：

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

预期生成 `outputs/part3_improvement/selection/fusion_frozen.yaml` 和对应证据。若 dev 选择改变，应从对应候选重新生成冻结凭据，再运行 test；不能只手改 frozen 状态。

## 6. Part 3：冻结后评估指定视频

复用 Parts 1/2 已保存的指定视频 DA/VDA 输出，不再运行模型推理。

```powershell
$clip = "course_rollercoaster_70"
$cache = "outputs/shared/video_cache/$clip"
$da = "outputs/part1_baseline/dav2_relative/test/$clip"
$vda = "outputs/part2_models/vda_relative/test/$clip"
$flow = "outputs/part3_improvement/flow/test/$clip"
$suite = "outputs/part3_improvement/test/$clip"
& $py -X utf8 scripts/part3_improvement/prepare_flow.py --cache $cache
& $py -X utf8 scripts/part3_improvement/run_fusion_suite.py --cache $cache --baseline $da --reference-vda $vda --flow $flow
& $py -X utf8 scripts/part3_improvement/verify_fusion_suite.py --suite $suite --cache $cache --baseline $da --reference-vda $vda --flow $flow
```

test 默认读取第 5 步生成的冻结配置，检查开发证据后运行。影片应有 70 帧；时序评测有 69 对。不要根据这一步的结果反过来修改候选选择。

## 7. 应用：背景虚化

Part 3 验收后生成课程要求的简单应用：

```powershell
$clip = "course_rollercoaster_70"
& $py -X utf8 scripts/application/render_blur_demo.py --cache "outputs/shared/video_cache/$clip" --baseline "outputs/part1_baseline/dav2_relative/test/$clip" --improved "outputs/part3_improvement/test/$clip/E4"
```

应用视频在 `outputs/application/course_rollercoaster_70/demo.mp4`，展示原 DA 与冻结后的 E4 在同参数背景虚化中的效果。

若需要减少近中景结构被误虚化，按 [应用参数对照](application_settings.md) 在独立输出目录比较原参数与较宽松阈值。此项是应用层探索，保持原预测与冻结的 E4 不变；先保留原演示，验收后再决定提交版本。

## 8. 分析和复现实验图表

按 [分析流程](analysis_workflow.md) 复算 Stereo/LR、NYU、视频、控制实验和应用的统计，并输出匹配帧图。每个分析入口默认在对应实验下创建 `analysis/`；已有目录时使用新的输出路径，保留原结果。

Stereo 选定场景箱型图、NYU RGB/GT/预测/误差图可以独立生成：

```powershell
& $py -X utf8 scripts/part1_baseline/plot_stereo_distributions.py
& $py -X utf8 scripts/part2_models/plot_nyuv2_cases.py
```

这两项分别保存到 Stereo 和 NYU 的 `analysis/selected_figures/`。视频、主融合与消融的图片/影片由各 Part 的运行和分析入口生成；无需运行报告制作工具。用新运行的指标、设置和实际观察更新报告，不能沿用旧结果冒充本次复现。

如需课程提交，按 [提交说明](submission.md) 收集报告和视频；这不属于模型或评测运行步骤。

## 只补可视化与中断处理

若原始数组已完整保存、仅可视化失败，可以沿用相同参数并增加 `--visualize-only`：

```powershell
& $py -X utf8 scripts/part1_baseline/run_framewise.py --cache outputs/shared/video_cache/dev_tokyo_100 --visualize-only
& $py -X utf8 scripts/part2_models/run_video.py --cache outputs/shared/video_cache/dev_tokyo_100 --visualize-only
& $py -X utf8 scripts/part2_models/run_nyuv2.py --model unidepth --subset test --visualize-only
```

这些模式不加载模型，只读取已保存预测。光流和融合入口也支持同参数加 `--visualize-only`。已经完整的视频会校验来源与文件哈希；不完整的可视化目录先交给助手诊断，不能为修复图片而删除 raw 数组。NYU 可以重建 PNG/浏览页，原始预测不变。

NYU 推理中断可在**同一代码版本、配置、设备、样本和输入**下原命令加 `--resume`。旧代码版本的未完成输出不能跨本次改造直接续跑。其他入口默认拒绝覆盖已有运行；不要自行加覆盖参数或清空整个 outputs。

旧结果的算法与指标复查仍可使用 [历史协议](course_protocol_20261102.md)中的显式运行路径。历史打包和分类迁移工具已退出当前实验流程；新结果直接保存在对应 Part。
