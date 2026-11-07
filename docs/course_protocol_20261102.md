# 新版 Project 3：协议与复现

任务依据：2026-11-02 收到的 Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf（修订日期 2026-10-31）。[文件哈希与视频帧数差异](../results/course_input_audit_20261102.json)、[实际结果](../results/course_20261102/README.md)。

## 1. 当前范围和比较规则

| 部分 | 采用方案 | 规则 |
| --- | --- | --- |
| Stereo | Middlebury trainQ 15 对，StereoSGBM | 正且有限视差 GT；有效预测 MAE + 固定 GT 分母有效比例 |
| Metric | UniDepthV2 ViT-S，NYUv2 标准 654 test | RGB-only；480×640；crop [45:471,41:601]；GT 0.001–10 m；预测 clip 同范围；逐图等权 AbsRel/RMSE/δ1 |
| Video | DA V2 Small relative、VDA Small relative | 同帧、原始 PTS、FP16；整段固定色标 |
| ONE Improvement | Flow-Guided Fusion | 三候选独立 dev；冻结 α=0.70 后 test E0–E5 |
| Application | 相对深度背景虚化 | 原 DA / E4 同范围、同参数；导出演示 |
| Delivery | CVPR 报告、GitHub、depth_results.zip | 6–8 页正文+参考文献；指定阅读；三个真实困难案例 |

官方 rollercoaster 文件完整只有 **70 帧 / 24 fps**，与说明“前 100 帧”不符。用户已确认使用全部 70 帧并在报告说明；没有重复/插帧，也不冒称教师已批准。开发片段来自另一个固定官方文件 Tokyo-Walk 的前 100 帧 / 15 fps，scene group 和源哈希独立。

新版不要求补拍自录视频、米制视频比较、额外边界/时序数值指标或 arXiv 发布。旧实验保留为历史/补充。视频没有深度 GT，变化指标下降不等于准确率提升。

- Stereo：OpenCV fixed-point /16 得 px，再用 f×B/[1000×(d+doffs)] 得 m。左右一致性、越界与正有限 Z 筛选；无效不填洞。MAE 场景平均，coverage 像素加权；15 对包含 2 个开发场景。
- NYUv2：只给模型 RGB，不做测试 GT 尺度拟合。评测区域出现非有限预测直接报错，不删像素改善成绩。新版结果与历史未裁剪七指标分开。
- Relative：有限零值、负值不因“非正”而无效；它们不是米制 Z。相对模型之间没有共同物理数值尺度。
- Fusion：当前帧 1%–99% 范围只用于置信权重，不逐帧重新归一化输出。输出仍与原 DA 同单位。
- Display：E0–E4 共用原 DA 整段 1%–99% 范围；VDA 用自己的整段范围。颜色不能作跨模型米制距离比较。
- Compute：VDA input=518 在 warmup OOM，故双方采用 364；VDA 峰值 allocated 约 2.33 GiB，DA 约 0.263 GiB。不包含别的程序占用。E4 推理+CPU 光流+融合合计约 3.10 fps，是加和估算，不是端到端 FPS。

## 2. 本机已有结果：先复算

在仓库根目录执行，选择本机环境；不要把 Windows venv 复制给另一台机器。输出目录已存在时换一个新名字，保留旧结果。

~~~powershell
$py = ".\.venv-cuda\Scripts\python.exe"
& $py -m unittest discover -s tests -v
& $py scripts/part1_baseline/run_stereo_suite.py --phase verify --output-dir outputs/stereo_all_week1_20261021
& $py scripts/part1_baseline/audit_course_stereo.py --output-dir outputs/stereo_personal_recheck
& $py scripts/part2_models/evaluate_course_nyu.py --run outputs/nyuv2_test_20261028/unidepth --output-dir outputs/nyu_personal_recheck
$v = "outputs/course_video_20261102"
$c = "$v/course_rollercoaster_70"
& $py scripts/part3_improvement/verify_fusion_suite.py --suite "$c/frozen_suite" --cache "$c/cache" --baseline "$c/dav2_relative_364" --reference-vda "$c/vda_relative_364" --flow "$c/flow" --verification-output outputs/video_personal_recheck.json
& $py scripts/package_depth_results.py --plan configs/course_export_plan.json --output outputs/submission_20261102/depth_results.zip --check-only
~~~

预期：Stereo 全 15 对通过；NYU 为 0.1107756635 / 0.3928173152 m / 0.8947810718；video 为 69 对 flow/mask、E0–E4 原始预测/权重及 E0–E5 分数复算通过；打包 2 clips / 80 payload。单元测试通过不等于成员完成实验验收。

## 3. 新工作副本：准备环境与数据

下面命令用于没有对应 outputs 的新 clone。现有副本用第 2 节，不能删除原结果后原地重跑。计时会随硬件/驱动变化；新输出必须生成新元数据与冻结凭据，不能沿用旧哈希声明。

先按 [CUDA 环境](../environments/cuda.md)、[官方源码](../third_party/README.md)、[模型环境](../environments/README.md)和 [UniDepth](../environments/unidepth.md)安装固定版本；复制并填写本机 paths.local.yaml。Middlebury、NYUv2 按各自数据说明准备，标准 split 的 manifests 随 Git 同步。

~~~powershell
& $py scripts/prepare_model_assets.py --model dav2_relative
& $py scripts/prepare_model_assets.py --model vda_relative
& $py scripts/part1_baseline/run_stereo_suite.py --phase all --output-dir outputs/stereo_all_week1_20261021 --config configs/stereo_sgbm_selected.yaml
& $py scripts/part2_models/run_nyuv2.py --model unidepth --subset test --device cuda --precision fp16 --output-dir outputs/nyuv2_test_20261028/unidepth
& $py scripts/part2_models/evaluate_course_nyu.py --run outputs/nyuv2_test_20261028/unidepth --output-dir outputs/course_nyu_20261102/unidepth
~~~

UniDepth 权重见其独立环境说明。run_nyuv2 的内置历史七指标不是新版主成绩；新版以 evaluate_course_nyu 输出为准。无需训练或更换大模型。

## 4. 官方视频缓存

固定 VDA 源码 assets/example_videos/ 包含两个源文件；先复制到 manifest 规定位置。上游源哈希若改变应调查，不能只改清单绕过检查。

~~~powershell
New-Item -ItemType Directory -Force data/videos/raw | Out-Null
Copy-Item third_party/Video-Depth-Anything/assets/example_videos/Tokyo-Walk_rgb.mp4 data/videos/raw/dev_tokyo_100.mp4
Copy-Item third_party/Video-Depth-Anything/assets/example_videos/davis_rollercoaster.mp4 data/videos/raw/course_rollercoaster_70.mp4
$v = "outputs/course_video_20261102"
$source = "https://github.com/DepthAnything/Video-Depth-Anything/blob/4f5ae23172ba60fd7bc11ef671cca678842c7072/assets/example_videos"
& $py scripts/prepare_video.py --video data/videos/raw/dev_tokyo_100.mp4 --clip-id dev_tokyo_100 --scene-group tokyo_walk --split dev --scene-type camera_motion --original-fps --source-kind public_example --source-url "$source/Tokyo-Walk_rgb.mp4" --expected-frames 100 --max-frames 100 --max-long-edge 640 --output-dir "$v/dev_tokyo_100/cache"
& $py scripts/prepare_video.py --video data/videos/raw/course_rollercoaster_70.mp4 --clip-id course_rollercoaster_70 --scene-group davis_rollercoaster --split test --scene-type camera_motion --original-fps --source-kind course_example --source-url "$source/davis_rollercoaster.mp4" --expected-frames 70 --max-frames 100 --max-long-edge 640 --output-dir "$v/course_rollercoaster_70/cache"
~~~

两个 ID 已在清单里，所以这里不加 --register；该参数只用于登记新片段。缓存保留连续原始 frame index、PTS 和方向信息，不静默补齐缺失帧。

## 5. 开发、选择和冻结

~~~powershell
$d = "$v/dev_tokyo_100"
foreach ($model in @("dav2_relative","vda_relative")) {
    $entry = if ($model -eq "dav2_relative") { "scripts/part1_baseline/run_framewise.py" } else { "scripts/part2_models/run_video.py" }
    & $py $entry --model $model --cache "$d/cache" --input-size 364 --device cuda --precision fp16 --output-dir ($d+"/"+$model+"_364")
    if ($LASTEXITCODE -ne 0) { throw "Model failed: $model" }
}
& $py scripts/part3_improvement/prepare_flow.py --cache "$d/cache" --output-dir "$d/flow"
foreach ($a in @("050","070","085")) {
    & $py scripts/part3_improvement/run_fusion_suite.py --cache "$d/cache" --baseline "$d/dav2_relative_364" --reference-vda "$d/vda_relative_364" --flow "$d/flow" --config "configs/fusion_relative$a.yaml" --output-dir "$d/alpha$a"
    if ($LASTEXITCODE -ne 0) { throw "Suite failed: $a" }
    & $py scripts/part3_improvement/verify_fusion_suite.py --suite "$d/alpha$a" --cache "$d/cache" --baseline "$d/dav2_relative_364" --reference-vda "$d/vda_relative_364" --flow "$d/flow"
    if ($LASTEXITCODE -ne 0) { throw "Verification failed: $a" }
}
~~~

开发中看固定帧与完整视频，比较轮廓、拖影、遮挡、当前帧偏离和指标。按 docs/fusion_selection_review.example.json 字段写 review，保存在 outputs/course_video_20261102/selection/review.json。可参考已保存的本次 review，但不要复制未检查的观察作为自己的验收。

~~~powershell
& $py scripts/part3_improvement/freeze_fusion_config.py --candidate "alpha050=$d/alpha050" --candidate "alpha070=$d/alpha070" --candidate "alpha085=$d/alpha085" --selected alpha070 --review "$v/selection/review.json" --output-dir "$v/selection/frozen"
~~~

本次 0.70 是保守取舍：0.85 的指标更低，但更偏离当前帧，三组 E4 画面差别较小。不能手动只改 status=frozen，测试会核对 dev 凭据、参数和源哈希。

## 6. 冻结后测试与导出

~~~powershell
$c = "$v/course_rollercoaster_70"
foreach ($model in @("dav2_relative","vda_relative")) {
    $entry = if ($model -eq "dav2_relative") { "scripts/part1_baseline/run_framewise.py" } else { "scripts/part2_models/run_video.py" }
    & $py $entry --model $model --cache "$c/cache" --input-size 364 --device cuda --precision fp16 --output-dir ($c+"/"+$model+"_364")
    if ($LASTEXITCODE -ne 0) { throw "Model failed: $model" }
}
& $py scripts/part3_improvement/prepare_flow.py --cache "$c/cache" --output-dir "$c/flow"
& $py scripts/part3_improvement/run_fusion_suite.py --cache "$c/cache" --baseline "$c/dav2_relative_364" --reference-vda "$c/vda_relative_364" --flow "$c/flow" --config "$v/selection/frozen/fusion_frozen.yaml" --output-dir "$c/frozen_suite"
& $py scripts/part3_improvement/verify_fusion_suite.py --suite "$c/frozen_suite" --cache "$c/cache" --baseline "$c/dav2_relative_364" --reference-vda "$c/vda_relative_364" --flow "$c/flow"
foreach ($clip in @("dev_tokyo_100","course_rollercoaster_70")) {
    $b = "$v/$clip"
    $s = if ($clip -eq "dev_tokyo_100") { "$b/alpha070" } else { "$b/frozen_suite" }
    & $py scripts/render_depth_video.py --cache "$b/cache" --run "$b/dav2_relative_364" --run "$s/E4" --run "$b/vda_relative_364" --output-dir "$b/render_primary"
    & $py scripts/render_depth_video.py --cache "$b/cache" --run "$s/E0" --run "$s/E1" --run "$s/E2" --run "$s/E3" --run "$s/E4" --output-dir "$b/render_ablations"
}
& $py scripts/application/render_blur_demo.py --cache "$c/cache" --baseline "$c/dav2_relative_364" --improved "$c/frozen_suite/E4" --output-dir "$c/application"
~~~

三个实际案例见 results/course_20261102/failure_cases.csv。完整视频仍需成员播放检查，图片不能独自证明流畅性或无拖影。测试结果不能用于再次选择参数。

报告图片由 scripts/build_course_figures.py 从保存数组生成，并记录源哈希和裁剪区域。报告采用本次原结果；跨机重跑若数值变化，应同步更新表格和 provenance。最终按 [提交说明](submission.md)核验并打包。验收者无需重跑训练或所有 GPU 推理，优先使用第 2 节的复算路径。
