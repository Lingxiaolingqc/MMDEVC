# 应用参数对照：保留近中景结构、虚化远背景

当前应用按相对远近混合原图与 Gaussian blur，不识别天空、乘客或轨道类别。固定整段 DA 的 1%–99% 范围，将相对 inverse-depth-like 预测转成接近程度 q；q 越大，预测越近。

原演示参数是 transition=[0.45,0.70]、sigma=5 px：q≤0.45 完全虚化，q≥0.70 保留原图，中间平滑过渡。这是手设的演示规则，不是课程指定值、米制距离或经过验证的最优参数。

为减少乘客、轨道和支架被虚化，先比较 transition=[0.10,0.30]，仍保持 sigma=5 px。新规则只完全模糊 q≤0.10 的区域，在 q≥0.30 时保留原图。它扩大了清晰区域，但不能保证所有真实近景都保留：模型若把细杆预测为远景，单改阈值仍可能失败。

## 运行和查看

在原应用已完成后运行。深度网络、E4 冻结参数、原预测和原应用都保持不变；这只是观察过指定视频之后的应用参数探索，不作为新的盲测深度改进。

```powershell
$clip = "course_rollercoaster_70"
$candidate = "outputs/application_exploratory/$clip/transition010_030"
& $py -X utf8 scripts/application/render_blur_demo.py --cache "outputs/shared/video_cache/$clip" --baseline "outputs/part1_baseline/dav2_relative/test/$clip" --improved "outputs/part3_improvement/test/$clip/E4" --transition 0.10 0.30 --sigma-px 5 --reference-application "outputs/application/$clip" --output-dir $candidate
```

查看 `$candidate/settings_comparison.mp4`：

| | 左列 | 右列 |
|---|---|---|
| 上排：原参数 | DA + 原虚化规则 | E4 + 原虚化规则 |
| 下排：新参数 | DA + 新虚化规则 | E4 + 新虚化规则 |

- 上下比较：同一预测，应用参数改变的影响。
- 左右比较：相同应用参数，DA 与 E4 预测的影响。
- `demo.mp4` 仍提供 RGB / E4 relative depth / DA 应用 / E4 应用四列。
- `settings_comparison.csv` 记录每帧的平均虚化权重与两种参数产生的 RGB 差异。它们不是准确率或画质分数。
- `metadata.json` 保存实际参数、旧应用来源哈希和文件哈希。旧应用必须来自相同预测、同一时间缓存、同一显示范围和相同效果函数。

两种导出都保留原时间戳。先检查车厢、乘客与支架是否更清楚、远景是否仍模糊，再检查视频中是否出现新的跳变。不能把新参数改善视觉效果归因于 E4；它也可能同样改善 DA。

当前提交计划仍指向原应用。确认新演示后再决定是否调整提交计划，并同时保留旧/新参数比较。报告图像会读取所选应用元数据中的参数，避免图像和视频采用不同设置。
