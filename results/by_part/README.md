# 按 Part 查结果

从 [项目总导航](../../PARTS.md) 查看代码来源和运行入口。此目录是经过 SHA256 核对的精选结果副本，随 Git 同步；生成清单见 [index.json](index.json)。原路径继续用于实验复算。

- [part1_baseline](part1_baseline/README.md)：Stereo 的 per_scene.csv 和 summary.csv 给出视差误差、有效比例；framewise/ 的两个 metadata 对应原始 DA V2。Jadeplant 八图示例在本机媒体目录。原始 DA 的比较画面在 Part 2 的四栏视频中。
- [part2_models](part2_models/README.md)：metric/ 是全部 654 张 NYUv2 的三指标与来源；video/ 是 VDA 元数据、帧数差异和显存记录。test/dev_comparison.mp4 由四栏组成：RGB、原 DA、E4 改进、VDA；它同时服务于 Part 2 比较和 Part 3 改进展示。
- [part3_improvement](part3_improvement/README.md)：selection/ 是三候选开发结果和冻结证据；dev/、test/ 的 comparison.csv 是 E0–E5。ablations.mp4 展示 RGB 和 E0–E4，E5 原 VDA 在 Part 2 的四栏比较中。failure_cases.csv 和困难案例图对应三个实际观察。
- [application](application/README.md)：demo.mp4 四栏是 RGB、E4 相对深度、原 DA 虚化、E4 虚化。blur_strength.csv 给出每帧效果数值；metadata.json 记录固定参数和时间戳验证。应用值不是距离米数。
- [delivery](delivery/README.md)：报告 PDF 和 depth_results.zip 已有完整副本；这里的 JSON 是交付验证记录。报告正文 8 页、参考文献 1 页，ZIP 有 80 个 payload。人工验收与课程提交尚未完成。

在仓库根目录运行 python scripts/organize_part_results.py --tables-only --check-only，可检查 Git 中这些副本。新 clone 的本机媒体链接可能不存在，需要另行传输或重建。
