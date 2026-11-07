# part1_baseline：表格与记录

[总导航](../../../PARTS.md) · [本机媒体](../../../outputs/by_part/part1_baseline/README.md)

Stereo 的 per_scene.csv 和 summary.csv 给出视差误差、有效比例；framewise/ 的两个 metadata 对应原始 DA V2。Jadeplant 八图示例在本机媒体目录。原始 DA 的比较画面在 Part 2 的四栏视频中。

## 本目录文件

- [stereo/per_scene.csv](stereo/per_scene.csv)
- [stereo/audit.json](stereo/audit.json)
- [stereo/summary.csv](stereo/summary.csv)
- [framewise/dev_metadata.json](framewise/dev_metadata.json)
- [framewise/test_metadata.json](framewise/test_metadata.json)

## 原始预测与运行目录

- [Stereo 原始运行与数组](../../../outputs/stereo_all_week1_20261021/)
- [DA 开发预测](../../../outputs/course_video_20261102/dev_tokyo_100/dav2_relative_364/)
- [DA 测试预测](../../../outputs/course_video_20261102/course_rollercoaster_70/dav2_relative_364/)

这些原始路径只在准备了数据/输出的机器上存在，不随 Git 下载。分类副本中的元数据仍引用原始运行路径；复算请按照 [课程协议](../../../docs/course_protocol_20261102.md) 操作。副本不是新实验，也不是新的评分结果。
