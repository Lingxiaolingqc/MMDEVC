# part2_models：表格与记录

[总导航](../../../PARTS.md) · [本机媒体](../../../outputs/by_part/part2_models/README.md)

metric/ 是全部 654 张 NYUv2 的三指标与来源；video/ 是 VDA 元数据、帧数差异和显存记录。test/dev_comparison.mp4 由四栏组成：RGB、原 DA、E4 改进、VDA；它同时服务于 Part 2 比较和 Part 3 改进展示。

## 本目录文件

- [video/dev_metadata.json](video/dev_metadata.json)
- [video/test_metadata.json](video/test_metadata.json)
- [metric/summary.json](metric/summary.json)
- [metric/per_image.csv](metric/per_image.csv)
- [metric/provenance.json](metric/provenance.json)
- [video/input_audit.json](video/input_audit.json)
- [video/resource_attempts.json](video/resource_attempts.json)

## 原始预测与运行目录

- [UniDepth 原始 654 张预测](../../../outputs/nyuv2_test_20261028/unidepth/)
- [VDA 开发预测](../../../outputs/course_video_20261102/dev_tokyo_100/vda_relative_364/)
- [VDA 测试预测](../../../outputs/course_video_20261102/course_rollercoaster_70/vda_relative_364/)

这些原始路径只在准备了数据/输出的机器上存在，不随 Git 下载。分类副本中的元数据仍引用原始运行路径；复算请按照 [课程协议](../../../docs/course_protocol_20261102.md) 操作。副本不是新实验，也不是新的评分结果。
