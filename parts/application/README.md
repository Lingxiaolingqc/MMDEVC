# Application：深度感知背景虚化

[总导航](../../PARTS.md) · [应用结果](../../results/by_part/application/README.md)

## 课程要求出处

[新版任务书](../../../Project_3__Monocular_Metric_Depth_Estimation_and_Video_Consistency.pdf)第 2 页，Section 2 Submission Requirements 下的 Depth Demo & Video (Mandatory) 写道：

> Include one simple application output, such as depth-aware background blur or a point-cloud view, from a provided image or clip.

意思是：从给定图像或视频生成一个简单应用输出，例如深度感知背景虚化或点云视图。下一句说明录制的演示足够。第 5 页 Qualitative Evaluation 也要求展示 RGB、预测深度和所选应用输出。

这是所有组需要交的简单应用。第 4 页 Part 3 Direction F 是另一个可选改进方向；我们选了 Direction A 的光流融合，仍要交简单应用，不需要把 Direction F 再当第二项独立改进完成。

## 我们实现了什么

- [背景虚化函数](../../src/p3_depth/application.py)：助手实现固定整段相对范围、近处清晰/远处模糊权重、图像混合；GaussianBlur 来自 OpenCV。
- [应用视频导出](../../scripts/render_blur_demo.py)：同一 RGB、同一阈值和模糊参数，对比原 DA 与 E4。保存数值并核查视频帧数与时间戳。
- [比较视频导出](../../scripts/render_depth_video.py)：助手实现标签、色标和拼版，使用现成编码库。

相对预测一般大值更近。应用使用相对排序，不显示距离米数，也不模拟经过标定的镜头光圈。两种效果差别较小，边界光晕、误判前后景仍可能存在。

## 使用入口

```powershell
python parts/application/run.py blur --help
python parts/application/run.py package --help
```

package 转发到正式视频包验证/打包脚本；报告是单独的 PDF。已验证交付物在 [delivery](../../results/by_part/delivery/README.md) 中查找，实际原始路径见 [提交说明](../../docs/submission.md)。
