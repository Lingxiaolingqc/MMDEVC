# 视频融合与评测

2026-10-28：E0–E5 的程序、保存结果复算和固定色标视频已在官方示例短片上跑通。真实自录视频尚未提供，因此融合参数仍为 `provisional`（待开发集验证），正式效果和最终参数尚未确定。主责仍为 A 实现、B 实验，由助手执行技术工作；人工复核待完成。

## 相同输入与光流

所有方法读取同一个 RGB 帧缓存及其实际时间戳。缓存最长边 640，目标采样 10 fps；模型 short side 默认 364。在 16:9 视频上，两种模型实际 encoder 输入均为 364×644。VDA 保留官方 offline 32-frame window；它的吞吐率不等于在线应用延迟。

使用 [OpenCV Farneback](https://docs.opencv.org/4.13.0/d4/dee/tutorial_optical_flow.html) 估计每对相邻 RGB 的两个方向光流，参数固定在 `configs/flow.yaml`。光流单位为缓存图像上的像素。对当前像素 p，backward flow（当前指向上一帧）为 b(p)，历史采样位置 q=p+b(p)。`warp(history,b)` 在 q 处双线性采样。越界或插值支持包含无效深度时，历史信息不可用。

公共评测区域只由 RGB 和光流确定，在比较任何深度模型之前计算并保存：

`||b(p)+f(q)||² ≤ 0.01 × (||b(p)||²+||f(q)||²) + 0.5`

同时要求 q 在图像内、光流有限，且 RGB 平均绝对差除以 255 后不超过 0.2。这里的前后向一致性和颜色检查是遮挡/错误匹配的估计，不是真实遮挡 GT。重复纹理、透明表面和低纹理区仍可能通过错误匹配。

## E0–E5

设当前原始米制深度为 D，上一帧融合结果经过采样后为 H：

`output = (1 − w) × D + w × H`

| 版本 | 历史采样与权重 |
| --- | --- |
| E0 | 原始 DA V2 metric，不融合 |
| E1 | EMA（指数移动平均）：同一像素位置使用固定历史权重 |
| E2 | 用光流采样 H，使用固定历史权重 |
| E3 | E2 加入前后向光流一致性与 RGB 颜色差过滤 |
| E4 | E3 加入自适应权重，深度变化大或位于深度边界时减弱历史影响 |
| E5 | 独立运行的原始 VDA metric |

E4 使用以下启发式权重，不把它解释为校准后的概率：

`w = alpha × exp(−photo/sigma_photo − fb_error/sigma_flow − |ln(H/D)|/sigma_depth − edge/sigma_edge)`

`edge` 是当前深度与四邻居的最大绝对 log-depth 差。初始值见 `configs/fusion_provisional.yaml`，尚未通过自录 dev 视频选择。首帧直接使用原始预测；当前预测无效时继续保留无效状态，不用旧深度填洞。E2 也必须排除越界与无效插值，这是完成数值运算的基本条件。

光流只对齐图像位置，不把上一帧 Z 深度转换到当前相机坐标系。相机靠近目标时真实 Z 会改变；E4 的深度变化项用于减少历史滞后，是否有效必须用实际移动/遮挡片段检查。

## 指标与固定分母

公共区域 R 对所有方法完全相同。某方法在 R 内同时拥有当前和采样历史深度的像素组成 V。

- `temporal_l1_m_on_valid`：V 内 mean(|D−H|)，单位 m。另存绝对 log 差、对称相对差，以及除以实际时间间隔的 m/s 变化率。
- `pair_coverage`：|V|/|R|。每对帧以及整段都报告，不能只比较条件误差。
- `capped_l1_m_full_reference`：R 内有效差值截断到 1 m，无效预测按 1 m 计入；分母始终为 |R|。
- `agreement_5pct_full_reference`：|D−H|/((D+H)/2) 严格小于 0.05 的像素比例，分母为 |R|，无效预测算不一致。
- `frame_median_m`：每帧有效预测中位数；`median_abs_log_change` 汇总相邻中位数变化。
- 固定相机且静止场景才报告逐像素时间标准差，并同时报告整段均有有效预测的像素覆盖率。

整段时序指标按对应参考/有效像素数汇总；空公共区域记为 undefined（未定义），不记成零误差。内部融合拒绝区域不参与选择 R。中位数变化和光流对齐差均可能含真实运动信号；无 GT 视频只支持稳定性和响应分析，不能证明绝对深度准确率提高。边缘拖影与遮挡后的恢复需同时查看固定色标视频。

## 运行和复算

先按 [模型说明](../environments/depth_models.md) 为同一缓存生成 DA metric 与 VDA metric 输出，再执行：

```sh
python scripts/prepare_flow.py --cache outputs/videos/dev_static/cache --output-dir outputs/videos/dev_static/flow
python scripts/run_fusion_suite.py --cache outputs/videos/dev_static/cache --baseline outputs/videos/dev_static/dav2_metric --reference-vda outputs/videos/dev_static/vda_metric --flow outputs/videos/dev_static/flow --output-dir outputs/videos/dev_static/ablations
python scripts/verify_fusion_suite.py --suite outputs/videos/dev_static/ablations --cache outputs/videos/dev_static/cache --baseline outputs/videos/dev_static/dav2_metric --reference-vda outputs/videos/dev_static/vda_metric --flow outputs/videos/dev_static/flow
python scripts/render_depth_video.py --cache outputs/videos/dev_static/cache --run outputs/videos/dev_static/ablations/E0 --run outputs/videos/dev_static/ablations/E4 --run outputs/videos/dev_static/vda_metric --output-dir outputs/videos/dev_static/comparison
```

成功输出应包含 E0–E5 六行 `comparison.csv`，六行 `reference_pixels` 完全一致。E0–E4 保存浮点深度、有效 mask 和历史权重；E5 复用原 VDA 文件。复算程序重算 RGB 光流、公共区域、融合数组和指标，不重新运行深度网络。

开发时只在自录 dev 视频尝试少量候选参数，同时比较闪烁、边界、前后移动响应和遮挡拖影。选定后保存开发片段 ID、参数和证据文件哈希，再将配置设为 frozen。测试入口会拒绝 provisional 配置或缺少开发选择证据的配置。正式 test 应在冻结后统一运行一次；公开视频检查结果不能充当开发选择证据。

效率表把模型推理、CPU 光流和融合开销分开。`estimated_pipeline_fps` 是三项耗时相加后的估计，排除解码/磁盘写入，不称为实测端到端 FPS。
