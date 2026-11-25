# S0 decoder / ES0 encoder comparison

Three fixed seeds, 577 fitting / 218 scene-disjoint validation. No new test evaluation.

ES0 AbsRel 0.104583 versus S0 0.099848: +4.74% relative error.
Boundary F1 +0.002720 and boundary-band AbsRel -0.000954 are small average changes; their paired scene intervals include zero.
AbsRel difference CI also includes zero; report higher error on this fixed validation set, not proven population-wide inferiority.
Delta1 is lower by 0.036951; exploratory paired scene interval [-0.075065,-0.001706], without multiplicity correction.

All three runs have lower fitting loss but worse final validation AbsRel than the selected best point. These are signs consistent with overfitting/validation instability, not just the existence of a train/validation gap.
Best epochs 22/36/40; final epochs 32/46/50. One AMP overflow in seed320114 occurred after its best point; the skipped optimizer step and scaler recovery are retained.

ES0 30,737 trainable parameters versus S0 28,689. ES0 training peak allocated memory 442.92 MiB; three training/scoring runs total 26.84 minutes (excludes cache loading/extraction).
Same-session batch1 cached-refiner medians: S0 0.9235 ms, ES0 0.9951 ms; excludes foundation and IO. Do not call these full UniDepth pipeline latency.

Important source limit: S0 decoder init is derived from the block3 input branch plus camera conditioning; ES0 uses block12. This is a complete feature-source comparison, not a pure before/after-decoder transformation.

Current F3 remains selected; remaining encoder counterparts are queued, so no final queue-wide selection is made.

| NYUv2 sample | Viewed observation |
|---|---|
| 0005 | 桌面和部分背景的误差在 ES0 中比 S0 更高。轮廓大体保留原状，白板等缺失边界没有恢复；边界 F1 略升不能代替边界带深度误差。 |
| 0598 | ES0 的墙面、地面和桌面大区域误差低于 S0，但沙发与桌腿的细结构缺失仍存在；该图 AbsRel 改善而边界 F1 下降。 |
| 1440 | ES0 比 S0 减轻墙面误差，但两者整体仍差于原始 B0。椅背和台面高误差保留，边界图变化很小。 |
| 0226 | ES0 减轻 S0 对远处背景的深度高估，人物和家具周围仍有大片高误差。枝条等细结构错配持续，边界 F1 基本不变。 |
| 0246 | ES0 将部分远处背景预测得更近，墙面、顶面和架子附近误差扩大；部分右侧区域改善不能抵消整体退步。 |
| 0080 | ES0 的大区域误差略降，但按固定阈值检测到的架子边界比 S0 少，边界 F1 下降。三者均缺少大部分 GT 架子轮廓。 |
