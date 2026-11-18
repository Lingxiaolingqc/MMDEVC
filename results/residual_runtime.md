# Residual deployment cost — isolated local measurement

Hardware: RTX4060 Laptop 8 GiB. Batch1, official frozen UniDepthV2 Small resolution_level=0, official FP16 inference; refiners at 240x320. First five fitting images, no GT access. Two opposite method orders, five warmups and50 measured calls per block (100 calls per method).

Scope: decoded RGB already in CPU memory to full-resolution float32 CPU depth. Includes foundation inference, any required flip, CPU feature preparation/transfers and refiner; excludes model loading, file IO, visualization and video encoding. Training was not running concurrently.

| Method | Mean ms | Median ms | p95 ms | Inverse-mean FPS | Peak allocated MiB | Refiner params |
|---|---:|---:|---:|---:|---:|---:|
| B0 | 24.758 | 24.657 | 25.913 | 40.39 | 332.52 | 0 |
| R0_seed320112 | 28.899 | 28.673 | 30.623 | 34.60 | 332.60 | 19969 |
| frozen_features_seed320112 | 30.287 | 29.965 | 31.733 | 33.02 | 332.63 | 28689 |
| C1_features_unet_seed320112 | 32.706 | 32.303 | 35.896 | 30.58 | 332.68 | 35945 |
| legacy_bundle_seed320112 | 67.739 | 65.900 | 80.479 | 14.76 | 343.23 | 388001 |

The selected internal-feature CNN adds about 22.3% mean latency in this short local benchmark. It uses the same original foundation pass; the historical bundle incurs a second flip pass. C1 is slower here without an established accuracy benefit over the internal-feature CNN.

Peak allocated MiB is the PyTorch allocator measurement, not total driver/reserved GPU memory. FPS is the inverse mean single-image latency, not a video throughput benchmark. Laptop thermal state and background activity can vary; the two orders reduce but do not eliminate those effects.

Real-data deployment parity also passed for R0, internal-feature CNN and C1 on the first eight validation images: batch1 versus batch8 predictions differed by at most 0.001351 m, within the predeclared0.005m tolerance; per-image AbsRel differences were within0.0001 and batch8 reproduced saved validation within0.000001. This is numerical implementation evidence, not an accuracy improvement.
