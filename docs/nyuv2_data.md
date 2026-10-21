# NYUv2 labeled 数据

使用 Spring 2027 指定的 labeled 部分，不下载 428 GB raw 视频。原始数据说明见 [NYU 官方页面](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html)。数据与 processed 数组保留本地，Git 只保存清单、代码和必要检查记录。

## 文件与标准划分

把以下两个文件放到 `configs/paths.local.yaml` 的 `data_roots.nyuv2`（默认 `data/nyuv2`）：

- [nyu_depth_v2_labeled.mat](https://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat)：2,972,037,809 bytes。公开副本校验值 MD5 为 `520609c519fba3ba5ac58c8fefcc3530`；完成下载后再核验，不对部分文件宣称完整性。
- [splits.mat](https://github.com/ankurhanda/nyuv2-meta-data/raw/master/splits.mat)：本次文件 SHA256 为 `6e081404491a8bfba2f066beaa9713ef6046f9007aa5f799f2a350506a580cee`。读取验证 795 个 train、654 个 test，均为 1-based 索引、互不重叠且覆盖 1449 个 ID。

`manifests/nyuv2.csv` 保留全部 1449 个原始 ID 和标准 split；`manifests/nyuv2_dev.csv` 选训练侧按索引首次出现的三个不同场景：0003 / office_0003、0005 / office_0004、0010 / office_0006。不是从 test 中选择好看的图，也没有把 train 改名为 test。

## 读取与提取

```sh
python -m pip install -e ".[data,stereo,nyu]"
python scripts/prepare_nyuv2.py --output-dir outputs/nyuv2_check_NEW
```

该命令读取完整 MAT，提取全部 RGB 和未量化的 float32 GT，并保存三个 dev 样本的检查图、mask、全样本统计和来源哈希。已有 processed 文件只有逐值等于 MAT 解码数据时才接受，内容不同会报错，不覆盖。检查输出目录必须是新的。

HDF5 中 `images=(1449,3,640,480)`，索引一张后转置为 HWC；`depths=(1449,640,480)`，索引后转置为 HW。输出 RGB 480×640×3 uint8，GT 480×640 float32，字段使用填补后的 `depths`，原单位为米。没有旋转、镜像、除以 1000 或 PNG 深度量化。

固定 mask 为 `[45:471,41:601]` 与有限且 **0.001≤GT≤10 m** 的交集。三个 dev 样本各有 238,560 个有效评测像素。指标、crop 来源和课程参考实现的边界见 [评测协议](evaluation_protocol.md)。

## 当前核验边界

已从官方 MAT 的真实 HTTP ranges 读取并核验三个 dev 样本、原始数据结构和全部场景名称；三个 RGB/GT 已保存于 `processed/`，尺寸、方向、单位及 train 归属检查通过，0003 已连接 UniDepthV2。原始范围读取复用了 `NYUv2Labeled.load`；不是生成或替代的数据。

完整 MAT 下载及其与已缓存三个样本的逐值复核仍待完成。清单中存在某个 ID 不代表本机已提取该文件；正式 654 张测试前必须完成全量下载和上述提取命令。
