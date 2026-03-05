# FastTracker + OIM Person Search ReID 增强

## 概述

本项目在 [FastTracker](https://github.com/Hamidreza-Hashempoor/FastTracker) 的基础上，引入了基于 **OIMNetPlus**（ECCV 2022）的行人重识别（ReID）模块，通过外观特征匹配显著增强了多目标跟踪（MOT）在复杂场景下的身份保持能力。

原版 FastTracker 完全依赖纯几何算法（Kalman Filter 预测 + IoU 匹配）进行数据关联——这在目标运动平稳、检测连续时表现出色，但面对检测丢失、目标重现、KF 漂移等常见问题时存在结构性不足。**ReID 模块的引入正是为了从根本上弥补这些短板。**

---

## 原版 FastTracker 面临的核心问题

### 问题 1：检测短暂丢失后，目标获得新 ID

| 场景 | 原因 |
|------|------|
| 目标被遮挡 / 检测器漏检数帧后重新出现 | KF 预测框已漂移，IoU 远低于匹配阈值 |
| **结果** | 同一个人被分配全新的 track ID，ID Switch +1 |

纯 IoU 匹配完全无法判断"这个新检测是不是之前丢失的那个人"——它只看空间位置重叠，不看"这个人长什么样"。

### 问题 2：KF 预测框膨胀导致匹配失败

| 场景 | 原因 |
|------|------|
| 目标连续数帧未匹配，KF 沿速度方向持续外推 | 预测框面积远大于真实检测框 |
| **结果** | IoU = 交集 / 并集，分母被膨胀框撑大，IoU 显著偏低（如 0.18） |

即使新检测此刻就在 track 的预测范围内，由于面积比严重失衡，标准 IoU 计算认为这是"不同的目标"。

### 问题 3：遮挡分离后 ID 互换

| 场景 | 原因 |
|------|------|
| 两个目标互相遮挡、KF 框互相重叠后分开 | 匈牙利匹配在重叠期间可能将检测分配给错误的 track |
| **结果** | 两个 track 的 ID 在分离后交换，后续跟踪全错 |

纯 IoU 匹配在两个框高度重叠时没有区分依据——它不知道哪个检测属于哪个人。

---

## 引入 OIM Person Search 后的解决方案

### 方案 1：批量 ReID 恢复（Step 4 增强）

**切入点**：在新 track 即将被创建时介入。

```
新检测未匹配任何 active track
          ↓
空间过滤：在丢失/维持 track 附近（IoU > 0.3 或中心距 < 1.5x 框尺寸）
          ↓
ReID 比对：从 frame_buffer 中提取该 track 最后一次成功检测的图像，
          与当前检测做 256 维 embedding 余弦相似度比较
          ↓
sim > 0.6 → 恢复原 ID，不创建新 track
sim < 0.6 → 正常创建新 track，不干扰原有逻辑
```

**核心设计**：
- **批量冲突解决**：同一个丢失 track 被多个检测匹配时，取相似度最高的；每个检测只能恢复一个 track
- **30 帧图像缓存**（`frame_buffer`）：保留最近 30 帧原始 RGB 图像，确保 ReID 特征提取有图可用
- **400×400 裁剪提取**：以目标中心裁剪固定区域输入 ReID 模型，保证目标占合理比例
- **不污染主流程**：只在 Step 4（新 track 创建前）介入，完全不修改 Step 2/3 的 IoU 匹配逻辑

### 方案 2：ContainPenalty（Step 2 IoU 增强）

**切入点**：在主匹配阶段修正 IoU 失真。

当一个 track 已经 `not_matched > 0`（KF 在外推），其膨胀预测框可能"包含"了真实检测但 IoU 很低。ContainPenalty 额外计算包含比（`交集 / 检测面积`），当检测大部分区域都落在 track 的预测框内时，用 `1 - 包含比` 作为替代代价，使得匈牙利算法有机会匹配成功。

```
T12 的 KF 预测框 [2242, 700, 46×104]（膨胀）
新检测          [2274, 705, 40×92] （真实大小）

IoU = 0.18 ✗ （远低于阈值 0.7）
包含比 = 0.35 → 替代代价 = 0.65 < 0.7 ✓ → 匹配成功
```

### 方案 3：ReID 分裂检测（重叠/分离 swap 纠正）

**切入点**：在两个 track 从重叠状态分离时校验。

```
检测叠 → 记录重叠对
         ↓
分离（IoU < 0.1）→ 取两个 track 在重叠前采集的参考 embedding
         ↓
对比当前位置的外观与两个参考 → 投票判断 ID 是否正确
         ↓
检测到互换 → 交换两个 track 的状态
```

每个 track 在 tracklet_len = 5/10/15 时采集 3 个参考 embedding，确保参考特征稳健（多时刻均值投票）。

---

## 与纯算法方案的本质差异

| 维度 | 纯算法（原版） | + OIM ReID（本版） |
|------|:---:|:---:|
| 匹配依据 | 仅空间位置（IoU） | 空间 + **外观特征** |
| 丢失 15 帧后重现 | 新 ID（必然失败） | **sim=0.77 恢复原 ID** ✓ |
| KF 膨胀框匹配 | IoU 失真 → 匹配失败 | ContainPenalty 修正 ✓ |
| 遮挡后 ID 互换 | 无法检测 | ref_embedding 投票纠正 ✓ |
| 计算开销 | 极低 | +ReID 推理（~5ms/crop） |
| 误匹配风险 | 无（但也无法恢复） | sim 阈值 0.6 把关 |

---

## 关键技术指标

- **ReID 模型**：OIMNetPlus（ResNet-50 backbone），256 维 L2 归一化 embedding
- **模型权重**：`person_search_oim.pth`（约 140MB）
- **推理设备**：自动检测 CUDA / MPS / CPU
- **恢复条件**：
  - 余弦相似度 > 0.6
  - 丢失时间 ≤ 30 帧（`track_buffer`）
  - 空间约束：中心距 < 1.5× 框尺寸
- **冲突解决**：全局最优（每个 track 取最高 sim，每个检测只用一次）

---

## 文件修改清单

| 文件 | 改动 | 作用 |
|------|------|------|
| `yolox/tracker/fasttracker.py` | +570 行 | ReID 恢复、ContainPenalty、分裂检测、遮挡处理 |
| `tools/demo_track.py` | +58 行 | raw_img 传递、ReID 可视化、日志输出 |
| `yolox/models/yolo_head.py` | +8 行 | MPS 设备兼容 |
| `yolox/utils/model_utils.py` | +18 行 | MPS 性能分析兼容 |
| `yolox/reid/person_search.py` | 新增 | OIMNetPlus ReID 模块 |

---

## 环境安装

**Python >= 3.8**，需要以下依赖：

```bash
# 1. 安装 PyTorch（根据你的 CUDA 版本选择，详见 https://pytorch.org）
pip install torch torchvision

# 2. 安装项目依赖
cd FastTracker
pip install -r requirements.txt

# 3. 编译安装项目
python setup.py develop

# 4. 安装额外依赖
pip install cython cython_bbox pycocotools
```

> 权重文件 `person_search_oim.pth` 和 `pretrained/bytetrack_x_mot20.pth.tar` 已包含在仓库中，无需额外下载。

## 运行

```bash
cd FastTracker
python tools/demo_track.py video \
  -f exps/example/mot/yolox_x_mix_mot20_ch.py \
  -c pretrained/bytetrack_x_mot20.pth.tar \
  --path "demo/cam01 (2).mp4" \
  --fuse --save_result
```

- `--path`：替换为你自己的视频路径
- `--save_result`：保存跟踪结果视频和 txt 到 `YOLOX_outputs/` 目录

ReID 模块在检测到 `person_search_oim.pth` 权重文件时自动启用，无需额外配置。Console 日志会输出所有 `[ReID-Recover]`、`[ContainPenalty]`、`[ReID] SWAP DETECTED` 事件供调试分析。

---

## 参考

- [FastTracker](https://arxiv.org/abs/2508.14370) - Hamidreza Hashempoor, Yu Dong Hwang
- [OIMNetPlus](https://github.com/cvlab-yonsei/OIMNetPlus) - ECCV 2022
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 基础关联框架
