# 实验日志

## 2026-06-22 环境配置

### Issue: CUDA 驱动与 PyTorch 版本不兼容

**现象**：
- `torch.cuda.is_available()` 返回 False
- 报错: `The NVIDIA driver on your system is too old (found version 12060)`

**原因**：
- 系统驱动版本: 470.199.02 (最高支持 CUDA 12.6)
- 预装 PyTorch: 2.12.0+cu130 (需要 CUDA 13.0)
- CUDA 13.0 需要更新的驱动 (>470)，当前驱动不支持

**解决方案**：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
降装为 PyTorch 2.6.0+cu124 (CUDA 12.4，兼容驱动 470.x)

**教训**：
1. 使用 GPU 前先 `nvidia-smi` 看驱动版本和 CUDA 版本
2. PyTorch 的 cuXXX 后缀表示编译时 CUDA 版本，需 ≤ 驱动支持的 CUDA 版本
3. 驱动版本 vs CUDA 版本对照: https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/

---

## 2026-06-22 数据探索

### 数据概览
- 100 例，NCCT/CTA/SEG 各 100 个文件，全部一一对应
- 格式: nii / nii.gz (NIfTI)，nibabel 读取
- 每个体积: 512×512 × Z切片数 (Z 范围 311~415，各患者不同)
- SEG: 二值 mask (0/1)，float64

### HU 值分布 (5例抽样)

| 百分位 | NCCT | CTA |
|--------|------|-----|
| P0 | -1023 | -1023 |
| P50 | -946 | -945 |
| P95 | 117 | 162 |
| P99 | 711 | 732 |
| P99.9 | 1346 | 1552 |
| P100 | 35988 | 35631 |

### 血管区域 (SEG==1) 统计

| 指标 | NCCT | CTA |
|------|------|-----|
| 均值 | ~45 HU | ~254 HU |
| 标准差 | ~40 | ~54 |
| 范围 | [-800, 1770] | [-169, 1720] |

### 预处理决策
- HU 裁剪: NCCT [-200, 500], CTA [-200, 800] (覆盖软组织+血管+部分骨骼)
- P100 极值 (~35000+) 为金属伪影，裁剪后消除
- 归一化: Min-Max 到 [0, 1]
- 切片: 沿 Z 轴 (shape[2]) 提取 2D 切片
- Resize: 256×256 (参考报告同尺寸，减少计算量)
- 数据增强: 仅训练集，p=0.5 随机水平/垂直翻转
- 训练/验证/测试: 70/10/20 按 patient ID 划分，seed=42

---

## 2026-06-22 模型与训练决策

### 架构选择
- 多任务 U-Net (Shared Encoder + Dual Decoder)
- 编码器: 64→128→256→512, 瓶颈 1024
- CTA Decoder: Tanh 输出 (回归)
- SEG Decoder: Sigmoid 输出 (二分类)
- 输入: 1×256×256 (单通道灰度)

### 损失函数
- L1 Loss (CTA) + BCEWithLogitsLoss (SEG)
- 权重: λ_cta = 0.5, λ_seg = 0.5
- 理由: 两任务同等重要，先不引入额外超参

### 训练配置
- Optimizer: Adam, lr=1e-3, weight_decay=1e-5
- Batch size: 32
- Epochs: 100 (ReduceLROnPlateau 早停, patience=10)
- Scheduler: ReduceLROnPlateau, factor=0.5, patience=5
- 数据增强: RandomHorizontalFlip(p=0.5), RandomVerticalFlip(p=0.5)
- 训练/验证集切片动态加载 (on-the-fly slicing)

### 决策记录
- 仅轴向切片: 标准做法，横截面血管结构清晰
- 2D 非 3D: 数据扩充、训练速度、参考报告验证可行
- 256×256: 平衡细节与显存，参考报告同尺寸
- 基础增强: 翻转足够，弹性变形留作后续优化
- 无外部参考: 参考报告范式已足够

---

## 2026-06-22 目录结构

```
NCCT2CTA-VesselSeg/
├── data/
│   └── data_100/           # 解压后数据
│       ├── NCCT/           # 100 nii 文件
│       ├── CTA/            # 100 nii 文件
│       └── SEG/            # 100 nii.gz 文件
├── src/                    # 源代码
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── train.py
│   ├── test.py
│   ├── visualize.py
│   └── utils.py
├── outputs/                # 训练输出
│   ├── checkpoints/        # 模型权重
│   ├── predictions/        # 测试集分割结果 nii
│   └── figures/            # 可视化图片
└── doc/
    ├── 要求.md
    ├── plan.md
    ├── experiment_log.md   # 本文件
    └── ref/
```

---

## 2026-06-22 实际执行中的 Issue

### Issue: NFS 数据加载极慢，训练无法启动

**现象**：
- `train.py` 启动后长时间无 epoch 输出
- DataLoader 每次 `__getitem__` 从 NFS 读取完整 .nii 文件（~200MB/文件）
- 每个 batch 耗时 20+ 秒，第一 epoch 预计 4+ 小时

**根因**：
原始 `dataset.py` 的 `__getitem__` 每次调用 `nib.load().get_fdata()` 读取完整 3D 体积，然后只取一个切片。即使加了 volume cache（5 个病人），随机 shuffle 导致频繁 cache miss。

**解决过程**：
1. 第一版 fix：volume cache（5 病人）→ 无效，shuffle 导致 miss
2. **最终方案**：预处理脚本 `preprocess.py`
   - 一次性读取 100 例原始 nii
   - 执行所有预处理（HU 裁剪、归一化、resize 256×256、切片）
   - 保存为 .npz 文件（3.7GB，vs 37GB 原始数据）
   - 训练时用 `PreprocessedDataset` 全量加载到内存（24GB RAM）
3. 训练日志写 `/tmp/` 而非 NFS（NFS 写入缓存导致日志不更新）
4. 设置 `PYTHONUNBUFFERED=1` 确保 stdout 实时刷新

### Issue: 多个训练进程同时运行抢占 GPU

**现象**：
多次 `nohup` 启动后旧进程未清理，积累 3+ 个 Python 进程同时训练。

**解决**：
每次重启前 `kill -9` 清理所有 train.py 进程。

### Issue: matplotlib NumPy 版本冲突

**现象**：
```
ImportError: numpy.core.multiarray failed to import
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**解决**：
```bash
pip install --upgrade matplotlib
```

### Issue: .nii.gz 双后缀导致 Patient ID 匹配失败

**现象**：
`get_patient_ids()` 返回 0 个病人，因为 `Path.stem` 对 `patient0533.nii.gz` 只去掉 `.gz`，返回 `patient0533.nii`，与 NCCT/CTA 的 `patient0533` 不匹配。

**解决**：
改用 `f.name.split('.')[0]` 提取 ID。

---

## 2026-06-22 第一次训练结果 (Round 1)

### 训练过程

训练 79 个 epoch 后 early stop（patience=15），最佳 Dice 出现在 epoch 64。

| 阶段 | Train Loss | Val Loss | Dice | PSNR | SSIM |
|------|-----------|----------|------|------|------|
| Epoch 1 | 0.0351 | 0.0191 | 0.0651 | 25.55 | 0.89 |
| Epoch 10 | 0.0120 | 0.0111 | ~0.41 | ~27 | ~0.91 |
| Epoch 20 | 0.0108 | 0.0111 | ~0.47 | ~27 | ~0.91 |
| Epoch 64 **(best)** | 0.0096 | 0.0111 | **0.5267** | 26.98 | 0.9084 |
| Epoch 79 (stop) | 0.0095 | 0.0111 | 0.5217 | 26.97 | 0.9081 |

**Loss 收敛情况**：
- L1 (CTA 生成): 0.0245 → 0.0148，降幅 40%
- BCE (分割): 0.0458 → 0.0042，降幅 91%
- 分割 loss 下降更快，但 Dice 止步于 0.52~0.53 → 典型类别不平衡问题

### 测试集结果

#### 2D Slice-level evaluation（256×256 切片）

| Loss | Dice | IoU | PSNR | SSIM | Accuracy |
|------|------|-----|------|------|----------|
| 0.011 | 0.5179 | 0.4059 | 26.36 | 0.9123 | 0.9978 |

#### 3D Per-case evaluation（完整体积，20 例测试集）

| 指标 | Mean ± Std | Min | Max |
|------|-----------|-----|-----|
| **PSNR** | **24.90 ± 1.53** | 21.98 | 27.03 |
| **Dice** | **0.7919 ± 0.0593** | 0.5948 | 0.8439 |
| **IoU** | **0.6591 ± 0.0754** | 0.4233 | 0.7299 |
| **Accuracy** | **0.9979 ± 0.0010** | 0.9962 | 0.9991 |

#### 逐例详情

| Case | PSNR | Dice | IoU | 评价 |
|------|------|------|-----|------|
| 0434 | 26.31 | 0.8439 | 0.7299 | ★ 最佳 |
| 0441 | 26.72 | 0.8369 | 0.7196 | ★ |
| 0523 | 26.36 | 0.8395 | 0.7234 | ★ |
| 0528 | 24.86 | 0.8122 | 0.6837 | 中等 |
| 0498 | 22.11 | 0.5948 | 0.4233 | ▼ 最差 |
| 0521 | 24.58 | 0.7233 | 0.5665 | ▼ |

### 可视化输出

5 例代表性病例 + 1 张汇总柱状图，保存在 `outputs/figures/`：
- `case_0434.png` (Dice=0.84, 最佳之一)
- `case_0544.png` (Dice=0.84, 最佳之一)
- `case_0528.png` (Dice=0.81, 中位数)
- `case_0521.png` (Dice=0.72, 较差)
- `case_0498.png` (Dice=0.59, 最差)
- `summary_metrics.png` (全部 20 例 Dice/PSNR 柱状图)

---

## 2026-06-22 Round 1 结果分析

### 成绩定位

| 指标 | 计划预期 | Round 1 实际 | 判定 |
|------|---------|-------------|------|
| CTA PSNR | 28~35 | 24.90 | ⚠️ 偏低 |
| CTA SSIM | 0.85~0.95 | 0.91 | ✅ 合格 |
| SEG Dice (3D) | 0.80~0.92 | 0.79 | ⚠️ 边缘 |
| SEG IoU | 0.70~0.85 | 0.66 | ⚠️ 偏低 |

**结论**：CTA 生成质量尚可（SSIM 达标），但 PSNR 偏低；血管分割勉强接近预期下界，有较大提升空间。

### 关键发现：2D Dice (0.52) vs 3D Dice (0.79) 的巨大差异

这是本次实验最重要的发现之一：

1. **2D slice-level Dice (0.52)**：在 256×256 上直接 sigmoid → threshold → 比较
2. **3D per-case Dice (0.79)**：256×256 推理 → 概率图堆叠 → resize 回 512×512 → threshold → 比较

差异来源：**resize 步骤**。将连续概率图用 INTER_LINEAR 从 256×256 上采样到 512×512，保留了亚像素级别的信息，细小血管在更高分辨率下得以恢复。Binarize 前做 resize 相当于在更高分辨率上决策边界位置。

**启示**：0.52 是模型的"真实"分段能力（在 256×256 上），0.79 是经过后处理"修复"的效果。要真正提升性能，需要增大输入尺寸或在 512×512 上训练。

### 为什么分割 Dice 止步于 0.52？（根因分析）

这组指标中有一个非常典型的信号：**Accuracy 0.998 但 Dice 只有 0.52**。

这是"极端类别不平衡"的经典表现——血管占整张切片的面积比很小（可能 <1%），模型只需要把所有像素预测为背景就能拿到 99%+ 的准确率。BCE Loss 优化的是逐像素分类精度，当一种类别极度稀缺时，模型收到的梯度信号几乎全部来自"没有血管的区域"，导致它对血管区域的学习不充分。

**5 个根本原因**：

1. **BCE Loss 不适合极度不平衡的分割任务**：BCE 优化逐像素准确率，不直接优化 Dice。血管占比 << 1%，模型倾向预测全 0
2. **256×256 丢失细小血管**：原始 512×512 中 1-3 像素宽的血管在 resize 后可能只剩 0-1 像素
3. **2D 切片丢失 Z 轴连续性**：相邻切片血管是连续的，但模型每片独立预测，看不到上下文
4. **损失权重 0.5/0.5**：BCE 收敛更快（91% vs L1 40%），可能主导优化方向
5. **验证集仅 10 例**：统计波动大，early stop 判断不够稳定

---

## 2026-06-22 改进方向 (Round 2 计划)

### 高优先级（预期明显提升）

**1. Dice Loss 替代/补充 BCE**
- 公式：`DiceLoss = 1 - (2|A∩B| + smooth) / (|A| + |B| + smooth)`
- 直接用 Dice 作为优化目标，与评价指标对齐
- 组合：`Loss = 0.5*L1 + 0.5*DiceLoss + 0.05*BCE`（保留少量 BCE 稳定训练）

**2. 增大图像尺寸到 512×512**
- 保留更多细小血管信息
- 代价：显存翻倍，batch_size 从 32 降到 8（A100 80GB 仍可运行）

**3. 偏重分割任务的损失权重**
- 从 0.5/0.5 改为 λ_seg=0.7, λ_cta=0.3
- 因为分割是更难的任务（类别不平衡）

### 中优先级

**4. 2.5D 输入**：连续 3 张相邻切片 → 3 通道输入，提供 Z 轴上下文
**5. 验证集扩大**：10 → 15 例（从训练抽），更稳定
**6. CosineAnnealing 调度器**：替代 ReduceLROnPlateau，更平滑

### 低优先级

7. Attention U-Net（注意力门控 skip connection）
8. 弹性变形增强（对细长血管结构有效）

---

*日志持续更新中，基于 NCCT2CTA-VesselSeg 项目。*
