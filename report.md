# ButterflyNet 阶段性方法说明 (Progress Report Draft)

> 本报告为阶段性草稿，记录当前已完成部分的设计思想、实现方式与选择理由。后续阶段（模型、训练、评估、改进、解释性）完成后将扩展完善。遵循项目原则：Seed=42 可复现、代码自主实现、重视分析与解释。

## 1. 环境与依赖设计
**目标**: 构建一个最小但可扩展的图像分类研究环境，确保复现性与后续扩展（增强/解释性）便利。

### 1.1 Conda 环境
- 使用 `conda create -n butterfly python=3.10` 固定 Python 主版本，避免不同 Python 次版本导致依赖行为差异。
- Conda 便于 GPU/CPU 切换和后续增加库（如 `pyyaml`、`wandb`）时的兼容性管理。

### 1.2 依赖选择与原因
| 库 | 作用 | 选择理由 |
| --- | --- | --- |
| torch / torchvision | 模型与数据 transforms | 社区主流，易于自定义训练循环，满足“自主”要求 |
| numpy | 数值操作 | 高效处理索引、统计计算 |
| scikit-learn | 指标 (precision, recall, f1, confusion matrix) | 复用成熟评估实现，避免重复造轮子 |
| pillow | 图像加载 | 与 torchvision 兼容，处理 RGB 转换 |
| matplotlib / seaborn | 可视化 (loss 曲线、混淆矩阵、Grad-CAM 热力图叠加) | 组合提供美观与灵活性 |
| tqdm | 进度条 | 提升可观察性，辅助性能与卡顿定位 |
| opencv-python | 后续热力图插值 / 叠加处理 | Grad-CAM 生成与视觉叠加更方便 |

### 1.3 复现性策略
- 后续所有脚本将统一设置：
  ```python
  import random, numpy as np, torch
  random.seed(42); np.random.seed(42); torch.manual_seed(42)
  if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
  ```
- 分割索引、统计结果均持久化到文件（`splits/*.csv`, `dataset_stats.json`），避免运行波动。

## 2. 数据集组织与划分策略
**数据结构**: `ButterflyClassificationDataset/` 下 50 个类别文件夹，每个文件夹存放该类原始图片。

### 2.1 划分文件形式
- 生成 `splits/train.csv` 与 `splits/val.csv`：结构统一列名 `path,label`。
- 使用 CSV 而不是直接目录复制的原因：
  1. 轻量且易于版本控制 (diff 清晰)。
  2. 可追加新列（例如未来的 `weight`、`quality_flag`）。
  3. 读取整洁：`csv.DictReader` 可读性好。

### 2.2 类别映射与元信息
- `class_to_idx.json`：保存类别到整数标签的双向可逆索引（后续报告、可解释性需要原始名称）。
- `meta.json`：可扩展字段（时间戳、划分策略、图像总数、均值方差来源）。

### 2.3 划分策略说明
- 比例：当前采用 8:2（训练:验证）。理由：
  - 类别总数为 50，验证集中保留每类足够样本用于稳定统计。
  - 训练集仍保持绝对数量规模，利于模型初期拟合。
- 随机性控制：在生成索引前设置全局随机种子，以确保任意机器复现相同划分。
- 样本层面策略：按**类别内随机打乱**再切分，避免长尾类别出现验证集严重不足情况。

### 2.4 风险与备用策略
- 若后续观察到验证指标方差大，可切换 7:3 比例或使用分层交叉验证抽样生成额外 `val_alt.csv`。
- 若类别不平衡严重，将计算每类样本数用于加权损失或采样器设计（计划后续在报告扩展）。

## 3. 自定义数据集类 `ButterflyDataset`
### 3.1 设计目标
- 明确、透明、最小依赖：只处理路径解析 + 图像加载 + 变换调用。
- 支持后续扩展：`return_path` 参数方便可解释性阶段获取文件名生成热力图结果。

### 3.2 结构说明
- 初始化：读取 CSV 逐行解析相对路径 -> 绝对路径校验 -> 形成 `samples: List[(Path, int)]`。
- `__getitem__`: 加载 -> 转 RGB -> 调用 transforms -> 返回 `(tensor, label)` 或 `(tensor, label, path)`。
- 加载函数 `default_loader`：集中处理 PIL 转换；可未来替换成更高效的 `accimage` 或 OpenCV。

### 3.3 为什么不用 `ImageFolder`
- 需要自定义的 CSV 划分与潜在元数据扩展，不依赖目录层次表达集合切分。
- 便于后续添加“过滤列表”“困难样本标记”“伪标签”等额外列。

### 3.4 可维护性考虑
- 保持字段名称语义清晰 (`samples`, `loader`, `return_path`)；避免隐藏状态。
- 将类别映射单独存储于 JSON，不嵌入类属性，防止模型训练脚本重复加载。

## 4. 数据统计 (RGB Mean / Std)
### 4.1 统计范围
- 使用训练集样本（避免信息泄漏验证集），对全部训练图片统一 resize 到 224 以保证像素规模一致。
- 统计结果文件：`src/data/dataset_stats.json`
  - mean: `[0.4731, 0.4614, 0.3254]`
  - std:  `[0.2678, 0.2560, 0.2647]`

### 4.2 为什么自行统计
- 与自然图像（ImageNet）的分布存在偏差（蝴蝶数据背景与翅膀纹理颜色集中度较高，蓝/绿通道较低）。
- 自行统计可避免过度依赖预训练统计，提高输入归一化贴合度，促进收敛稳定性。

### 4.3 实现要点与优化考虑
- 当前实现使用 PIL 读入并转 tensor 时做了较朴素的像素展开；后续可改成：
  ```python
  from torchvision import transforms
  tensor = transforms.ToTensor()(img)  # 更简洁可靠
  ```
- 为避免内存峰值，逐图增量累积 sum 和 sumsq，而不是整体堆叠。

### 4.4 结果解读
- R/G 均值接近，B 均值显著更低：可能与数据集中背景（叶子/泥土）主色调偏暖相关。
- Std 较低：指颜色分布集中，后续可考虑颜色扰动增强（ColorJitter）增加泛化。

## 5. 基础预处理管线 `get_base_transforms`
### 5.1 组成
1. Resize(224 x 224, Bicubic) — 保持与主流架构兼容 (ResNet/Vision Transformer baseline)。
2. ToTensor — 标准张量格式便于后续 GPU 加速。
3. Normalize(mean/std) — 使用统计结果；若缺失则回退 ImageNet 统计便于快速实验。

### 5.2 设计理由
- **简洁性优先**：基线阶段避免随机增强，保证后续对比实验的公平起点。
- **扩展性**：函数式接口允许后续替换为增强版本：`get_aug_transforms()`。
- **可复用**：验证与训练统一使用统计归一化；提升特征分布稳定性。

### 5.3 后续计划
- 增强阶段新增：RandomHorizontalFlip / RandomRotation / ColorJitter 等对比分析其提升幅度。
- 归一化参数将用于 Grad-CAM 时的反归一化可视化（恢复原始色彩）。

## 6. 原创性与复现性落实情况
| 要求 | 当前状态 | 说明 |
| --- | --- | --- |
| Seed=42 | 已在分割阶段执行 (脚本层设定) | 训练脚本后续将加入 GPU seed 设置 |
| 自主数据处理 | 已实现 | 自定义 CSV + Dataset 类，无高封装 API |
| 统计与预处理 | 已实现 | 手写均值方差统计脚本与 transforms 管线 |
| 持久化 | 已实现 | splits + stats JSON 可追踪 | 

## 7. 当前阶段的潜在风险与对策
| 风险 | 描述 | 对策 |
| --- | --- | --- |
| 类别不均衡未知 | 还未统计每类样本数 | 添加类频次分析，决定是否加权损失 |
| 图像质量差异 | 可能存在模糊或裁剪异常 | 数据清洗脚本（计划第2项）中加入简单尺寸/加载异常检测 |
| 统计脚本效率 | 当前像素提取方式稍低效 | 改用 `transforms.ToTensor()` + 批处理优化 |

## 8. 下一步工作规划 (Preview)
1. 设计基线模型 `ButterflyNet`：3 个卷积块（Conv2d+BN+ReLU+MaxPool）+ 全连接输出 50 类。
2. 编写训练循环（日志：loss, acc；保存最佳验证集权重）。
3. 添加评估脚本生成分类报告与混淆矩阵，为后续改进提供依据。

## 9. 报告格式后续扩展说明
- 后续章节将加入：模型结构图、训练曲线、混淆矩阵图、Grad-CAM 示例。
- 避免粘贴大段源代码，只引用核心伪代码或结构摘要。

---
## 10. 基线模型 `ButterflyNet` 设计与原理 (新增)
### 10.1 设计目标
- 在不依赖预训练的条件下提供一个“适中容量”基线：足够学习 50 类间的纹理/色彩差异，又避免早期严重过拟合。
- 结构清晰，方便后续插入改进（Dropout、Attention、更多卷积层）并进行消融研究。

### 10.2 架构概述
| Block | 组成 | 输出通道 | 输出尺寸 (输入 224x224) | 作用 |
|-------|------|----------|-------------------------|------|
| ConvBlock1 | Conv3x3 + BN + ReLU + MaxPool2 | 32 | 32x112x112 | 初步低层边缘/颜色捕捉，保持分辨率较高 |
| ConvBlock2 | Conv3x3 + BN + ReLU + MaxPool2 | 64 | 64x56x56 | 中层纹理与斑点形状抽象 |
| ConvBlock3 | Conv3x3 + BN + ReLU + MaxPool2 | 128 | 128x28x28 | 更高层语义模式（翅膀大区域结构） |
| GAP | AdaptiveAvgPool2d(1x1) | 128 | 128x1x1 | 全局聚合，降低参数量避免全连接巨量参数 |
| FC | Linear(128 -> 50) | 50 | 50 | 输出类别 logits |

### 10.3 关键设计取舍说明
1. 卷积核大小 3x3：保持局部感受野增长平滑，与BN+ReLU搭配稳定训练。
2. 扩张通道 (32→64→128)：逐层翻倍为常见规律，兼顾容量与显存/速度；不采用过大（如 256+）避免参数膨胀。
3. MaxPool 降采样：简洁且推理速度快；后续可用 stride=2 的卷积替换以对比（可写入改进策略部分）。
4. BatchNorm：缓解不同蝴蝶背景光照差异带来的分布偏移；加速收敛。
5. Global Average Pooling：相比 flatten 再接多层 FC，可显著降低参数（控制过拟合），同时提高空间位置不敏感的鲁棒性。
6. 参数规模 (~100K)：对中等规模数据友好，利于快速迭代与调试；为后续引入更复杂模型（如 ResNet18 ~11M 参数）对比提供层级差异。

### 10.4 与替代方案对比
| 方案 | 优点 | 缺点 | 适用阶段 |
|------|------|------|----------|
| 当前 Baseline (3 Blocks) | 简单、可快速训练、易解释 | 表达力有限 | 基线建立 |
| 更深自定义 CNN (5-6 Blocks) | 更强特征层次 | 过拟合风险增大、训练更慢 | 改进阶段 |
| 预训练 ResNet18 修改最后层 | 更高初始准确率 | 需引入预训练依赖，分析原创性需说明 | 对比/提升 |

### 10.5 可扩展点预留
- 在 `features` 前后容易插入：Dropout、Attention (SE/CBAM)、额外 ConvBlock。
- GAP 前可以添加 `nn.Conv2d(128, 128, 1)` 做通道压缩或轻量特征适配。

### 10.6 结构伪代码 (逻辑摘要)
```pseudo
input -> [Conv3x3, BN, ReLU, MaxPool] -> [Conv3x3, BN, ReLU, MaxPool] ->
      [Conv3x3, BN, ReLU, MaxPool] -> GAP -> Linear(128, 50) -> logits
```

### 10.7 参数与感受野分析 (简化)
- 单个 3x3 卷积使局部感受野线性增长；三层叠加后在 28x28 特征图级别已经汇聚为较大原图区域信息。
- MaxPool(2)×3 将空间缩放至 1/8：在蝴蝶翅膀纹理保留与背景去除间取得折衷。

### 10.8 可视化结构图生成方法
选择一种或两种：
1. torchinfo (推荐用来展示维度)
  ```bash
  pip install torchinfo
  python -c "from src.models.butterfly_net import create_model; from torchinfo import summary; m=create_model(); summary(m, input_size=(1,3,224,224))"
  ```
2. torchviz (计算图)
  ```bash
  pip install torchviz
  python -c "import torch; from src.models.butterfly_net import create_model; m=create_model(); x=torch.randn(1,3,224,224); from torchviz import make_dot; g=make_dot(m(x), params=dict(m.named_parameters())); g.render('butterfly_net_graph', format='png')"
  ```
3. torch.fx + Graphviz (更细致控制)
  ```bash
  pip install graphviz
  python -c "import torch; from src.models.butterfly_net import create_model; m=create_model(); gm=torch.fx.symbolic_trace(m); print(gm.graph)"
  ```
- 若需嵌入报告，可将生成的 PNG 放入 `analysis/figures/` 并在后续正式报告引用。

### 10.9 后续评估指标与模型改进接口
- 训练脚本将收集：loss、accuracy（top-1），后续添加 F1 需在 `evaluate.py` 统一实现。
- 改进阶段可修改：
  - 通道进程：尝试 32→64→128→256 四层；比较参数与验证 F1 演化。
  - 正则：在每个 ConvBlock 后添加 `nn.Dropout(p)` 或在 `classifier` 前添加 Dropout。
  - Learning Rate Scheduler：如 CosineAnnealing 或 ReduceLROnPlateau，观察训练曲线平滑度。

### 10.10 小结
当前基线模型强调“清晰 + 适度容量 + 易比较”。它不是追求最高性能，而是为后续所有改进（数据增强/正则化/更深结构/预训练迁移/可解释性）建立可复现、易分析的参照点。

---
## 11. 训练循环设计与实现 (新增)
### 11.1 目标与约束
- 满足课程“自主实现”要求：手写 `for epoch in ...` 与 `loss.backward()`/`optimizer.step()`。
- 保证可复现、可观察、可扩展（便于对比改进策略）。

### 11.2 数据与加载策略
- 数据类：`ButterflyDataset` 读取 `splits/train.csv`、`splits/val.csv`。
- 预处理：优先使用 `src/data/dataset_stats.json` 的 mean/std，否则回退 ImageNet 统计。
- DataLoader：`shuffle=True` 仅用于训练集；`pin_memory=True` 在 CUDA 可用时加速 Host→Device 传输。

### 11.3 优化与损失
- 损失函数：`CrossEntropyLoss` — 多分类标准做法，梯度稳定，易于与 softmax logits 对齐。
- 优化器：`Adam(lr=1e-3)` — 对初期学习率较鲁棒，减少过多手工调参成本。

### 11.4 学习率调度 (LR Scheduler)
- `ReduceLROnPlateau`，监控 `val_acc`，`factor=0.5`，`patience=3`。
- 原因：当验证集准确率停滞时自动减小 LR，提升后期收敛稳定性，避免过早震荡或停滞。

### 11.5 训练/验证流程概要
1. 训练：`zero_grad()` → 前向 → `CE loss` → `backward()` → `optimizer.step()`；每批计算 `accuracy` 便于监控。
2. 验证：`no_grad()` 全量评估，返回 `val_loss/val_acc`；作为保存最优与 LR 调度的依据。

### 11.6 Checkpoint 与日志
- 最优模型：以 `val_acc` 为准保存 `checkpoints/baseline_best_<timestamp>.pt`（含模型/优化器状态与类别映射）。
- 训练历史：保存到 `checkpoints/history_<timestamp>.json`，字段含 `epoch/train_loss/train_acc/val_loss/val_acc/lr`。

### 11.7 干运行 (Dry Run) 与限批
- 目的：快速验证数据→模型→反向→保存的完整流程，不消耗大量时间。
- 使用方式：`--dry_run`（强制单 epoch）与 `--limit_train_batches / --limit_val_batches`。
- 示例：
  ```bash
  source ~/miniconda3/etc/profile.d/conda.sh
  conda activate butterfly
  python train.py --dry_run --limit_train_batches 2 --limit_val_batches 2 --batch_size 8
  ```

### 11.8 复现性措施
- 训练入口调用 `set_seeds(42)`：同步设置 `random`/`numpy`/`torch`（含 `cuda.manual_seed_all`）。
- 说明：PyTorch 完全确定性还可设置 `torch.use_deterministic_algorithms(True)` 与禁用 CUDNN benchmark；本项目暂未启用，后续可按需要加入作为实验变量记录在报告中。

### 11.9 后续扩展点
- 不平衡处理：引入 `class_weight` 到 `CrossEntropyLoss` 或 `WeightedRandomSampler`；权重来源于每类频次。
- 正则化：梯度裁剪（如 `clip_grad_norm_`）、Dropout、数据增强（在改进阶段实现）。
- 训练效率：混合精度（`torch.cuda.amp`）与更合适的 Scheduler（Cosine、OneCycle）。

### 11.10 直接训练命令
```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate butterfly
python train.py --epochs 20 --batch_size 32 \
  --dataset_root ButterflyClassificationDataset \
  --train_csv splits/train.csv --val_csv splits/val.csv \
  --stats src/data/dataset_stats.json --seed 42
```

---
**版本信息**: Draft v0.3 （日期：2025-11-19） 已包含基线模型与训练循环设计与实现。

后续更新：添加独立评估脚本与加权指标/混淆矩阵、长训结果曲线与误差分析、类别分布统计与不平衡对策。
## 12. 100 Epoch 基线训练结果与分析 (新增)
### 12.1 曲线可视化
已用脚本 `scripts/plot_history.py` 生成折线图（单图 3 子图）：`analysis/figures/training_curves.png`，包含：
- 子图1：Train Loss vs Val Loss
- 子图2：Train Acc vs Val Acc
- 子图3：Learning Rate (log-scale 自动切换)

生成命令示例：
```bash
python scripts/plot_history.py \
  --history checkpoints/history_20251119_135455.json \
  --out analysis/figures/training_curves.png \
  --title "ButterflyNet Baseline 100 Epochs"
```

### 12.2 训练阶段表现概述
| 阶段 | 特征 | 指标变化 | 说明 |
|------|------|----------|------|
| 初始 (1-10) | 损失快速下降 | Val Acc ~0.11→0.37 | 模型建立基本区分能力，LR=1e-3 有效 |
| 中期一 (11-30) | 持续稳步提升 | Val Acc ~0.41→0.62 波动 | 首次 LR 降至 5e-4 后稳定提升；偶发回落体现类间难度 |
| 中期二 (31-60) | 缓慢震荡上升 | Val Acc ~0.58→0.62 | 多次 Plateau 触发 LR 下调；泛化进入微增区间 |
| 后期 (61-100) | 轻度波动/收敛 | Val Acc 峰值 0.625 附近 | 进一步 LR 下降至 ~1.5e-8 后收益有限，曲线趋于平滑 |

### 12.3 关键观察
1. Loss 收敛：Train/Val Loss 差距不大（末期约 1.77 vs 1.69），说明当前容量未出现明显过拟合，仍有提升空间（增强/更深网络）。
2. Accuracy 走势：训练准确率保持与验证接近（约 0.59 vs 0.62 峰值）；验证略高于训练的偶发现象可能源于随机批次分布与 BN 统计效果。
3. LR 调度有效性：每次验证精度停滞后降低 LR（1e-3→5e-4→2.5e-4→...）伴随新一轮提升（如 Epoch 17→18 Val Acc 0.383→0.509）。后期 LR 过低收益递减，可考虑提前停止节约时间。
4. 模型容量：约 100K 参数在 50 类任务上达到 ~0.62 Val Acc，表明网络表达力有限；后续引入更丰富数据增强或更深结构可验证性能潜力。

### 12.4 初步问题与假设
| 现象 | 假设原因 | 后续验证手段 |
|------|-----------|--------------|
| Val Acc Plateau 反复 | 样本多样性不足 | 加入旋转/色彩增强对比收敛速度 |
| 收敛后提升缓慢 | 表达能力受限 | 引入第四卷积块或预训练 ResNet18 对比 |
| 偶发验证优于训练 | Batch 统计差异或类不均衡 | 统计每类频次 + 使用 Weighted Loss 检验 |

### 12.5 改进方向优先级建议
1. 数据增强：HorizontalFlip + ColorJitter + RandomRotation 预计提升泛化（优先级高）。
2. 模型扩展：添加一层 ConvBlock 或迁移 ResNet18 评估增益（中等）。
3. 正则化：轻量 Dropout (p=0.3) 置于 GAP 前后验证对稳定性的影响（中等）。
4. Scheduler 替换：尝试 CosineAnnealingLR 观察是否更平滑收敛（次级）。

### 12.6 训练早停建议
从曲线与指标看，Epoch ~80 后提升极其微弱；实际部署场景可使用早停（patience≈15）在约 Epoch 65-80 结束，节省 ~20% 时间成本。

### 12.7 下一步衔接
- 编写 `evaluate.py` 输出 Precision / Recall / F1 / Confusion Matrix，定位高混淆类别。 
- 整理类别样本数，判断是否需要类权重或采样策略。 
- 实施数据增强并对比曲线变化与最终指标提升幅度。

---
**版本信息**: Draft v0.4 （日期：2025-11-19） 已添加 100 Epoch 训练曲线与分析。

后续更新：评估脚本、混淆矩阵与类别分布统计；改进策略实验记录与对比表。
## 13. 验证集指标与混淆矩阵分析 (新增)
### 13.1 整体指标摘要 (Baseline Checkpoint)
来源文件：`analysis/metrics/val_metrics.json` （使用 `checkpoints/baseline_best_20251119_135455.pt`）。

| 指标 | 数值 |
|------|------|
| Accuracy | 0.5291 |
| Weighted Precision | 0.5401 |
| Weighted Recall | 0.5291 |
| Weighted F1 | 0.5065 |
| 样本总数 (support_total) | 894 |

说明：F1 < Accuracy 暗示类别预测不平衡（部分高精度低召回和低精度高召回并存）。

### 13.2 混淆矩阵可视化
图像路径：`analysis/figures/confusion_matrix.png`（行归一化版本）。行归一化突出“该真实类别在被预测到其它类别的分布”。

### 13.3 低表现类别与典型混淆对
按 per-class F1 排序最低的若干类别（<0.35）：
- `clodius parnassian` (recall 0.071, F1 0.125) — 极低召回，常被误分为背景色近似的白色类（与 `beckers white`, `cabbage white` 混淆）。
- `copper tail` (F1 0.226) 与 `crecent` (F1 0.143) — 图案细碎，模型当前卷积层深度不足以捕捉局部纹理差异。
- `orange oakleaf` (F1 0.333) — 叶片伪装背景与其它橙/褐色翅膀类（`question mark`, `morning cloak`）外观相近。
- `purple hairstreak` (F1 0.286) — 颜色低饱和 + 背景遮挡，易与灰/褐色类 (`grey hairstreak`, `wood satyr`) 混淆。

高召回但精度偏低的类别（例如 `indra swallow` recall 0.571, precision 0.522）：预测覆盖面广但包含较多 False Positive，提示特征边界不清晰。

### 13.4 误差模式总结
1. **浅色系白/淡黄类混淆**：`beckers white`, `cabbage white`, `clouded sulphur`, `pine white` 间互相误认。原因：翅膀主色 + 闪光条件相近。改进：增加颜色扰动和局部对比度增强数据增广。
2. **橙褐伪装类互混**：`orange oakleaf`, `question mark`, `morning cloak`, `eastern coma`。原因：边缘裂纹纹理相近。改进：加入随机裁剪与更深层卷积（提升局部纹理抽象）。
3. **细纹发丝类不足**：`purple hairstreak`, `grey hairstreak`, `black hairstreak`。原因：细线条特征在 28x28 最终特征图上可能过度下采样。改进：减少一次 MaxPool 或添加第四卷积层但保持更高分辨率（使用 stride=1 + Dilated Conv）。
4. **低支持高不确定类**：`clodius parnassian` 支持较低（14），召回极低；可能需要采样策略或类权重提高关注度。

### 13.5 定量指示（局部片段）
示例：`clodius parnassian` 行中真值 14 仅 1 个正确预测；其余散落于相似浅色类别列（具体在原始 CM 中第 8 行）。
`cabbage white` 真值 17 中 15 正确，表现较好（F1≈0.64），为改进可作为正向参照（分析与 `beckers white` 的背景差异）。

### 13.6 初步改进实验规划
| 问题 | 拟定策略 | 预计影响 |
|------|----------|----------|
| 白/黄类混淆 | ColorJitter(亮度/对比/饱和) + 随机灰度 | 提升颜色不变性与区分力 |
| 伪装橙褐类 | RandomRotation(±15°) + RandomResizedCrop | 改善边缘裂纹与形状泛化 |
| 细纹类召回低 | 减少一次池化或加入 Dilated Conv | 保留高频细节纹理 |
| 低支持类 | Class-weight 或 Oversampling | 提升梯度贡献与召回 |
| 模型容量不足 | 添加第4卷积块或迁移 ResNet18 | 增强中高层表达 |

### 13.7 指标解释与后续验证
- Weighted Precision≈Weighted Recall 但 F1 更低，表明部分类别的极端低召回拖累综合 F1。
- 后续在实施增强后将重新生成同结构指标表，对比每类 F1 改善幅度（重点观察“低召回类”）。

---
**版本信息**: Draft v0.5 （日期：2025-11-19） 已纳入验证集混淆矩阵与误差分析。

后续：执行增强与结构改进实验，写入对比表（Baseline vs Augmented vs Deeper）。
## 14. 数据增强实验与对比分析 (新增)
### 14.1 增强策略
当前实验使用两种模式：
1) full（旧，多操作增强）：RandomResizedCrop + HorizontalFlip + Rotation + ColorJitter。
2) simple（新，仅单一增强）：Resize(224) + RandomHorizontalFlip(0.5)。

根据最新要求，后续重训采用 simple 模式（只保留水平翻转）以降低扰动复杂度并观察基线性能是否回升。
验证阶段继续使用基础预处理（无随机操作）。

实现位置：`src/data/transforms.py` 新增 `get_aug_transforms()` 与 `get_aug_transforms_from_stats()`；`train.py` 新增 `--aug` 开关仅对训练集启用增强，验证集保持基线变换。

### 14.2 训练入口脚本
- 新增：`scripts/train_aug.sh`
  - 功能：一键训练（带增强）并在完成后自动调用 `evaluate.py` 评估。
  - 可通过环境变量覆盖默认参数：`EPOCHS`、`BATCH_SIZE`、`DATASET_ROOT`、`TRAIN_CSV`、`VAL_CSV`、`STATS`、`SEED`。

运行示例：
```bash
# 100 epoch 增强版训练 + 自动评估（输出 aug 指标与混淆矩阵）
bash scripts/train_aug.sh

# 自定义并行度与 epoch 数
NUM_WORKERS=8 EPOCHS=60 bash scripts/train_aug.sh --batch_size 48

# 多 GPU 可使用 torchrun（可参考 scripts/train_ddp.sh 模式），示例：
# torchrun --standalone --nproc_per_node=2 train.py --aug --epochs 60 --batch_size 32
```

训练输出命名规范：
- 检查点：`checkpoints/aug_best_<timestamp>.pt`
- 历史曲线：`checkpoints/history_aug_<timestamp>.json`
- 评估指标（增强）：`analysis/metrics/val_metrics_aug.json`
- 评估报告（增强）：`analysis/metrics/classification_report_aug.txt`
- 混淆矩阵（增强）：`analysis/figures/confusion_matrix_aug.png`

### 14.3 曲线与指标对比
生成曲线：
```bash
python scripts/plot_history.py \
  --history checkpoints/history_aug_<timestamp>.json \
  --out analysis/figures/training_curves_aug.png \
  --title "ButterflyNet Augmented"
```

对比表（填充运行后结果）：
| 方案 | Val Accuracy | Weighted Precision | Weighted Recall | Weighted F1 |
|------|--------------|--------------------|-----------------|-------------|
| Baseline | 0.5291 | 0.5401 | 0.5291 | 0.5065 |
| Augmented | 0.5045 | 0.5363 | 0.5045 | 0.4785 |
| Augmented-Simple | 0.4944 | 0.5100 | 0.4944 | 0.4637 |

每类 F1 Top-N 提升（观察重点）：白/淡黄混淆类、伪装橙褐类、发丝细纹类、低支持类。

### 14.4 预期影响与检验要点与实际结果
- 泛化：增强应降低过拟合风险，使 Val 曲线更平滑，Plateau 前推迟。
- 类别层面：
  - 白/淡黄类：ColorJitter + Flip 应提升区分背景/光照的鲁棒性（召回上升）。
  - 伪装橙褐类：RandomResizedCrop + Rotation 改善边缘纹理与形状变换不变性（F1 上升）。
  - 低支持类：总体 F1 或 Recall 有望上升，但需结合 class weight/采样进一步优化。
- 代价：训练时间略增（随机裁剪/色彩扰动），但推理成本不变。

实际第一轮 100 epoch 增强训练未达到预期：整体 Accuracy/F1 低于基线（Acc -0.0246；F1 -0.0280）。可能原因：
1) 增强提高输入分布多样性但模型容量有限（~100K 参数），特征表达不足以吸收更复杂变换。
2) 同步使用多种增强（裁剪+旋转+色彩抖动）增加学习难度，早期阶段未充分收敛；ReduceLROnPlateau 以 val_acc 为监控在噪声增大的情形下更易频繁降 LR。
3) 部分少样本/易混淆类别（morning cloak, viceroy, pine white）在强裁剪+色彩扰动下纹理被弱化导致召回显著下降。

改进计划：
- 分阶段增强：前 30 epoch 仅 Flip + 轻度 ColorJitter；中期再加入 Rotation；后期再启用 RandomResizedCrop（或减小 scale 范围到 (0.9,1.0)）。
- 调整 Scheduler：尝试 CosineAnnealingLR 或 OneCycleLR 减少因指标波动触发的过早 LR 缩小。
- 提升模型容量：添加第4卷积块或迁移 ResNet18（仅替换最后层）再对比同一增强策略的收益。
- 类别再平衡：对 F1 下降最严重的类使用 WeightedRandomSampler 或 class_weight（morning cloak, pine white, viceroy, malachite, metalmark）。
- 减弱色彩扰动幅度（brightness/contrast/saturation 0.15, hue 0.03）并观察低饱和细纹类（purple hairstreak, wood satyr）是否进一步改善。

### 14.5 每类 F1 变化（Augmented − Baseline，选取显著变化）
提升最大的类别（ΔF1 > +0.05）：
- eastern coma +0.1412
- wood satyr +0.1692
- ulyses +0.1419
- two barred flasher +0.1250
- clouded sulphur +0.0881
- copper tail +0.0645
- beckers white +0.0634
- purple hairstreak +0.0606
- red admiral +0.0591
- red spotted purple +0.0697
- southern dogface +0.0566

显著下降的类别（ΔF1 < -0.10）：
- morning cloak -0.3545
- pine white -0.2429
- viceroy -0.2104
- malachite -0.1876
- metalmark -0.1487
- painted lady -0.1369
- peacock -0.0903（接近阈值）
- question mark -0.1623
- clodius parnassian -0.1250
- pipevine swallow 轻微负向基线低（不显著）

观察要点：
- 增强对“背景颜色敏感类” (beckers white, clouded sulphur) 有正增益，对“伪装/纹理大型类” (morning cloak, painted lady) 反而削弱。
- 细纹类 purple hairstreak 有中等提升 (+0.0606)，说明旋转与轻度裁剪帮助对齐纹理，但过强的裁剪/色彩扰动可能仍损伤其它复杂纹理类别。
- wood satyr、ulyses 的提升表明增强改善了对局部光照和尺度变化的鲁棒性。

下一实验建议：固定改进策略后仅改变单一增强元素做消融（Baseline vs +Flip vs +Flip+ColorJitter vs +Flip+ColorJitter+Rotation vs 全增强），绘制指标与收敛速度对比，定位真正贡献度。

### 14.6 小结与后续
- 当前全套增强一次性启用未带来整体提升，反而略降；需分阶段/消融优化。
- 重点关注大幅下降类（morning cloak, pine white, viceroy, malachite, metalmark）并结合类样本分布进行再平衡处理。
- 下一步实施：分阶段增强 + 适度扩容模型 + 重新训练记载曲线；同时准备 Grad-CAM 对比基线与增强注意力差异以解释性能变化。
### 14.7 简化增强重训计划 (simple 模式)
目的：验证“仅水平翻转”轻量增强是否能避免 full 模式的性能下滑，同时保持对左右对称/姿态变化的泛化。
配置：`--aug --aug_mode simple`（脚本中通过 `AUG_MODE=simple`）。
预期：
- 整体 Accuracy/F1 贴近或略高于原基线 (≥0.51 F1)。
- 下降严重的类（morning cloak, pine white）召回回升；提升类保持部分增益（wood satyr, ulyses）。
运行命令（多卡示例）：
```bash
CUDA_VISIBLE_DEVICES=0,1 AUG_MODE=simple bash scripts/train_aug_ddp.sh --epochs 80 --batch_size 48
```
评估：
```bash
bash scripts/eval_aug_simple.sh  # 自动选择最新 aug-best(simple) checkpoint
```
补充：已完成 simple 模式评估并更新对比表（见上）。指标与产物：
- 指标文件：`analysis/metrics/val_metrics_aug_simple.json`（Acc 0.4944 / F1 0.4637）
- 分类报告：`analysis/metrics/classification_report_aug_simple.txt`
- 混淆矩阵：`analysis/figures/confusion_matrix_aug_simple.png`

结论简述：simple（仅 Flip）未能恢复基线，且略低于 full 增强总体水平，说明单一水平翻转不足以改善当前误差模式。下一步可在 simple 基础上仅引入轻度 ColorJitter 做最小增量对比，或转向模型容量/类再平衡方向。