# ButterflyNet: 蝴蝶分类模型 (Butterfly Classification Model)
## 安装 (Installation)
```bash
# 推荐 Python 3.10+，CUDA 12.1 环境
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 \
	--index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

若遇到 H20/CUBLAS 相关错误，可升级 cublas：
```bash
pip install nvidia-cublas-cu12==12.4.5.8
```

## 数据准备 (Data)
- 数据根目录默认：`ButterflyClassificationDataset/`。
- 如需从文件夹结构重新生成切分：
```bash
# 生成 8:2 训练/验证划分（固定 Seed=42）
python src/data/split_dataset.py \
	--dataset-root ButterflyClassificationDataset \
	--output-dir splits \
	--val-ratio 0.2 --seed 42

# 基于训练集计算均值/方差（用于 Normalize）
python src/data/compute_dataset_stats.py \
	--dataset_root ButterflyClassificationDataset \
	--train_csv splits/train.csv \
	--out src/data/dataset_stats.json
```

## 训练 (Training)
支持单机单/多卡。所有随机种子均已固定为 42。

通用参数（节选）：
- `--model_variant {baseline,deep}`：模型结构；默认 baseline。
- `--aug` 与 `--aug_mode {simple,full}`：开启数据增强与模式。
- `--dropout_p`：在 GAP 后、全连接前的 Dropout 概率。
- `--epochs/--batch_size/--lr`：训练超参。

方式 A：直接调用 Python（单卡示例）
```bash
# Baseline，无增强、无 Dropout，训练 50 轮
python train.py --device cuda --epochs 50 --batch_size 32 \
	--model_variant baseline

# Deep，无增强、无 Dropout
python train.py --device cuda --epochs 50 --batch_size 32 \
	--model_variant deep

# Deep + Aug-Simple（轻量增强）
python train.py --device cuda --epochs 50 --batch_size 32 \
	--model_variant deep --aug --aug_mode simple

# Deep + Aug-Full（组合增强）
python train.py --device cuda --epochs 50 --batch_size 32 \
	--model_variant deep --aug --aug_mode full

# Deep + Dropout p=0.20（推荐）
python train.py --device cuda --epochs 50 --batch_size 32 \
	--model_variant deep --dropout_p 0.20
```

方式 B：使用脚本（支持多卡，zsh/bash）
```bash
# 脚本默认 MODEL_VARIANT=deep，可通过环境变量覆盖
# 单卡（GPU 0）训练 Deep，无增强：
bash scripts/train_ddp.sh 0

# 双卡（GPU 0,1）训练 Deep + Aug-Simple：
AUG=1 AUG_MODE=simple bash scripts/train_ddp.sh 0,1 --epochs 50 --batch_size 48

# 单卡训练 Baseline + Dropout p=0.05：
MODEL_VARIANT=baseline DROPOUT_P=0.05 bash scripts/train_ddp.sh 0 --epochs 50
```

产出（默认）：
- 最优权重：`checkpoints/<tag>_best_<timestamp>.pt`
- 训练曲线历史：`checkpoints/history_<tag>_<timestamp>.json`
- `<tag>` 包含：模型（deep 可省略）、是否增强及模式（`baseline` 或 `aug-simple/aug-full`）、以及 `dropXX` 标识。

## 评估 (Evaluation)
方式 A：脚本（默认自动寻找 baseline 最优权重）
```bash
# 评估最近 baseline 最优权重
bash scripts/eval.sh

# 指定检查点（支持 deep/增强/Dropout），并保留行归一化的混淆矩阵
CHECKPOINT=checkpoints/deep_aug-simple_best_YYYYmmdd_HHMMSS.pt \
	bash scripts/eval.sh

# 生成非归一化混淆矩阵
CHECKPOINT=checkpoints/deep_baseline_best_YYYYmmdd_HHMMSS.pt \
	bash scripts/eval.sh --no_normalize_cm
```

方式 B：直接调用 Python
```bash
python evaluate.py \
	--checkpoint checkpoints/deep_aug-full_best_YYYYmmdd_HHMMSS.pt \
	--dataset_root ButterflyClassificationDataset \
	--val_csv splits/val.csv \
	--stats src/data/dataset_stats.json \
	--out_json analysis/metrics/val_metrics.json \
	--out_report analysis/metrics/classification_report.txt \
	--out_cm analysis/figures/confusion_matrix.png
```

## 作图 (Plotting)
训练曲线（基于保存的 history JSON）：
```bash
python scripts/plot_history.py \
	--history checkpoints/history_deep_YYYYmmdd_HHMMSS.json \
	--out analysis/figures/training_curves_deep.png \
	--title "Deep (50 epochs)"
```

## 导出 ONNX (Export)
```bash
# 注意：导出脚本不会从权重中读取 Dropout p，请手动指定与训练一致的值
python scripts/export_to_onnx.py \
	--checkpoint checkpoints/deep_baseline_best_YYYYmmdd_HHMMSS.pt \
	--model_variant deep --dropout_p 0.20 --num_classes 50 \
	--output checkpoints/deep_baseline_best_YYYYmmdd_HHMMSS.onnx
```

## 可解释性（Grad-CAM）
使用 Grad-CAM 观察模型在图像上的关注区域，验证可靠性并解释决策依据。

- 脚本：`scripts/grad_cam.py`
- 说明：默认解释 Top-1 预测类别；需与训练一致地传入 `--model_variant` 与 `--dropout_p`。

示例用法：
```bash
# 以 deep + drop0.20 最优权重为例，随机抽取 6 张验证集样本生成覆盖图
CHECKPOINT=checkpoints/deep_baseline_drop0.20_best_YYYYmmdd_HHMMSS.pt \
python scripts/grad_cam.py \
	--checkpoint "$CHECKPOINT" \
	--model_variant deep \
	--dropout_p 0.20 \
	--dataset_root ButterflyClassificationDataset \
	--val_csv splits/val.csv \
	--stats src/data/dataset_stats.json \
	--num_samples 6 \
	--out_dir analysis/figures/gradcam
```

## 复现实验清单 (Recipes)
以下命令可复现报告中的关键实验（均 50 epochs）：
```bash
# 1) Baseline（无增强/无 Dropout）
MODEL_VARIANT=baseline bash scripts/train_ddp.sh 0

# 2) Baseline + Dropout p=0.30（对比实验）
MODEL_VARIANT=baseline DROPOUT_P=0.30 bash scripts/train_ddp.sh 0

# 3) Baseline + Dropout p=0.05（弱正则对比）
MODEL_VARIANT=baseline DROPOUT_P=0.05 bash scripts/train_ddp.sh 0

# 4) Deep（无增强/无 Dropout）
bash scripts/train_ddp.sh 0

# 5) Deep + Dropout p=0.10/0.20/0.30（网格）
DROPOUT_P=0.10 bash scripts/train_ddp.sh 0
DROPOUT_P=0.20 bash scripts/train_ddp.sh 0
DROPOUT_P=0.30 bash scripts/train_ddp.sh 0

# 6) Deep + Aug-Simple / Aug-Full
AUG=1 AUG_MODE=simple bash scripts/train_ddp.sh 0
AUG=1 AUG_MODE=full   bash scripts/train_ddp.sh 0
```

评估每个实验（以 Deep + Aug-Simple 为例）：
```bash
CKPT=$(ls -t checkpoints/deep_aug-simple_best_*.pt | head -n1)
CHECKPOINT="$CKPT" bash scripts/eval.sh
```

## 目录结构 (Structure)
- `src/`：数据集类、增强、模型定义。
- `train.py`：训练入口（支持 DDP、增强、Dropout、baseline/deep）。
- `evaluate.py`：评估脚本（输出加权指标与混淆矩阵）。
- `scripts/`：训练/评估/作图/导出工具脚本。
- `splits/`：训练/验证集 CSV 与类别映射。
- `analysis/metrics|figures/`：评估 JSON 与可视化产物。
- `checkpoints/`：模型权重与训练历史。

## 复现与注意事项 (Reproducibility)
- 全局随机种子固定为 42（random/NumPy/torch/cuda）。
- 训练/评估均默认使用 `src/data/dataset_stats.json` 的 mean/std；若文件缺失会回退到 ImageNet 统计。
- 多卡训练使用 `torch.distributed.run`，脚本会据 `CUDA_VISIBLE_DEVICES` 自动设置进程数。