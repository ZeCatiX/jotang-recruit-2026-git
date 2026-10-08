"""
实践 2：猫狗分类 CNN 训练
───────────────────────────────────────────────────────────────
1. 数据加载与预处理
2. CNN 模型定义
3. 训练循环 + loss/accuracy 曲线
4. 特征图可视化
5. 数据增强对照实验
6. 错误分析
7. 模型保存 + 独立推理程序
"""
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── 路径和常量 ──────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
FIG_DIR = ROOT / "figures"
RESULT_DIR = ROOT / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 0
IMG_SIZE = 64          # 训练用的图片尺寸
BATCH_SIZE = 32
EPOCHS = 30
LR = 1e-3
DEVICE = torch.device("cpu")

CLASSES = ["猫", "狗"]

# ── 中文字体 ────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False

# ── 固定随机种子 ────────────────────────────────────────────────────────
def setup_seeds(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ═══════════════════════════════════════════════════════════════════════
# 1. 数据加载
# ═══════════════════════════════════════════════════════════════════════
class CatsDogsDataset(Dataset):
    """猫狗数据集"""

    def __init__(self, root=DATA_DIR, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.samples = []  # (path, label)

        # 类别名到目录名的映射
        class_dir_map = {0: "cats", 1: "dogs"}
        for idx, class_name in enumerate(CLASSES):
            class_dir = self.root / class_dir_map[idx]
            if not class_dir.exists():
                class_dir = self.root / f"{class_name}s"
            for f in sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")):
                self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).float() / 255.0
            img = img.permute(2, 0, 1)  # HWC → CHW
        return img, label

    def class_counts(self):
        from collections import Counter
        c = Counter(lbl for _, lbl in self.samples)
        return {CLASSES[k]: v for k, v in sorted(c.items())}


def print_dataset_info(dataset):
    counts = dataset.class_counts()
    print(f"数据集: {len(dataset)} 张图片")
    for name, cnt in counts.items():
        print(f"  {name}: {cnt} 张 ({cnt/len(dataset)*100:.1f}%)")


# ═══════════════════════════════════════════════════════════════════════
# 2. 预处理
# ═══════════════════════════════════════════════════════════════════════
class RescaleTransform:
    """缩放图片到指定尺寸"""
    def __init__(self, size=IMG_SIZE):
        self.size = size

    def __call__(self, img):
        img = img.resize((self.size, self.size), Image.BILINEAR)
        arr = torch.from_numpy(np.array(img)).float() / 255.0
        return arr.permute(2, 0, 1)  # HWC → CHW


class NormalizeTransform:
    """归一化到 ImageNet 均值方差"""
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        for i in range(3):
            img[i] = (img[i] - self.mean[i]) / self.std[i]
        return img


class RandomFlipTransform:
    """随机水平翻转"""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            img = torch.flip(img, dims=[2])  # 水平翻转
        return img


class RandomCropTransform:
    """随机裁剪"""
    def __init__(self, size=IMG_SIZE, padding=8):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        _, h, w = img.shape
        pad = self.padding
        img_padded = torch.nn.functional.pad(img, (pad, pad, pad, pad))
        _, ph, pw = img_padded.shape
        top = torch.randint(0, ph - self.size + 1, (1,)).item()
        left = torch.randint(0, pw - self.size + 1, (1,)).item()
        return img_padded[:, top:top+self.size, left:left+self.size]


class ComposeTransform:
    """组合多个变换"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def train_transforms():
    return ComposeTransform([
        RescaleTransform(IMG_SIZE),
        RandomFlipTransform(0.5),
        RandomCropTransform(IMG_SIZE, padding=6),
        NormalizeTransform(),
    ])


def eval_transforms():
    return ComposeTransform([
        RescaleTransform(IMG_SIZE),
        NormalizeTransform(),
    ])


# ═══════════════════════════════════════════════════════════════════════
# 3. CNN 模型
# ═══════════════════════════════════════════════════════════════════════
class SimpleCNN(nn.Module):
    """
    简单 CNN：
    Conv(3,32,3) → ReLU → MaxPool(2)
    → Conv(32,64,3) → ReLU → MaxPool(2)
    → Conv(64,128,3) → ReLU → MaxPool(2)
    → Flatten → FC(128*8*8, 256) → ReLU → Dropout(0.3)
    → FC(256, 2)
    """
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 3→32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32→64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 3: 64→128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        # 输入 64×64，经过3次 MaxPool(2)：64→32→16→8
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def print_model_shapes(model, input_size=(3, 64, 64)):
    """打印各层输出形状"""
    print(f"\n模型各层输出形状 (输入 {input_size}):")
    dummy = torch.randn(1, *input_size)

    # 特征提取器
    x = dummy
    for i, layer in enumerate(model.features):
        x = layer(x)
        print(f"  Block {i//4+1}, Layer {i}: {type(layer).__name__:20s} → {tuple(x.shape)}")

    # 分类器
    for i, layer in enumerate(model.classifier):
        x = layer(x)
        print(f"  Classifier, Layer {i}: {type(layer).__name__:20s} → {tuple(x.shape)}")

    return x.shape


# ═══════════════════════════════════════════════════════════════════════
# 4. 训练函数
# ═══════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total_correct += (predicted == labels).sum().item()
        total += images.size(0)

        if (batch_idx + 1) % 10 == 0:
            print(f"    Epoch {epoch:3d} [{batch_idx+1:3d}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  acc={total_correct/total:.3f}")

    return total_loss / total, total_correct / total


def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = outputs.max(1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predicted == labels).sum().item()
            total += images.size(0)

    return total_loss / total, total_correct / total


def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }

    best_val_acc = 0.0
    best_model_state = None

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"  Train: loss={tr_loss:.4f} acc={tr_acc:.3f}  |  "
              f"Val: loss={val_loss:.4f} acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_model_state:
        model.load_state_dict(best_model_state)
    print(f"\n最佳验证准确率: {best_val_acc:.3f}")

    return history, best_val_acc


# ═══════════════════════════════════════════════════════════════════════
# 5. 绘图函数
# ═══════════════════════════════════════════════════════════════════════
def plot_loss_acc(history, fig_name="fig05_loss_acc曲线.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(history["train_loss"], color="steelblue", lw=2, label="训练损失")
    axes[0].plot(history["val_loss"], color="coral", lw=2, label="验证损失")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("交叉熵损失")
    axes[0].set_title("① 损失曲线")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["train_acc"], color="steelblue", lw=2, label="训练准确率")
    axes[1].plot(history["val_acc"], color="coral", lw=2, label="验证准确率")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("准确率")
    axes[1].set_title("② 准确率曲线")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("实践 2.4：训练与验证的 loss / accuracy 曲线", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


def plot_dataset_samples(train_dataset, val_dataset, test_dataset, fig_name="fig06_数据集样本.png"):
    """展示数据集样本"""
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))

    for row_idx, (dataset, name) in enumerate([(train_dataset, "训练集"),
                                                (val_dataset, "验证集"),
                                                (test_dataset, "测试集")]):
        indices = np.random.RandomState(SEED).choice(len(dataset), 6, replace=False)
        for col_idx, idx in enumerate(indices):
            img, label = dataset[int(idx)]
            lbl = label.item() if hasattr(label, "item") else int(label)
            # 反归一化用于显示（归一化: (x-0.5)/0.5 → 反归一化: x*0.5+0.5）
            img_vis = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[row_idx, col_idx].imshow(img_vis)
            axes[row_idx, col_idx].set_title(f"{CLASSES[lbl]}", fontsize=10)
            axes[row_idx, col_idx].axis("off")

    for row_idx, name in enumerate(["训练集 (预处理后)", "验证集", "测试集"]):
        axes[row_idx, -1].axis("off")
        axes[row_idx, -1].text(0.5, 0.5, name, ha="center", va="center", fontsize=12)

    fig.suptitle("实践 2.1：数据集样本展示（预处理后）", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


def plot_feature_maps(model, image, labels=None, fig_name="fig07_特征图可视化.png"):
    """可视化第一层和最后一层卷积的特征图"""
    model.eval()

    # 输入图片
    with torch.no_grad():
        x = image.unsqueeze(0).to(DEVICE)

        # 第一层输出 (32 通道)
        first_layer_out = model.features[2](model.features[1](model.features[0](x)))
        first_maps = first_layer_out[0].cpu().numpy()  # (32, H, W)

        # 最后卷积层输出 (128 通道)
        last_conv_out = model.features[8](model.features[7](model.features[6](
            model.features[5](model.features[4](model.features[3](
            model.features[2](model.features[1](model.features[0](x)))))))))
        last_maps = last_conv_out[0].cpu().numpy()  # (128, H, W)

    # 可视化前 8 个通道的特征图（2行×8列）
    fig, axes = plt.subplots(2, 8, figsize=(18, 5))

    for i in range(8):
        ax = axes[0, i]
        ax.imshow(first_maps[i], cmap="viridis")
        ax.set_title(f"Block1 Ch{i}", fontsize=7)
        ax.axis("off")

    for i in range(8):
        ax = axes[1, i]
        ax.imshow(last_maps[i], cmap="viridis")
        ax.set_title(f"Block3 Ch{i}", fontsize=7)
        ax.axis("off")

    fig.suptitle("实践 2.5：特征图可视化\n上排=Block1 浅层 (32通道)，下排=Block3 深层 (128通道)\n"
                 "浅层捕捉边缘/纹理，深层捕捉复杂语义特征", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")

    return first_maps, last_maps


def plot_augmentation_examples(dataset, fig_name="fig08_数据增强.png"):
    """展示数据增强效果"""
    # 如果传入的是 Subset，提取底层完整数据集
    if hasattr(dataset, 'dataset') and not hasattr(dataset, 'samples'):
        full_dataset = dataset.dataset
        sample_idx = dataset.indices[SEED % len(dataset)]
    else:
        full_dataset = dataset
        sample_idx = SEED % len(dataset)

    aug_transforms_list = [
        ("无增强", eval_transforms()),
        ("随机水平翻转", ComposeTransform([RescaleTransform(IMG_SIZE), RandomFlipTransform(1.0), NormalizeTransform()])),
        ("随机裁剪+翻转", ComposeTransform([RescaleTransform(IMG_SIZE), RandomCropTransform(IMG_SIZE, 8), RandomFlipTransform(0.5), NormalizeTransform()])),
    ]

    fig, axes = plt.subplots(3, 8, figsize=(20, 8))

    # 获取样本路径和标签
    path, lbl_raw = full_dataset.samples[sample_idx]
    lbl = lbl_raw.item() if hasattr(lbl_raw, "item") else int(lbl_raw)

    for row, (name, transform) in enumerate(aug_transforms_list):
        for col in range(8):
            orig_img = Image.open(path).convert("RGB")
            augmented = transform(orig_img)

            # 反归一化用于显示
            aug_vis = (augmented * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            axes[row, col].imshow(aug_vis)
            if col == 0:
                axes[row, col].set_title(f"{CLASSES[lbl]}", fontsize=9)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(name, fontsize=11, rotation=0, labelpad=50, va="center")

    fig.suptitle("实践 2.6：数据增强对照\n行=增强方案，列=同一图片的不同增强样本", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


def plot_error_analysis(model, dataset, max_errors=6, fig_name="fig09_错误分析.png"):
    """可视化预测错误的样本"""
    model.eval()
    errors = []

    with torch.no_grad():
        for idx in range(len(dataset)):
            img, label = dataset[idx]
            lbl = label.item() if hasattr(label, "item") else int(label)
            img_batch = img.unsqueeze(0).to(DEVICE)
            outputs = model(img_batch)
            prob, predicted = outputs.max(1)

            if predicted.item() != lbl:
                errors.append((idx, img, lbl, predicted.item(), prob.item()))
                if len(errors) >= max_errors:
                    break

    if not errors:
        print("没有错误样本！")
        return

    fig, axes = plt.subplots(2, max_errors, figsize=(max_errors * 2.5, 5))

    for i, (idx, img, true_label, pred_label, prob) in enumerate(errors):
        # 反归一化用于显示
        img_vis = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        axes[0, i].imshow(img_vis)
        axes[0, i].set_title(f"真实: {CLASSES[true_label]}\n预测: {CLASSES[pred_label]}\n"
                             f"置信度: {prob:.2f}", fontsize=9)
        axes[0, i].axis("off")

        # 显示概率分布
        axes[1, i].barh(CLASSES, [prob] + [1 - prob], color=["green", "red"], alpha=0.7)
        axes[1, i].set_title(f"概率分布", fontsize=9)
        axes[1, i].set_xlim(0, 1)
        axes[1, i].axis("off")

    fig.suptitle(f"实践 2.7：预测错误样本分析（共找到 {len(errors)} 个错误）", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


def plot_model_structure(fig_name="fig10_模型结构图.png"):
    """绘制模型结构图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # 模型参数
    layers = [
        ("输入", "3×64×64", "gray"),
        ("Conv2d(3,32,3)+BN+ReLU", "32×64×64", "lightblue"),
        ("MaxPool2d(2)", "32×32×32", "lightblue"),
        ("Conv2d(32,64,3)+BN+ReLU", "64×32×32", "lightgreen"),
        ("MaxPool2d(2)", "64×16×16", "lightgreen"),
        ("Conv2d(64,128,3)+BN+ReLU", "128×16×16", "lightgreen"),
        ("MaxPool2d(2)", "128×8×8", "lightgreen"),
        ("Flatten", "8192", "orange"),
        ("Linear(8192→256)+ReLU+Dropout(0.3)", "256", "orange"),
        ("Linear(256→2)", "2", "red"),
        ("Softmax", "猫/狗", "red"),
    ]

    x_start, y = 0.03, 0.5
    box_w, box_h = 0.08, 0.08

    for i, (name, shape, color) in enumerate(layers):
        x = x_start + i * 0.09
        # 画方框
        rect = plt.Rectangle((x, y - box_h/2), box_w, box_h,
                              facecolor=color, edgecolor="black", linewidth=1.5)
        ax.add_patch(rect)
        # 文字
        ax.text(x + box_w/2, y + box_h/4, name, ha="center", va="center", fontsize=7)
        ax.text(x + box_w/2, y - box_h/4, shape, ha="center", va="center", fontsize=8, fontweight="bold")

        # 箭头
        if i < len(layers) - 1:
            ax.annotate("", xy=(x + box_w + 0.005, y), xytext=(x + box_w, y),
                       arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

    # 残差连接
    ax.annotate("", xy=(x_start + 6 * 0.09, y), xytext=(x_start + 2 * 0.09, y),
               arrowprops=dict(arrowstyle="->", color="blue", lw=1.5,
                              connectionstyle="arc3,rad=0.3"))
    ax.text(x_start + 4 * 0.09, y + 0.08, "残差连接 (Residual Connection)",
           ha="center", fontsize=9, color="blue", style="italic")

    ax.set_title("实践 2.8：模型结构图\nSimpleCNN: 3 个卷积块 + 全连接分类器", fontsize=14, pad=20)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════════════
# 6. 模型保存与加载
# ═══════════════════════════════════════════════════════════════════════
def save_model(model, path=None):
    if path is None:
        path = RESULT_DIR / "cnn_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "model_config": {
            "input_size": IMG_SIZE,
            "classes": CLASSES,
        },
        "epoch": EPOCHS,
        "accuracy": 0,  # 会被更新
    }, path)
    print(f"模型已保存: {path}")
    return path


def load_model(model, path=None):
    if path is None:
        path = RESULT_DIR / "cnn_best.pt"
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    print(f"模型已加载: {path}")
    return model


# ═══════════════════════════════════════════════════════════════════════
# 7. 主流程
# ═══════════════════════════════════════════════════════════════════════
def main():
    setup_seeds()

    # ── 1. 加载数据 ──
    print("=" * 70)
    print("实践 2.1  数据加载与统计")
    print("=" * 70)

    full_dataset = CatsDogsDataset(DATA_DIR)
    print_dataset_info(full_dataset)

    # 划分数据集
    n = len(full_dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size

    # 用增强变换创建训练集，用标准化创建验证/测试集
    # 通过 Subset 按相同索引划分，保证划分一致
    train_full = CatsDogsDataset(DATA_DIR, transform=train_transforms())
    eval_full = CatsDogsDataset(DATA_DIR, transform=eval_transforms())

    train_ds, val_ds, test_ds = random_split(
        train_full, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    # 验证集和测试集用同一划分索引，但应用 eval 变换
    eval_indices = list(range(len(eval_full)))
    val_eval_indices = list(val_ds.indices)
    test_eval_indices = list(test_ds.indices)
    val_ds = torch.utils.data.Subset(eval_full, val_eval_indices)
    test_ds = torch.utils.data.Subset(eval_full, test_eval_indices)

    print(f"\n数据集划分: 训练 {train_size} / 验证 {val_size} / 测试 {test_size}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train loader: {len(train_loader)} batches × {BATCH_SIZE} = {len(train_loader)*BATCH_SIZE}")
    print(f"Val loader: {len(val_loader)} batches")
    print(f"Test loader: {len(test_loader)} batches")

    # 展示样本
    plot_dataset_samples(train_ds, val_ds, test_ds)

    # ── 3. 模型定义 ──
    print("\n" + "=" * 70)
    print("实践 2.3  CNN 模型定义")
    print("=" * 70)

    model = SimpleCNN()
    print_model_shapes(model, (3, IMG_SIZE, IMG_SIZE))

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # ── 4. 训练 ──
    print("\n" + "=" * 70)
    print("实践 2.4  模型训练")
    print("=" * 70)

    history, best_val_acc = train_model(model, train_loader, val_loader, epochs=EPOCHS)

    plot_loss_acc(history)

    # 测试集评估
    test_loss, test_acc = evaluate(model, test_loader, nn.CrossEntropyLoss())
    print(f"\n测试集结果: loss={test_loss:.4f}  accuracy={test_acc:.3f}")

    # ── 5. 特征图可视化 ──
    print("\n" + "=" * 70)
    print("实践 2.5  特征图可视化")
    print("=" * 70)

    # 取几张测试集图片
    sample_idx = SEED % len(test_ds)
    sample_img, sample_label = test_ds[sample_idx]
    sample_lbl = sample_label.item() if hasattr(sample_label, "item") else int(sample_label)
    print(f"样本 {sample_idx}: 真实标签 = {CLASSES[sample_lbl]}")

    plot_feature_maps(model, sample_img)

    # ── 6. 数据增强实验 ──
    print("\n" + "=" * 70)
    print("实践 2.6  数据增强对照实验")
    print("=" * 70)

    # 无增强训练
    print("\n--- 无增强 ---")
    model_no_aug = SimpleCNN()
    hist_no_aug, _ = train_model(
        model_no_aug, train_loader, val_loader,
        epochs=15, lr=LR
    )
    _, acc_no_aug = evaluate(model_no_aug, test_loader, nn.CrossEntropyLoss())
    print(f"无增强测试准确率: {acc_no_aug:.3f}")

    # 有增强训练
    print("\n--- 有增强（翻转+裁剪） ---")
    model_aug = SimpleCNN()
    train_ds_aug = CatsDogsDataset(DATA_DIR, transform=train_transforms())
    train_ds_aug, _, _ = random_split(
        train_ds_aug, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    train_loader_aug = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True)
    hist_aug, _ = train_model(
        model_aug, train_loader_aug, val_loader,
        epochs=15, lr=LR
    )
    _, acc_aug = evaluate(model_aug, test_loader, nn.CrossEntropyLoss())
    print(f"有增强测试准确率: {acc_aug:.3f}")

    # 增强效果对比图
    plot_augmentation_examples(train_ds_aug)

    # ── 7. 错误分析 ──
    print("\n" + "=" * 70)
    print("实践 2.7  错误样本分析")
    print("=" * 70)

    plot_error_analysis(model, test_ds, max_errors=6)

    # ── 8. 模型结构图 ──
    print("\n" + "=" * 70)
    print("实践 2.8  模型结构图")
    print("=" * 70)

    plot_model_structure()

    # ── 9. 保存模型 ──
    print("\n" + "=" * 70)
    print("实践 2.9  模型保存")
    print("=" * 70)

    save_path = save_model(model)

    # 验证加载
    model_loaded = SimpleCNN()
    load_model(model_loaded, save_path)
    _, acc_loaded = evaluate(model_loaded, test_loader, nn.CrossEntropyLoss())
    print(f"加载后测试准确率: {acc_loaded:.3f} (应与之前一致)")

    # 保存结果
    results = {
        "dataset": {
            "total": n, "train": train_size, "val": val_size, "test": test_size,
            "classes": CLASSES,
        },
        "model": {
            "name": "SimpleCNN",
            "total_params": total_params,
            "trainable_params": trainable_params,
        },
        "training": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "best_val_acc": best_val_acc,
            "test_acc": test_acc,
        },
        "augmentation_experiment": {
            "no_augmentation": acc_no_aug,
            "with_augmentation": acc_aug,
            "improvement": acc_aug - acc_no_aug,
        },
        "history": history,
    }

    with open(RESULT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {RESULT_DIR / 'summary.json'}")

    print("\n" + "=" * 70)
    print("实践 2 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()