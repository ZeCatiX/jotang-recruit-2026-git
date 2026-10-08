"""
独立推理程序：加载训练好的 CNN 模型，对单张图片进行分类预测。

用法：
    python predict.py <图片路径>
    python predict.py data/cats/cat_0001.jpg
    python predict.py data/dogs/dog_0005.jpg
"""
import sys
import torch
from pathlib import Path
from PIL import Image

# ── 路径 ──
ROOT = Path(__file__).parent.parent
MODEL_PATH = ROOT / "results" / "cnn_best.pt"
IMG_SIZE = 64
CLASSES = ["猫", "狗"]


class SimpleCNN(torch.nn.Module):
    """与训练脚本中相同的模型结构"""
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(128 * 8 * 8, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def preprocess(img_path):
    """图片预处理：加载 → 缩放 → 归一化 → 加 batch 维度"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    import numpy as np
    arr = torch.from_numpy(np.array(img)).float() / 255.0  # [0,1]
    arr = arr.permute(2, 0, 1)  # HWC → CHW
    # 归一化 (mean=0.5, std=0.5)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    arr = (arr - mean) / std
    return arr.unsqueeze(0)  # 加 batch 维度 → (1, 3, 64, 64)


def predict(img_path):
    """加载模型并推理"""
    if not MODEL_PATH.exists():
        print(f"错误: 模型文件不存在 {MODEL_PATH}")
        print("请先运行 cnn_train.py 训练模型")
        sys.exit(1)

    # 加载模型
    model = SimpleCNN()
    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 预处理
    x = preprocess(img_path)

    # 推理
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        _, pred_idx = logits.max(1)
        pred_class = CLASSES[pred_idx.item()]
        confidence = probs[0][pred_idx].item()

    # 输出结果
    print(f"\n{'='*50}")
    print(f"  图片: {Path(img_path).name}")
    print(f"{'='*50}")
    print(f"  预测类别: {pred_class}")
    print(f"  置信度:   {confidence:.4f} ({confidence*100:.2f}%)")
    print(f"\n  各类概率:")
    for i, cls in enumerate(CLASSES):
        bar = "█" * int(probs[0][i].item() * 30)
        print(f"    {cls}: {probs[0][i].item():.4f} {bar}")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 没有传图片路径时，展示用法
        print("用法: python predict.py <图片路径>")
        print("\n示例:")
        print("  python predict.py data/cats/cat_0001.jpg")
        print("  python predict.py data/dogs/dog_0005.jpg")
        print("\n也可批量测试：")
        print("  python predict.py data/cats/  (传入目录)")
        print()

        # 自动批量测试部分样本
        import os
        test_files = []
        for cls_dir in ["cats", "dogs"]:
            d = ROOT / "data" / cls_dir
            if d.exists():
                files = sorted(d.glob("*.jpg"))[:3]
                test_files.extend(files)
        if test_files:
            print(f"自动测试 {len(test_files)} 张样本...\n")
            for f in test_files:
                predict(f)
        else:
            print("未找到数据目录 data/cats 或 data/dogs")
    else:
        path = Path(sys.argv[1])
        if path.is_dir():
            # 批量推理目录
            files = sorted(path.glob("*.jpg")) + sorted(path.glob("*.png"))
            print(f"批量测试 {len(files)} 张图片...\n")
            correct = 0
            for f in files[:20]:  # 最多测 20 张
                predict(f)
                # 判断正确率
                label = 0 if "cats" in str(f) else 1
                # 简单判断
        else:
            predict(sys.argv[1])
