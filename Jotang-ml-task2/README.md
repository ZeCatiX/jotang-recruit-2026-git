# 招新 Task 2：猫狗分类

> 招新挑战 #2 — CNN 图像分类，权重 100 分，截止 2026/10/18 23:59

## 目录结构

```
Jotang-ml-task2/
├── README.md                 # 本文件
├── 思考题.md                  # 7 道趁热打铁思考题（含实验验证）
├── download_dataset.py       # 数据集下载脚本（Wikimedia Commons）
├── code/
│   ├── conv_practice.py      # 实践 1：手写卷积 + 4种卷积核
│   ├── cnn_train.py          # 实践 2：CNN 训练全流程
│   └── predict.py            # 独立推理程序
├── data/
│   ├── scene.png             # 用户提供的高速公路图片（卷积练习用）
│   ├── cats/                 # 猫图片（82 张）
│   └── dogs/                 # 狗图片（63 张）
├── figures/
│   ├── fig01_图片表示.png          # 图片的数字表示（RGB/灰度/像素）
│   ├── fig02_四种卷积核.png         # 均值模糊/高斯模糊/锐化/Sobel
│   ├── fig03_padding对比.png       # valid vs same padding
│   ├── fig04_彩色图卷积与裁剪.png   # 彩色图卷积 + 数值裁剪
│   ├── fig05_loss_acc曲线.png      # loss/accuracy 训练曲线
│   ├── fig06_数据集样本.png         # 数据集样本展示
│   ├── fig07_特征图可视化.png       # 浅层 vs 深层特征图
│   ├── fig08_数据增强.png           # 数据增强效果对比
│   ├── fig09_错误分析.png           # 错误样本分析
│   └── fig10_模型结构图.png        # CNN 模型结构图
└── results/
    ├── cnn_best.pt           # 最佳模型权重
    └── summary.json          # 实验结果汇总
```

## 实践 1：卷积核（手写）

**文件：** `code/conv_practice.py`

### 核心要求
- ✅ 手写 2D 卷积函数（不调用 cv2/scipy/torch.nn.Conv2d）
- ✅ 4 种卷积核：均值模糊 3×3、高斯模糊 5×5、锐化、Sobel 边缘检测
- ✅ valid vs same padding 对比 + 输出尺寸公式验证
- ✅ 彩色图卷积 + 数值裁剪

### 关键实现

```python
def conv2d_valid(image, kernel):
    """valid 模式：不填充，输出缩小"""
    kh, kw, _ = kernel.shape
    out_h = image.shape[0] - kh + 1
    out_w = image.shape[1] - kw + 1
    for i in range(out_h):
        for j in range(out_w):
            out[i, j] = np.sum(image[i:i+kh, j:j+kw] * kernel)
    return out

def conv2d_same(image, kernel):
    """same 模式：零填充，输出与输入同尺寸"""
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)))
    return conv2d_valid(padded, kernel)
```

### 输出尺寸公式

$$O = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1$$

实验验证：输入 5×5，kernel 3×3
- valid (P=0, S=1): O = (5-3+0)/1 + 1 = **3** ✓
- same (P=1, S=1): O = (5-3+2)/1 + 1 = **5** ✓
- stride 2 (P=0, S=2): O = (5-3)/2 + 1 = **2** ✓

## 实践 2：CNN 训练

**文件：** `code/cnn_train.py`

### 模型结构

```
SimpleCNN (2,191,618 参数)
├── Block 1: Conv2d(3→32, 3×3, pad=1) → BN → ReLU → MaxPool(2)
│   64×64×3 → 64×64×32 → 32×32×32
├── Block 2: Conv2d(32→64, 3×3, pad=1) → BN → ReLU → MaxPool(2)
│   32×32×32 → 32×32×64 → 16×16×64
├── Block 3: Conv2d(64→128, 3×3, pad=1) → BN → ReLU → MaxPool(2)
│   16×16×64 → 16×16×128 → 8×8×128
└── Classifier: Flatten → Linear(8192→256) → ReLU → Dropout(0.3) → Linear(256→2)
    8×8×128 → 8192 → 256 → 2
```

### 数据增强对照实验

| 配置 | 测试准确率 |
|------|-----------|
| 无增强 | 69.57% |
| 有增强（翻转+裁剪） | 65.22% |

> 注：数据集仅 145 张，增强带来的噪声可能大于正则化收益。

### 训练结果

| 指标 | 数值 |
|------|------|
| 数据集 | 145 张（82 猫 + 63 狗） |
| 训练/验证/测试 | 101 / 21 / 23 |
| 最佳验证准确率 | 80.95% |
| 测试准确率 | 60.87% |
| 训练 epochs | 30 |
| 学习率 | 1e-3 (Adam) |
| 批大小 | 32 |

### 独立推理程序

**文件：** `code/predict.py`

```bash
# 单张图片预测
python code/predict.py data/cats/cat_0001.jpg

# 批量预测（传入目录）
python code/predict.py data/cats/
```

输出示例：
```
==================================================
  图片: cat_0001.jpg
==================================================
  预测类别: 猫
  置信度:   0.9234 (92.34%)

  各类概率:
    猫: 0.9234 ████████████████████████████
    狗: 0.0766 ██
==================================================
```

## 趁热打铁

**文件：** `思考题.md`

7 道思考题全部配有实验代码验证：
1. 卷积层 vs 全连接层
2. 输入输出通道数、padding、stride 对尺寸的影响
3. 特征图高宽变小、通道增多的原因
4. 激活函数/损失函数选择
5. 验证集为什么不用数据增强
6. 增强变换的合理性
7. 如何改进模型（开放题）

## 拓展任务

### AlexNet vs VGG vs ResNet

| 模型 | 参数 | 深度 | 特点 | 适用场景 |
|------|------|------|------|---------|
| AlexNet | 60M | 8 层 | ReLU + Dropout + LRN | 开创深度学习时代 |
| VGG-16 | 138M | 16 层 | 全部 3×3 小卷积 | 理解卷积栈 |
| ResNet | 25M~255M | 50~152 层 | 残差连接跳过 | 解决梯度消失，可训练极深网络 |

### 预训练模型特征图 L2 范数可视化

使用 torchvision 的 ResNet-18 提取特征图，计算每个位置的 L2 范数：

```python
import torch.nn.functional as F
feat = model.layer1(x)  # 提取特征
l2_norm = F.normalize(feat, p=2, dim=1)  # L2 归一化
# 可视化各通道的激活强度
```

### BasicBlock vs Bottleneck

| 类型 | 结构 | 参数量 | 用途 |
|------|------|--------|------|
| BasicBlock | 3×3 → BN → ReLU → 3×3 → BN (+ shortcut) | 较少 | ResNet-18/34 |
| Bottleneck | 1×1 → 3×3 → 1×1 → BN (+ shortcut) | 较多但压缩 | ResNet-50/101/152 |

Bottleneck 用 1×1 卷积先降维再升维，在保持表达能力的同时减少参数量和计算量。

## 数据来源

数据集来自 Wikimedia Commons（https://commons.wikimedia.org），通过 `download_dataset.py` 自动下载缩略图构建。

- 猫：82 张
- 狗：63 张
- 图片来源：Wikimedia Commons 开放许可图片

## 运行环境

- Python 3.13.9
- PyTorch 2.11.0 (CPU)
- torchvision 0.26.0
- Pillow 12.3.0
- NumPy
- Matplotlib

## 运行步骤

```bash
# 1. 下载数据集（如已有数据可跳过）
python download_dataset.py

# 2. 运行卷积练习
python code/conv_practice.py

# 3. 训练 CNN
python code/cnn_train.py

# 4. 独立推理
python code/predict.py data/cats/cat_0001.jpg
```
