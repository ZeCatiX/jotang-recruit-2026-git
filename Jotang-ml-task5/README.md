# 招新任务 5：VLM 从这开始 —— CLIP

> 100 分 | 截止：2026/10/18 23:59
> 要求：阅读 CLIP 论文，理解图文对齐和对比学习，完成图文配对实验

## 目录结构

```
Jotang-ml-task5/
├── README.md                    # 本文件
├── 阅读笔记.md                   # CLIP 论文阅读笔记（8 个问题）
├── 思考题.md                    # 7 道思考题回答（结合代码实验）
├── code/
│   ├── clip_experiment.py       # CLIP 图文配对实验（4 个实验）
│   └── clip_diagram.py          # CLIP 结构图绘制
├── images/                      # 4 张测试图片
│   ├── img1_cat.jpg             # 猫
│   ├── img2_dog.jpg             # 狗
│   ├── img3_eiffel.jpg          # 埃菲尔铁塔
│   └── img4_pizza.jpg           # 披萨
├── figures/
│   ├── fig01_similarity_heatmap_correct.png    # 正确描述配对热力图
│   ├── fig02_similarity_heatmap_misleading.png # 误导性描述配对热力图
│   ├── fig03_comparison_correct_vs_misleading.png  # 正确 vs 误导性对比
│   ├── fig04_temperature_effect.png            # 温度系数对 softmax 的影响
│   ├── fig05_clip_architecture.png             # CLIP 整体架构
│   ├── fig06_contrastive_learning.png          # N×N 对比学习示意
│   └── fig07_zeroshot_classification.png       # Zero-shot 分类流程
└── results/
    └── clip_experiment_results.json  # 实验结果数据
```

## 完成内容

### 1. 阅读笔记

**文件**：`阅读笔记.md`

覆盖论文 8 个核心问题：
1. 传统分类 vs CLIP 的数据需求
2. 两个编码器的组成和输出
3. 图文对齐的含义和原理
4. N×N 配对与正负样本
5. InfoNCE 对比损失与温度系数
6. Zero-shot 分类的原理
7. Prompt engineering 的重要性
8. CLIP 的限制与数据偏见

### 2. CLIP 结构图

**代码**：`code/clip_diagram.py`

绘制了 3 张结构图：
- **fig05**：CLIP 整体架构（两个编码器 + 对比损失）
- **fig06**：N×N 对比学习示意（正样本对 vs 负样本对）
- **fig07**：Zero-shot 分类流程

### 3. CLIP 图文配对实验

**代码**：`code/clip_experiment.py`

使用 OpenAI CLIP RN50 模型，完成 4 个实验：

#### 实验 1：正确描述配对

- 4 张图片 + 4 条正确描述
- 计算 4×4 余弦相似度矩阵
- 结果：**全部配对正确** ✅
  - Cat → "a domestic cat" (0.1926)
  - Dog → "a dog walking" (0.1351)
  - Eiffel Tower → "the Eiffel Tower" (0.2254)
  - Pizza → "a pizza with cheese" (0.2092)

#### 实验 2：误导性描述

- 用相似但错误的描述替换（颜色/数量/动作错误）
- 结果：**2/4 被误导** ⚠️
  - Cat 被 "a cat walking in a park" 误导（关键词 "cat" 主导）
  - Dog 被 "a tabby dog sitting" 误导
  - Eiffel Tower 和 Pizza 仍正确区分

#### 实验 3：正确 vs 误导性对比

- 混合正确和错误描述（4+4=8 条）
- 结果：**3/4 正确区分**
  - Cat 被误导（错误描述相似度更高）
  - Dog、Eiffel Tower、Pizza 正确区分

#### 实验 4：温度系数实验

- 测试 τ 从 0.01 到 1.00 的 softmax 分布变化
- 结果：
  - τ=0.01：max_prob=0.863（过度集中）
  - τ=0.07（CLIP 默认）：max_prob=0.434（平衡）
  - τ=1.00：max_prob=0.264（接近均匀）

### 4. 思考题回答

**文件**：`思考题.md`

7 道题覆盖 CLIP 核心概念：
1. 为什么图文特征可以计算相似度？（共享特征空间 + 对比学习）
2. 为什么 L2 归一化？（点积 = 余弦相似度）
3. 为什么双向损失？（对称对齐，两个编码器都被优化）
4. Zero-shot 的真正含义？（不在目标任务训练，但见过类似内容）
5. 温度系数的作用？（控制 softmax 锐度）
6. 数据偏见的捷径？（学到虚假关联而非真正语义）
7. CLIP 的应用场景？（8 大场景：分类、检索、去重、异常检测等）

## 代码运行方式

```bash
# 环境要求：Python 3.13+, torch, openai-clip, matplotlib, torchvision
# 安装 CLIP
pip install openai-clip ftfy regex tqdm

cd Jotang-ml-task5

# 运行 CLIP 实验（需要下载 RN50 模型，约 244MB）
python code/clip_experiment.py

# 绘制 CLIP 结构图
python code/clip_diagram.py
```

## 图片来源

| 图片 | 来源 |
|------|------|
| img1_cat.jpg | Wikimedia Commons (task2 数据集) |
| img2_dog.jpg | Wikimedia Commons (task2 数据集) |
| img3_eiffel.jpg | Wikimedia Commons (File:Tour_Eiffel_Wikimedia_Commons.jpg) |
| img4_pizza.jpg | Wikimedia Commons (File:Pizza_margherita_at_Don_Pizzaiolo.jpg) |

## 关键实验发现

### 发现 1：CLIP 可以被误导性描述欺骗

**现象**：Cat 图片与 "a cat walking in a park"（错误描述）的相似度（0.2501）高于正确描述 "a domestic cat sitting on a couch"（0.1926）。

**原因**：两个描述都包含关键词 "cat"，CLIP 对关键词敏感，忽略了场景差异（couch vs park）。

**启示**：这类似于"数据偏见"——模型学到捷径（关键词匹配）而非真正语义理解。

### 发现 2：温度系数显著影响 softmax 分布

**τ=0.07（CLIP 默认）**：
- max_prob=0.434，熵=1.717 bits
- 分布适中，能区分正负样本

**τ 变化趋势**：
- τ 越小 → 越确信（但梯度可能消失）
- τ 越大 → 越不确定（但难以区分）

### 发现 3：CLIP 的零样本分类能力

- 4/4 正确描述配对成功
- 3/4 正确区分正确 vs 误导性描述
- 说明 CLIP 有很强的图文对齐能力，但对关键词捷径敏感

## 关键学习收获

1. **图文对齐**：CLIP 用对比学习把图文映射到共享特征空间
2. **对比损失**：InfoNCE 双向计算，提高正样本、降低负样本相似度
3. **温度系数**：控制 softmax 锐度，τ=0.07 是平衡点
4. **Zero-shot 分类**：用类名文本做"查询"，无需训练
5. **数据偏见**：模型会学到虚假关联（如"雪地"→"北极熊"）

---

*完成日期：2026-10-08*
