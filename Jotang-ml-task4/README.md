# 招新任务 4：Attention Is All You Need

> 100 分 | 截止：2026/10/18 23:59
> 要求：读懂论文，用自己的方法重绘结构图、手写注意力代码、回答思考题

## 目录结构

```
Jotang-ml-task4/
├── README.md              # 本文件
├── 阅读笔记.md            # 阅读笔记（资料来源、难点、理解过程、遗留疑问）
├── 思考题.md              # 8 道思考题回答（结合代码实验）
├── code/
│   ├── transformer_diagram.py    # 绘制 Transformer 结构图
│   ├── attention_single_head.py  # 手写单头 Scaled Dot-Product Attention
│   └── attention_multi_head.py   # 手写多头注意力
└── figures/
    ├── fig01_transformer_structure.png  # Transformer 整体架构（简化版）
    ├── fig02_attention_detail.png       # Attention 详细结构（含多头）
    ├── fig03_vit_structure.png          # ViT 结构图
    ├── fig04_attention_heatmap.png      # 单头注意力热力图
    └── fig05_multihead_heatmap.png      # 多头注意力热力图
```

## 完成内容

### 1. 重绘 Transformer 结构图

- **fig01**：Transformer 整体架构（Encoder-Decoder），标注了所有关键组件
- **fig02**：Attention 机制的详细结构，包括单头、多头、残差连接、LayerNorm
- **fig03**：ViT（Vision Transformer）的结构，展示 Patch Embedding → Encoder → 分类头

绘制工具：Matplotlib（`FancyBboxPatch` + `FancyArrowPatch`）

### 2. 手写 Scaled Dot-Product Attention（单头）

**文件**：`code/attention_single_head.py`

**核心实现**：
```python
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = np.dot(Q, K.T) / np.sqrt(d_k)   # 缩放点积
    weights = softmax(scores, axis=-1)          # softmax 归一化
    output = np.dot(weights, V)                 # 加权求和
    return output, scores, weights
```

**验证**：与 PyTorch `F.scaled_dot_product_attention` 对比，最大差异 ~2.87e-07（float64 vs float32 精度差异）

**可视化**：fig04 展示了注意力权重热力图，可以直观看到每个 query 关注哪些 key

### 3. 手写多头注意力

**文件**：`code/attention_multi_head.py`

**核心实现**：
```python
class MultiHeadAttention:
    def forward(self, Q, K, V):
        # 1. 线性投影
        Q_proj = np.dot(Q, self.W_Q)
        K_proj = np.dot(K, self.W_K)
        V_proj = np.dot(V, self.W_V)
        
        # 2. 拆分多头 (batch, seq, d_model) → (batch, h, seq, d_k)
        Q_heads = self._split_heads(Q_proj)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)
        
        # 3. 每个头独立计算 attention
        head_outputs = [single_head_attention(Q_heads[:, i], K_heads[:, i], V_heads[:, i]) for i in range(h)]
        
        # 4. Concat + 输出投影
        output = np.dot(self._merge_heads(head_outputs), self.W_O)
```

**关键设计**：用"聚类 + 位置编码"构造结构化输入，用不同随机种子初始化各头权重，成功展示了 **4 种不同的注意力模式**：

| 头 | 均匀度 | 模式 |
|---|-------|------|
| Head 1 | 中等 | 同聚类关注（A0→A1, C5→C4） |
| Head 2 | 41.87% | 跨聚类关注（A/B→C4, C/D→A0） |
| Head 3 | 22.28% | 配对模式（最集中） |
| Head 4 | 41.61% | 混合模式 |

**验证**：与 PyTorch 参考实现对比，最大差异 ~2.70e-06

### 4. ViT 结构说明

在 `阅读笔记.md` 和 `思考题.md`（Q8）中详细说明了 ViT 与原始 Transformer 的区别，并用 fig03 绘制了 ViT 的完整流程图。

### 5. 阅读笔记

**文件**：`阅读笔记.md`

包含：
- 资料来源（论文、代码、视频、博客）
- 理解过程（从迷茫到理解的 4 个阶段）
- 遇到的难点（Q/K/V 来源、mask 方向、LN 顺序、多头等价性）
- 代码实践中的发现（精度差异、注意力均匀性、np.dot vs np.matmul）
- 遗留疑问（√d_k 最优性、外推能力、头塌缩、patch 大小、O(n²) 优化）

### 6. 思考题回答

**文件**：`思考题.md`

8 道题覆盖 Transformer 的核心概念：
1. 为什么除以 √d_k？（方差稳定 + 代码验证）
2. 多头 vs 单头？（不同子空间 + 4 种模式实证）
3. sin/cos vs 可学习位置编码？（平移性质 + 外推 + 无参数）
4. Encoder vs Decoder？（结构对比 + Mask + Cross-Attention）
5. Masked Self-Attention？（上三角 mask + 训练推理一致性）
6. Cross-Attention？（Encoder-Decoder 桥梁 + 翻译对齐）
7. 残差 + LayerNorm？（梯度消失 + 数值稳定 + Post-LN vs Pre-LN）
8. ViT vs Transformer？（去 Decoder + Patch Embedding + 全局感受野）

## 代码运行方式

```bash
# 运行所有代码（需要 Python 3.13+, numpy, matplotlib, torch）
cd Jotang-ml-task4

# 绘制 Transformer 结构图（fig01, fig02, fig03）
python code/transformer_diagram.py

# 单头注意力 + 热力图（fig04）
python code/attention_single_head.py

# 多头注意力 + 热力图（fig05）
python code/attention_multi_head.py
```

## 关键学习收获

1. **Attention 的本质**：可微的加权求和，权重由 Q·K 相似度决定
2. **多头的意义**：在不同子空间学习不同模式（实测 4 头 4 种模式）
3. **位置编码**：sin/cos 具有平移不变性，支持序列长度外推
4. **架构灵活性**：Encoder-only（BERT/ViT）、Decoder-only（GPT）、Encoder-Decoder（翻译）
5. **踩坑经验**：`np.dot` vs `np.matmul` 在 3D 数组上的差异；随机权重下注意力接近均匀分布

---

*完成日期：2026-10-08*
