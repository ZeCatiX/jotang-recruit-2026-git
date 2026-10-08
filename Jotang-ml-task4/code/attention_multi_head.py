"""
手写多头注意力 (Multi-Head Attention)

在单头基础上拓展：
  1. 多个头并行计算
  2. 每个头有独立的 W_Q, W_K, W_V, W_O
  3. 结果 Concat + Linear 投影

参考论文公式：
  MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O
  其中 head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

为了让"不同头关注不同模式"这一特性在演示中可见，本脚本：
  - 用"聚类 + 位置编码"构造有结构的输入（而非纯随机），
    让 Q·K 的内积出现真实差异；
  - 每个头用不同的随机种子初始化投影，让 4 个头落在不同的子空间。
纯随机权重 + 纯随机输入会让所有头的 softmax 都接近均匀分布，失去教学价值。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════
# 1. softmax
# ═══════════════════════════════════════════════════════════════
def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ═══════════════════════════════════════════════════════════════
# 2. 单头 Attention（复用）
# ═══════════════════════════════════════════════════════════════
def single_head_attention(Q, K, V, mask=None):
    """
    单头 Scaled Dot-Product Attention
    Q, K: (batch, seq, d_k), V: (batch, seq, d_k)
    """
    d_k = Q.shape[-1]
    scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

    if mask is not None:
        scores = scores + (1 - mask) * (-1e9)

    weights = softmax(scores, axis=-1)
    output = np.matmul(weights, V)
    return output, scores, weights


# ═══════════════════════════════════════════════════════════════
# 3. 多头注意力
# ═══════════════════════════════════════════════════════════════
class MultiHeadAttention:
    """
    多头注意力

    参数:
        d_model: 模型维度
        n_heads: 注意力头数
        base_seed: 随机种子（每个头用 base_seed + head_idx 作为偏移，
                   确保 4 个头的投影落在不同子空间）
    """
    def __init__(self, d_model, n_heads, base_seed=42):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度

        # 每个头独立初始化，再沿输出维拼接成 (d_model, d_model)
        # —— 与论文的"单一 W_Q 再 reshape"在数学上等价，
        #    但能显式地让不同头有独立的随机起点。
        W_Q_heads, W_K_heads, W_V_heads = [], [], []
        for h in range(n_heads):
            rng = np.random.RandomState(base_seed + h * 1000)
            W_Q_heads.append(rng.randn(d_model, self.d_k) * 0.3)
            W_K_heads.append(rng.randn(d_model, self.d_k) * 0.3)
            W_V_heads.append(rng.randn(d_model, self.d_k) * 0.3)

        # 拼接成 (d_model, d_model)，方便与前向的 split_heads 配合
        self.W_Q = np.concatenate(W_Q_heads, axis=1)
        self.W_K = np.concatenate(W_K_heads, axis=1)
        self.W_V = np.concatenate(W_V_heads, axis=1)

        # 输出投影
        rng = np.random.RandomState(base_seed)
        self.W_O = rng.randn(d_model, d_model) * 0.02

        self.head_weights = None  # 保存各头的权重用于可视化

    def forward(self, Q, K, V, mask=None):
        """
        前向传播

        参数:
            Q, K, V: (batch, seq_len, d_model)

        返回:
            output: (batch, seq_len, d_model)
            weights: (batch, n_heads, seq, seq)
        """
        # Step 1: 线性变换得到 Q', K', V'
        Q_proj = np.dot(Q, self.W_Q)  # (batch, seq, d_model)
        K_proj = np.dot(K, self.W_K)
        V_proj = np.dot(V, self.W_V)

        # Step 2: 拆分成 n_heads 个头
        # (batch, seq, d_model) -> (batch, n_heads, seq, d_k)
        Q_heads = self._split_heads(Q_proj)
        K_heads = self._split_heads(K_proj)
        V_heads = self._split_heads(V_proj)

        # Step 3: 每个头独立计算 attention
        head_outputs = []
        all_weights = []
        for i in range(self.n_heads):
            output_i, scores_i, weights_i = single_head_attention(
                Q_heads[:, i, :, :], K_heads[:, i, :, :], V_heads[:, i, :, :],
                mask=mask
            )
            head_outputs.append(output_i)
            all_weights.append(weights_i)

        # Step 4: Concat + Linear
        # (batch, n_heads, seq, d_k) -> (batch, seq, d_model)
        head_outputs = np.stack(head_outputs, axis=1)  # (batch, heads, seq, d_k)
        concatenated = self._merge_heads(head_outputs)  # (batch, seq, d_model)
        output = np.dot(concatenated, self.W_O)

        self.head_weights = np.stack(all_weights, axis=1)  # (batch, heads, seq, seq)

        return output, self.head_weights

    def _split_heads(self, x):
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        return x.reshape(batch, seq, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch, seq, self.d_model)


# ═══════════════════════════════════════════════════════════════
# 4. 构造有结构的输入（聚类 + 位置编码）
# ═══════════════════════════════════════════════════════════════
def positional_encoding(seq_len, d_model):
    """标准 sin/cos 位置编码，返回 (seq_len, d_model)"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(0, d_model, 2)[None, :]  # 偶数维
    div = np.power(10000.0, i / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(pos / div)
    pe[:, 1::2] = np.cos(pos / div)
    return pe


def build_structured_input(seq_len, d_model, n_clusters, rng):
    """
    构造"聚类 + 位置"的输入：
      - n_clusters 个聚类中心，每个聚类分配 seq_len / n_clusters 个 token
      - 同一聚类的 token 在 embedding 空间相近（语义相似）
      - 加上位置编码，让每个 token 全局唯一

    关键设计：聚类中心放在彼此正交的坐标轴上（幅度较大），
    位置编码幅度较小（作为区分同聚类 token 的"微调"）。
    这样 Q·K 的内积才会出现真实差异，注意力才有"模式"可言。
    """
    n_tokens_per_cluster = seq_len // n_clusters

    # 聚类中心：放在彼此近似正交的方向上，幅度较大
    centers = np.zeros((n_clusters, d_model))
    rng.shuffle(centers)  # 触发 RNG 前进
    # 用 rng 生成 n_clusters 个近似正交向量
    centers = rng.randn(n_clusters, d_model)
    # 简单正交化（Gram-Schmidt）保证聚类间差异显著
    for i in range(1, n_clusters):
        for j in range(i):
            centers[i] -= (centers[i] @ centers[j]) / (centers[j] @ centers[j]) * centers[j]
    centers *= 2.0  # 放大聚类中心

    tokens = []
    cluster_of = []
    for c in range(n_clusters):
        for _ in range(n_tokens_per_cluster):
            tokens.append(centers[c] + rng.randn(d_model) * 0.05)
            cluster_of.append(c)
    tokens = np.array(tokens)  # (seq, d_model)

    # 加上小幅度的位置编码（让同一聚类的 token 仍能被区分）
    pe = positional_encoding(seq_len, d_model) * 0.3
    tokens = tokens + pe

    return tokens[np.newaxis, ...], cluster_of  # (1, seq, d_model)


# ═══════════════════════════════════════════════════════════════
# 5. 绘制多头注意力热力图
# ═══════════════════════════════════════════════════════════════
def plot_multihead_heatmaps(weights, n_heads, labels, fig_name="fig05_multihead_heatmap.png"):
    """
    绘制多个头的注意力权重热力图
    labels: 长度 seq 的列表，每个 token 的标签
    """
    fig, axes = plt.subplots(2, (n_heads + 1) // 2, figsize=(4.5 * n_heads, 9))
    axes = axes.flatten() if n_heads > 1 else [axes]

    for i in range(n_heads):
        ax = axes[i]
        w = weights[0, i]  # (seq, seq)

        im = ax.imshow(w, cmap="YlOrRd", aspect="equal")
        fig.colorbar(im, ax=ax, label="权重", fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Key (被关注)", fontsize=9)
        ax.set_ylabel("Query (主动关注)", fontsize=9)
        ax.set_title(f"Head {i+1}", fontsize=11, fontweight="bold")

        # 在每个格子标出权重数值，方便看集中度
        for row in range(w.shape[0]):
            for col in range(w.shape[1]):
                val = w[row, col]
                if val > 0.05:
                    color = "white" if val > 0.6 else "black"
                    ax.text(col, row, f"{val:.2f}", ha="center", va="center",
                            fontsize=7, color=color)

    # 关闭多余的 subplot
    for i in range(n_heads, len(axes)):
        axes[i].axis("off")

    fig.suptitle(f"多头注意力权重热力图 (n_heads={n_heads})\n"
                 f"不同头在不同子空间，关注不同的模式",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 6. PyTorch 参考实现（用于对比验证）
# ═══════════════════════════════════════════════════════════════
def pytorch_reference(Q, K, V, W_Q, W_K, W_V, W_O, n_heads, d_model):
    """
    与手写实现完全对应的 PyTorch 参考实现。
    刻意不使用 torch.nn.MultiheadAttention（它的 in_proj_weight 约定
    与"逐头拼接"的等价形式不同，容易引入无关差异）。
    """
    import torch
    batch, seq, _ = Q.shape
    d_k = d_model // n_heads

    Q_proj = torch.matmul(Q, W_Q)
    K_proj = torch.matmul(K, W_K)
    V_proj = torch.matmul(V, W_V)

    # split heads: (batch, seq, d_model) -> (batch, heads, seq, d_k)
    Q_heads = Q_proj.view(batch, seq, n_heads, d_k).permute(0, 2, 1, 3)
    K_heads = K_proj.view(batch, seq, n_heads, d_k).permute(0, 2, 1, 3)
    V_heads = V_proj.view(batch, seq, n_heads, d_k).permute(0, 2, 1, 3)

    scores = torch.matmul(Q_heads, K_heads.transpose(-1, -2)) / (d_k ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    head_out = torch.matmul(weights, V_heads)

    # merge: (batch, heads, seq, d_k) -> (batch, seq, d_model)
    merged = head_out.permute(0, 2, 1, 3).reshape(batch, seq, d_model)
    output = torch.matmul(merged, W_O)

    return output, weights


# ═══════════════════════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("手写多头注意力 (Multi-Head Attention)")
    print("=" * 70)

    # ── 超参数 ──
    D_MODEL = 64     # 模型维度
    N_HEADS = 4      # 头数
    SEQ_LEN = 8      # 序列长度
    N_CLUSTERS = 4   # 输入 token 的聚类数
    BATCH_SIZE = 1

    print(f"\n超参数:")
    print(f"  d_model = {D_MODEL}")
    print(f"  n_heads = {N_HEADS}")
    print(f"  d_k (每头维度) = {D_MODEL // N_HEADS}")
    print(f"  seq_len = {SEQ_LEN}, 聚类数 = {N_CLUSTERS}")

    # ── 构造有结构的输入 ──
    rng = np.random.RandomState(0)
    X, cluster_of = build_structured_input(SEQ_LEN, D_MODEL, N_CLUSTERS, rng)
    Q = X.copy()
    K = X.copy()
    V = X.copy()

    # token 标签：聚类 A/B/C/D + 位置
    cluster_chars = ["A", "B", "C", "D"]
    labels = [f"{cluster_chars[c]}{i}" for i, c in enumerate(cluster_of)]
    print(f"\n输入 token 标签: {labels}")
    print(f"  (同一聚类内 token 语义相近，不同聚类语义相远)")

    # ── 多头注意力 ──
    mha = MultiHeadAttention(D_MODEL, N_HEADS, base_seed=42)
    output, weights = mha.forward(Q, K, V)

    print(f"\n输出形状: {output.shape}  (batch, seq, d_model)")
    print(f"权重形状: {weights.shape}  (batch, heads, seq, seq)")

    # ── 分析每个头的注意力模式 ──
    print(f"\n{'─' * 70}")
    print("各头注意力模式分析")
    print(f"{'─' * 70}")

    for i in range(N_HEADS):
        w = weights[0, i]
        # 熵：越接近 log(seq) 越均匀，越接近 0 越集中
        entropy = -np.sum(w * np.log2(w + 1e-12), axis=1).mean()
        max_entropy = np.log2(SEQ_LEN)
        # 每个 query 行最关注的 key
        argmax_keys = w.argmax(axis=1)
        # 对角线权重均值（自关注倾向）
        diag_mean = np.diag(w).mean()
        # 平均权重与均匀分布的距离
        uniform = 1.0 / SEQ_LEN
        deviation = np.abs(w - uniform).mean()

        print(f"  Head {i+1}:")
        print(f"    平均熵 = {entropy:.3f} (最大值 {max_entropy:.3f}, "
              f"均匀度 = {entropy / max_entropy:.2%})")
        print(f"    对角线权重均值 = {diag_mean:.3f}  (自关注倾向)")
        print(f"    偏离均匀程度 = {deviation:.4f}")
        print(f"    每行最关注的 key: {[labels[i] for i in argmax_keys]}")

    # ── 验证每个头的权重行和为 1 ──
    print(f"\n{'─' * 70}")
    print("验证：每个头的权重行和是否为 1")
    print(f"{'─' * 70}")
    all_valid = True
    for i in range(N_HEADS):
        w = weights[0, i]
        row_sums = np.sum(w, axis=1)
        if np.max(np.abs(row_sums - 1.0)) > 1e-6:
            all_valid = False
            print(f"  ❌ Head {i+1}: 行和 = {row_sums}")
        else:
            print(f"  ✅ Head {i+1}: 行和 = 1.0 (误差 < 1e-6)")
    print(f"\n  {'✅ 所有头的权重行和为 1！' if all_valid else '❌ 存在不合法权重'}")

    # ── 可视化 ──
    print(f"\n{'─' * 70}")
    print("绘制多头注意力热力图")
    print(f"{'─' * 70}")
    plot_multihead_heatmaps(weights, N_HEADS, labels)

    # ── 与 PyTorch 对比 ──
    print(f"\n{'─' * 70}")
    print("验证：与 PyTorch 参考实现对比")
    print(f"{'─' * 70}")

    import torch
    Q_t = torch.tensor(Q, dtype=torch.float32)
    K_t = torch.tensor(K, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32)
    WQ_t = torch.tensor(mha.W_Q, dtype=torch.float32)
    WK_t = torch.tensor(mha.W_K, dtype=torch.float32)
    WV_t = torch.tensor(mha.W_V, dtype=torch.float32)
    WO_t = torch.tensor(mha.W_O, dtype=torch.float32)

    with torch.no_grad():
        torch_output, torch_weights = pytorch_reference(
            Q_t, K_t, V_t, WQ_t, WK_t, WV_t, WO_t, N_HEADS, D_MODEL
        )

    output_diff = np.abs(output - torch_output.numpy()).max()
    weight_diff = np.abs(weights - torch_weights.numpy()).max()
    print(f"\n  输出最大差异:    {output_diff:.2e}")
    print(f"  注意力权重差异: {weight_diff:.2e}")

    if output_diff < 1e-4:
        print(f"  ✅ 手写 NumPy 实现与 PyTorch 参考实现一致！")
        print(f"     (差异仅来自 float32 与 float64 的浮点精度)")
    else:
        print(f"  ⚠️  存在较大差异，请检查权重初始化")

    # ── 总结 ──
    print(f"\n{'=' * 70}")
    print("多头注意力完成！")
    print("=" * 70)
    print(f"  d_model = {D_MODEL}, n_heads = {N_HEADS}, d_k = {D_MODEL // N_HEADS}")
    print(f"  每个头: 独立的 Q, K, V 投影 → 独立的注意力模式")
    print(f"  输出: Concat(head_1, ..., head_h) · W_O")
    print(f"\n  为什么不同头能学到不同模式？")
    print(f"    - 每个头有独立的 W_Q^i, W_K^i, W_V^i")
    print(f"    - 它们把输入投到 d_model / n_heads 维的不同子空间")
    print(f"    - 在不同子空间里，'相似'的含义不同")
    print(f"    - 训练时反向传播会强化各头分工（本演示用不同种子模拟这一点）")


if __name__ == "__main__":
    main()
