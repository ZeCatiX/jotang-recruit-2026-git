"""
手写单头 Scaled Dot-Product Attention

实现步骤：
  1. 随机生成 Q, K, V 矩阵
  2. 计算 Q @ K^T / sqrt(d_k) → Attention 分数
  3. softmax → Attention 权重
  4. 权重 @ V → 输出
  5. 绘制注意力权重热力图

参考论文公式：
  Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) · V
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
# 1. 手写 Scaled Dot-Product Attention
# ═══════════════════════════════════════════════════════════════
def scaled_dot_product_attention(Q, K, V):
    """
    单头 Scaled Dot-Product Attention

    参数:
        Q: (d_model,) Query 向量
        K: (d_model,) Key 向量
        V: (d_model,) Value 向量

    返回:
        output: (d_model,) 注意力输出
        scores: 注意力分数 (softmax 前)
        weights: 注意力权重 (softmax 后)
    """
    d_k = Q.shape[0]

    # Step 1: 计算点积 Q · K / sqrt(d_k)
    scores = np.dot(Q, K) / np.sqrt(d_k)
    print(f"  Q·K / sqrt(d_k) = {scores:.4f}  (d_k={d_k})")

    # Step 2: softmax
    weights = softmax(scores)
    print(f"  softmax 权重 = {weights:.4f}")

    # Step 3: 加权求和
    output = weights * V
    print(f"  输出 = 权重 · V: shape={output.shape}, 前5维={output[:5]}")

    return output, scores, weights


def scaled_dot_product_attention_matrix(Q, K, V):
    """
    矩阵形式的 Scaled Dot-Product Attention

    参数:
        Q: (seq_q, d_k) Query 矩阵
        K: (seq_k, d_k) Key 矩阵
        V: (seq_k, d_v) Value 矩阵

    返回:
        output: (seq_q, d_v) 输出矩阵
        scores: (seq_q, seq_k) 注意力分数
        weights: (seq_q, seq_k) 注意力权重
    """
    d_k = Q.shape[1]

    # Step 1: 计算 Q @ K^T / sqrt(d_k)
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # Step 2: softmax (沿 seq_k 方向)
    weights = softmax(scores, axis=-1)

    # Step 3: 加权求和 weights @ V
    output = np.dot(weights, V)

    return output, scores, weights


# ═══════════════════════════════════════════════════════════════
# 2. softmax 函数
# ═══════════════════════════════════════════════════════════════
def softmax(x, axis=-1):
    """
    数值稳定的 softmax
    减去最大值防止 exp 溢出
    """
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ═══════════════════════════════════════════════════════════════
# 3. 绘制热力图
# ═══════════════════════════════════════════════════════════════
def draw_heatmap(data, title, xlabel="Key (被关注的位置)", ylabel="Query (关注的位置)",
                 fig_name=None, annotate=False, cmap="YlOrRd"):
    """绘制热力图"""
    fig, ax = plt.subplots(figsize=(7, 6))

    if annotate:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, label="权重")
        # 标注数值
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                ax.text(j, i, f"{data[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color="black" if data[i, j] < data.max() * 0.6 else "white")
    else:
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        fig.colorbar(im, ax=ax, label="值")

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)

    if fig_name:
        fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 4. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("手写单头 Scaled Dot-Product Attention")
    print("=" * 70)

    np.random.seed(42)

    # ── 示例 1：标量形式（单个查询） ──
    print(f"\n{'─' * 70}")
    print("【示例 1】标量形式：单个 Query 向量")
    print(f"{'─' * 70}")

    d_model = 64
    Q = np.random.randn(d_model) * 0.5 + 0.3   # (64,)
    K = np.random.randn(d_model) * 0.5 + 0.3   # (64,)
    V = np.random.randn(d_model) * 0.5 + 0.3   # (64,)

    output, scores, weights = scaled_dot_product_attention(Q, K, V)

    print(f"\n  Q 形状: {Q.shape}, K 形状: {K.shape}, V 形状: {V.shape}")
    print(f"  输出形状: {output.shape}")
    print(f"  注意力权重: {weights:.4f} (注意: 标量形式 softmax 只有一个值)")

    # ── 示例 2：矩阵形式（多个查询） ──
    print(f"\n{'─' * 70}")
    print("【示例 2】矩阵形式：多个 Query 向量")
    print(f"{'─' * 70}")

    seq_q = 5    # 查询序列长度
    seq_k = 8    # 键/值序列长度
    d_k = 64     # 键维度

    Q = np.random.randn(seq_q, d_k) * 0.5 + 0.3   # (5, 64)
    K = np.random.randn(seq_k, d_k) * 0.5 + 0.3   # (8, 64)
    V = np.random.randn(seq_k, d_k) * 0.5 + 0.3   # (8, 64)

    print(f"\n  Q 形状: {Q.shape}  (查询序列: {seq_q} 个, 每个 {d_k} 维)")
    print(f"  K 形状: {K.shape}  (键序列: {seq_k} 个, 每个 {d_k} 维)")
    print(f"  V 形状: {V.shape}  (值序列: {seq_k} 个, 每个 {d_k} 维)")

    output, scores, weights = scaled_dot_product_attention_matrix(Q, K, V)

    print(f"\n  ── 中间结果 ──")
    print(f"  Q @ K^T 形状:        {seq_q} × {seq_k}  (每个 query 和每个 key 的点积)")
    print(f"  scores (softmax前) 形状: {scores.shape}")
    print(f"  weights (softmax后) 形状: {weights.shape}")
    print(f"  output 形状:          {output.shape}")

    print(f"\n  ── 注意力权重矩阵 (每个 query 对每个 key 的权重) ──")
    print(f"  softmax 后每行和为 1:")
    for i in range(seq_q):
        row_sum = np.sum(weights[i])
        print(f"    Row {i}: 和 = {row_sum:.4f} {'✅' if abs(row_sum - 1) < 1e-10 else '❌'}")

    # ── 可视化 ──
    print(f"\n{'─' * 70}")
    print("绘制注意力热力图")
    print(f"{'─' * 70}")

    # 使用更大的序列以便可视化
    np.random.seed(0)
    seq_len = 8
    d_k_large = 64
    Q = np.random.randn(seq_len, d_k_large)
    K = np.random.randn(seq_len, d_k_large)
    V = np.random.randn(seq_len, d_k_large)

    output, scores, weights = scaled_dot_product_attention_matrix(Q, K, V)

    print(f"\n  序列长度: {seq_len}, 维度: {d_k_large}")
    print(f"  分数范围: [{scores.min():.2f}, {scores.max():.2f}]")
    print(f"  权重范围: [{weights.min():.4f}, {weights.max():.4f}]")

    # 绘制两个子图：分数和权重
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左：Attention 分数
    im1 = axes[0].imshow(scores, cmap="YlOrRd", aspect="equal")
    fig.colorbar(im1, ax=axes[0], label="分数")
    axes[0].set_xlabel("Key (被关注位置)", fontsize=10)
    axes[0].set_ylabel("Query (关注位置)", fontsize=10)
    axes[0].set_title("Attention 分数 (Q·K^T / √d_k)\nsoftmax 之前", fontsize=11)

    # 右：Attention 权重
    im2 = axes[1].imshow(weights, cmap="YlOrRd", aspect="equal")
    fig.colorbar(im2, ax=axes[1], label="权重")
    axes[1].set_xlabel("Key (被关注位置)", fontsize=10)
    axes[1].set_ylabel("Query (关注位置)", fontsize=10)
    axes[1].set_title("Attention 权重 (softmax 后)\n每行和为 1", fontsize=11)

    fig.suptitle("单头 Scaled Dot-Product Attention\n随机 Q, K, V 输入",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_attention_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / 'fig04_attention_heatmap.png'}")

    # ── 验证 PyTorch ──
    print(f"\n{'─' * 70}")
    print("验证：与 PyTorch 对比")
    print(f"{'─' * 70}")

    import torch
    torch.manual_seed(0)

    Q_t = torch.tensor(Q, dtype=torch.float32)
    K_t = torch.tensor(K, dtype=torch.float32)
    V_t = torch.tensor(V, dtype=torch.float32)

    d_k_t = Q_t.shape[-1]
    scores_t = torch.matmul(Q_t, K_t.T) / torch.sqrt(torch.tensor(float(d_k_t)))
    weights_t = torch.softmax(scores_t, dim=-1)
    output_t = torch.matmul(weights_t, V_t)

    # 对比
    score_diff = np.abs(scores - scores_t.numpy()).max()
    weight_diff = np.abs(weights - weights_t.numpy()).max()
    output_diff = np.abs(output - output_t.numpy()).max()

    print(f"\n  分数最大差异:   {score_diff:.2e}")
    print(f"  权重最大差异:   {weight_diff:.2e}")
    print(f"  输出最大差异:   {output_diff:.2e}")

    if max(score_diff, weight_diff, output_diff) < 1e-6:
        print(f"  ✅ 手写实现与 PyTorch 一致！")
    else:
        print(f"  ❌ 存在差异！")

    # ── 总结 ──
    print(f"\n{'=' * 70}")
    print("单头 Attention 完成！")
    print("=" * 70)
    print(f"  核心公式: Attention(Q,K,V) = softmax(QK^T / √d_k) · V")
    print(f"  分数形状: ({seq_q}, {seq_k})")
    print(f"  权重形状: ({seq_q}, {seq_k})  ← 每行和为 1")
    print(f"  输出形状: ({seq_q}, {d_k_large})")


if __name__ == "__main__":
    main()
