"""
实践 2：两层神经网络的梯度

网络结构：X → Linear → ReLU → Linear → Sigmoid
损失函数：二元交叉熵 (BCE)

完成：
  1. NumPy 前向 + 反向传播（链式法则手算梯度）
  2. PyTorch autograd 验证
  3. 对比 NumPy vs PyTorch 梯度（形状、数值、最大绝对误差）
  4. 单个样本验证 → 小 batch 验证
"""
import numpy as np
import torch
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
# 工具函数
# ═══════════════════════════════════════════════════════════════
def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1.0 / (1.0 + np.exp(-x))


def relu(x):
    """ReLU 激活函数"""
    return np.maximum(0, x)


def relu_grad(x):
    """ReLU 的导数：x>0 时为 1，否则为 0"""
    return (x > 0).astype(float)


def bce_loss(prob, y):
    """
    二元交叉熵损失 (mean reduction)
    L = -mean( y*log(p) + (1-y)*log(1-p) )
    为数值稳定，对 log 加一个极小值 eps
    """
    eps = 1e-12
    prob = np.clip(prob, eps, 1 - eps)
    return -np.mean(y * np.log(prob) + (1 - y) * np.log(1 - prob))


# ═══════════════════════════════════════════════════════════════
# 1. NumPy 前向传播
# ═══════════════════════════════════════════════════════════════
def forward_numpy(X, W1, b1, W2, b2, y_true):
    """
    前向传播：X → Linear1 → ReLU → Linear2 → Sigmoid → BCE

    形状追踪（单个样本，输入维度 d，隐藏维度 h=4）：
        X:    (1, d)    输入
        W1:   (d, h)    第一层权重
        b1:   (h,)      第一层偏置
        h:    (1, h)    h = X @ W1 + b1
        a:    (1, h)    a = ReLU(h)
        W2:   (h, 1)    第二层权重
        b2:   (1,)      第二层偏置
        logits: (1, 1)  logits = a @ W2 + b2
        prob: (1, 1)    prob = sigmoid(logits)
        loss: 标量      BCE loss
    """
    # ── Linear 1: h = X @ W1 + b1 ──
    h = X @ W1 + b1          # (1, d) @ (d, h) + (h,) → (1, h)
    print(f"  h  = X@W1+b1:      shape={h.shape}")

    # ── ReLU: a = max(0, h) ──
    a = relu(h)              # (1, h)
    print(f"  a  = ReLU(h):      shape={a.shape}")

    # ── Linear 2: logits = a @ W2 + b2 ──
    logits = a @ W2 + b2     # (1, h) @ (h, 1) + (1,) → (1, 1)
    print(f"  logits = a@W2+b2:  shape={logits.shape}")

    # ── Sigmoid: prob = σ(logits) ──
    prob = sigmoid(logits)   # (1, 1)
    print(f"  prob = σ(logits):  shape={prob.shape}")

    # ── BCE Loss ──
    loss = bce_loss(prob, y_true)  # 标量
    print(f"  loss = BCE(prob,y): 标量 = {loss:.6f}")

    cache = {"h": h, "a": a, "logits": logits, "prob": prob}
    return loss, cache


# ═══════════════════════════════════════════════════════════════
# 2. NumPy 反向传播
# ═══════════════════════════════════════════════════════════════
def backward_numpy(loss, cache, X, y_true, W2):
    """
    反向传播：从 loss 向前计算 W1, b1, W2, b2 的梯度

    链式法则推导：

    ① ∂L/∂prob:
       L = -mean(y*log(p) + (1-y)*log(1-p))
       ∂L/∂p = -(y/p - (1-y)/(1-p)) / N
             = -((y-p)/(p*(1-p))) / N

    ② ∂L/∂logits:
       p = σ(z), dp/dz = p*(1-p)
       ∂L/∂z = ∂L/∂p * p*(1-p) = -(y-p)/N
       ⭐ 这就是 Sigmoid+BCE 的著名简化！梯度 = -(y - pred) / N

    ③ ∂L/∂W2:
       logits = a @ W2 + b2
       ∂L/∂W2 = a.T @ ∂L/∂z   (h,1) @ (1,1) → (h,1)  注意转置

    ④ ∂L/∂b2:
       ∂L/∂b2 = ∂L/∂z  (1,)  广播求和

    ⑤ ∂L/∂a:
       logits = a @ W2 + b2
       ∂L/∂a = ∂L/∂z @ W2.T   (1,1) @ (1,h) → (1,h)

    ⑥ ∂L/∂h:
       a = ReLU(h)
       ∂L/∂h = ∂L/∂a * relu'(h)  逐元素相乘 (1,h)

    ⑦ ∂L/∂W1:
       h = X @ W1 + b1
       ∂L/∂W1 = X.T @ ∂L/∂h   (d,1) @ (1,h) → (d,h)

    ⑧ ∂L/∂b1:
       ∂L/∂b1 = ∂L/∂h  (h,)  广播求和
    """
    h, a, logits, prob = cache["h"], cache["a"], cache["logits"], cache["prob"]
    N = X.shape[0]  # batch size

    # ── ① ∂L/∂prob ──
    eps = 1e-12
    prob_safe = np.clip(prob, eps, 1 - eps)
    dL_dprob = -(y_true / prob_safe - (1 - y_true) / (1 - prob_safe)) / N

    # ── ② ∂L/∂logits（Sigmoid 简化） ──
    dL_dlogits = -(y_true - prob) / N
    print(f"\n  ② ∂L/∂logits = -(y-p)/N:  shape={dL_dlogits.shape}")

    # ── ③ ∂L/∂W2 ──
    dL_dW2 = a.T @ dL_dlogits
    print(f"  ③ ∂L/∂W2 = a.T@dL/dz:      shape={dL_dW2.shape}")

    # ── ④ ∂L/∂b2 ──
    dL_db2 = np.sum(dL_dlogits, axis=0)
    print(f"  ④ ∂L/∂b2 = sum(dL/dz):     shape={dL_db2.shape}")

    # ── ⑤ ∂L/∂a ──
    dL_da = dL_dlogits @ W2.T
    print(f"  ⑤ ∂L/∂a = dL/dz@W2.T:      shape={dL_da.shape}")

    # ── ⑥ ∂L/∂h（ReLU 梯度） ──
    dL_dh = dL_da * relu_grad(h)
    print(f"  ⑥ ∂L/∂h = dL/da * relu'(h): shape={dL_dh.shape}")

    # ── ⑦ ∂L/∂W1 ──
    dL_dW1 = X.T @ dL_dh
    print(f"  ⑦ ∂L/∂W1 = X.T@dL/dh:      shape={dL_dW1.shape}")

    # ── ⑧ ∂L/∂b1 ──
    dL_db1 = np.sum(dL_dh, axis=0)
    print(f"  ⑧ ∂L/∂b1 = sum(dL/dh):     shape={dL_db1.shape}")

    return {
        "dW1": dL_dW1, "db1": dL_db1,
        "dW2": dL_dW2, "db2": dL_db2,
        "dL_dlogits": dL_dlogits,
    }


# ═══════════════════════════════════════════════════════════════
# 3. PyTorch autograd 实现
# ═══════════════════════════════════════════════════════════════
def forward_backward_torch(X_np, W1_np, b1_np, W2_np, b2_np, y_np):
    """用 PyTorch 完成前向+反向，返回梯度和 loss"""
    # 转为 PyTorch tensor，requires_grad=True
    X = torch.from_numpy(X_np.copy())
    y = torch.from_numpy(y_np.copy())

    W1 = torch.from_numpy(W1_np.copy()).requires_grad_(True)
    b1 = torch.from_numpy(b1_np.copy()).requires_grad_(True)
    W2 = torch.from_numpy(W2_np.copy()).requires_grad_(True)
    b2 = torch.from_numpy(b2_np.copy()).requires_grad_(True)

    # ── 前向传播 ──
    # 注意：与 NumPy 保持相同约定 W1=(d,h)，所以用 matmul 而非 linear
    h = torch.matmul(X, W1) + b1          # X @ W1 + b1
    a = torch.nn.functional.relu(h)
    logits = torch.matmul(a, W2) + b2     # a @ W2 + b2
    prob = torch.sigmoid(logits)

    # ── BCE Loss ──
    criterion = torch.nn.BCELoss()
    loss = criterion(prob, y)

    # ── 反向传播 ──
    loss.backward()

    return {
        "loss": loss.item(),
        "dW1": W1.grad.numpy(),
        "db1": b1.grad.numpy(),
        "dW2": W2.grad.numpy(),
        "db2": b2.grad.numpy(),
    }


# ═══════════════════════════════════════════════════════════════
# 4. 梯度对比工具
# ═══════════════════════════════════════════════════════════════
def compare_gradients(numpy_grads, torch_grads, names, tolerance=1e-5):
    """逐项对比 NumPy 和 PyTorch 梯度"""
    print(f"\n{'─' * 65}")
    print(f"  梯度对比（阈值: {tolerance:.0e}）")
    print(f"{'─' * 65}")

    all_pass = True
    max_error_total = 0.0

    for name in names:
        ng = numpy_grads[name]
        tg = torch_grads[name]

        # 检查形状
        shape_match = ng.shape == tg.shape
        if not shape_match:
            print(f"  ❌ {name}: 形状不匹配! NumPy={ng.shape} vs PyTorch={tg.shape}")
            all_pass = False
            continue

        # 计算数值差异
        diff = np.abs(ng - tg)
        max_err = diff.max()
        mean_err = diff.mean()
        max_error_total = max(max_error_total, max_err)

        status = "✅" if max_err < tolerance else "⚠️"
        if max_err >= tolerance:
            all_pass = False

        print(f"  {status} {name:8s}: shape={ng.shape}, "
              f"NumPy范围=[{ng.min():+.6f}, {ng.max():+.6f}], "
              f"PyTorch范围=[{tg.min():+.6f}, {tg.max():+.6f}], "
              f"最大绝对误差={max_err:.2e}")

    print(f"\n  总最大绝对误差: {max_error_total:.2e}")
    print(f"  {'✅ 所有梯度一致！' if all_pass else '❌ 存在不一致！'}")
    print(f"{'─' * 65}")

    return all_pass, max_error_total


# ═══════════════════════════════════════════════════════════════
# 5. 绘制计算图
# ═══════════════════════════════════════════════════════════════
def draw_nn_computation_graph(X_shape, W1_shape, b1_shape, W2_shape, b2_shape,
                              hidden_dim, input_dim,
                              fig_name="fig03_计算图_神经网络.png"):
    """绘制两层神经网络计算图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # 节点定义
    nodes = [
        ("X", f"X\n{X_shape}", 0.05, 0.50, "#E3F2FD", "输入"),
        ("W1", f"W1\n{W1_shape}", 0.05, 0.15, "#FFF3E0", "权重"),
        ("b1", f"b1\n{b1_shape}", 0.05, 0.85, "#FFF3E0", "偏置"),
        ("h", f"h = X@W1+b1\n{hidden_dim}", 0.25, 0.50, "#F3E5F5", "线性层1"),
        ("a", f"a = ReLU(h)\n{hidden_dim}", 0.45, 0.50, "#E8F5E9", "激活"),
        ("W2", f"W2\n{W2_shape}", 0.60, 0.15, "#FFF3E0", "权重"),
        ("b2", f"b2\n{b2_shape}", 0.60, 0.85, "#FFF3E0", "偏置"),
        ("z", f"logits = a@W2+b2\n(1,1)", 0.75, 0.50, "#F3E5F5", "线性层2"),
        ("p", f"p = σ(z)", 0.85, 0.50, "#FFF8E1", "Sigmoid"),
        ("L", f"L = BCE(p, y)\n标量", 0.95, 0.50, "#FCE4EC", "损失"),
    ]

    node_pos = {}
    for label, text, nx, ny, color, _ in nodes:
        ax.scatter(nx, ny, s=800, color=color, edgecolors="#333", zorder=5, linewidth=1.5)
        ax.text(nx, ny, text, ha="center", va="center", fontsize=7, fontweight="bold", zorder=6)
        node_pos[label] = (nx, ny)

    # 前向边（实线）
    edges_fwd = [
        ("X", "h", "X@W1"),
        ("W1", "h", ""),
        ("b1", "h", ""),
        ("h", "a", "ReLU"),
        ("a", "z", "a@W2"),
        ("W2", "z", ""),
        ("b2", "z", ""),
        ("z", "p", "σ"),
        ("p", "L", "BCE"),
    ]
    for src, dst, label in edges_fwd:
        sx, sy = node_pos[src]
        dx, dy = node_pos[dst]
        ax.annotate("", xy=(dx - 0.02, dy), xytext=(sx + 0.02, sy),
                    arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1.5,
                                    connectionstyle="arc3,rad=0"))
        if label:
            mx, my = (sx + dx) / 2, (sy + dy) / 2 + 0.04
            ax.text(mx, my, label, ha="center", va="center", fontsize=7,
                    color="#1565C0", fontweight="bold")

    # 反向边（虚线，标注梯度）
    edges_bwd = [
        ("L", "p", "∂L/∂p"),
        ("p", "z", "∂L/∂z=-(y-p)/N"),
        ("z", "a", "∂L/∂a=∂L/∂z·W2ᵀ"),
        ("a", "h", "∂L/∂h=∂L/∂a·relu'(h)"),
        ("h", "W1", "∂L/∂W1=Xᵀ·∂L/∂h"),
        ("h", "b1", "∂L/∂b1=Σ∂L/∂h"),
        ("z", "W2", "∂L/∂W2=aᵀ·∂L/∂z"),
        ("z", "b2", "∂L/∂b2=Σ∂L/∂z"),
    ]
    for src, dst, label in edges_bwd:
        sx, sy = node_pos[src]
        dx, dy = node_pos[dst]
        ax.annotate("", xy=(dx + 0.02, dy), xytext=(sx - 0.02, sy),
                    arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.2,
                                    linestyle="--", connectionstyle="arc3,rad=-0.2"))
        mx, my = (sx + dx) / 2, (sy + dy) / 2 - 0.06
        ax.text(mx, my, label, ha="center", va="center", fontsize=6,
                color="#C62828", bbox=dict(boxstyle="round,pad=0.15",
                                            facecolor="#FFEBEE", edgecolor="#EF9A9A", lw=0.5))

    # 图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#1565C0", lw=1.5, label="前向传播 (Forward)"),
        Line2D([0], [0], color="#C62828", lw=1.2, ls="--", label="反向传播 (Backward)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    ax.set_title("实践 2：两层神经网络计算图\n实线=前向传播，虚线=反向传播（链式法则）",
                 fontsize=13, pad=10)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 1)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 6. 绘制梯度对比图
# ═══════════════════════════════════════════════════════════════
def draw_gradient_bar_comparison(numpy_grads, torch_grads, names,
                                  shapes, max_errors,
                                  fig_name="fig04_梯度对比_神经网络.png"):
    """绘制梯度对比柱状图"""
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5),
                             sharey=False)
    if len(names) == 1:
        axes = [axes]

    for i, name in enumerate(names):
        ax = axes[i]
        ng = numpy_grads[name].flatten()
        tg = torch_grads[name].flatten()

        x_pos = np.arange(len(ng))
        width = 0.35

        bars1 = ax.bar(x_pos - width/2, ng, width, label="NumPy 手算",
                       color="#42A5F5", edgecolor="black", linewidth=0.5)
        bars2 = ax.bar(x_pos + width/2, tg, width, label="PyTorch",
                       color="#66BB6A", edgecolor="black", linewidth=0.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{j}" for j in range(len(ng))], fontsize=7)
        ax.set_xlabel("元素索引", fontsize=9)
        ax.set_ylabel("梯度值", fontsize=9)
        ax.set_title(f"{name}\nshape={shapes[i]}\nmax_err={max_errors[i]:.2e}", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

        # 标注最大值
        max_idx = np.argmax(np.abs(ng))
        ax.annotate(f"max: {ng[max_idx]:.4f}", xy=(max_idx - width/2, ng[max_idx]),
                    xytext=(max_idx + 0.3, ng[max_idx]),
                    fontsize=7, color="#C62828",
                    arrowprops=dict(arrowstyle="->", color="#C62828", lw=0.8))

    fig.suptitle("实践 2：NumPy 手算梯度 vs PyTorch autograd 逐项对比",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 70)
    print("实践 2：两层神经网络梯度")
    print("网络: X → Linear → ReLU → Linear → Sigmoid")
    print("损失: BCE (二元交叉熵)")
    print("=" * 70)

    # ── 超参数 ──
    INPUT_DIM = 2    # 输入维度（make_moons 有 2 个特征）
    HIDDEN_DIM = 4   # 隐藏层神经元数（故意设小，方便观察）
    BATCH_SIZE = 4   # 小 batch

    # ── 初始化参数 ──
    scale = 0.5
    W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM) * scale   # (d, h)
    b1 = np.zeros((HIDDEN_DIM,))                            # (h,)
    W2 = np.random.randn(HIDDEN_DIM, 1) * scale            # (h, 1)
    b2 = np.zeros((1,))                                     # (1,)

    # ── 测试 1：单个样本 ──
    print(f"\n{'=' * 70}")
    print("【测试 1】单个样本 (batch_size=1)")
    print(f"{'=' * 70}")

    X_single = np.array([[0.5, -0.3]])   # (1, 2)
    y_single = np.array([[1.0]])          # (1, 1)

    print(f"\n输入形状: X={X_single.shape}, y={y_single.shape}")
    print(f"参数形状: W1={W1.shape}, b1={b1.shape}, W2={W2.shape}, b2={b2.shape}")

    # NumPy
    print(f"\n【NumPy 前向传播】")
    loss_np, cache = forward_numpy(X_single, W1, b1, W2, b2, y_single)

    print(f"\n【NumPy 反向传播】")
    grads_np = backward_numpy(loss_np, cache, X_single, y_single, W2)

    # PyTorch
    print(f"\n【PyTorch autograd】")
    result_torch = forward_backward_torch(X_single, W1, b1, W2, b2, y_single)

    print(f"  loss = {result_torch['loss']:.6f} (NumPy loss = {loss_np:.6f})")

    # 对比
    names = ["dW1", "db1", "dW2", "db2"]
    numpy_dict = {n: grads_np[n] for n in names}
    torch_dict = {n: result_torch[n] for n in names}
    shapes = [str(grads_np[n].shape) for n in names]

    all_pass, max_err = compare_gradients(numpy_dict, torch_dict, names)

    # 绘制
    draw_gradient_bar_comparison(
        numpy_dict, torch_dict, names,
        shapes,
        [np.abs(numpy_dict[n] - torch_dict[n]).max() for n in names]
    )

    # ── 测试 2：小 batch ──
    print(f"\n\n{'=' * 70}")
    print(f"【测试 2】小 batch (batch_size={BATCH_SIZE})")
    print(f"{'=' * 70}")

    # 从 make_moons 取一个 batch
    from sklearn.datasets import make_moons
    X_all, y_all = make_moons(n_samples=20, noise=0.1, random_state=42)
    X_batch = X_all[:BATCH_SIZE].astype(np.float64)
    y_batch = y_all[:BATCH_SIZE].reshape(-1, 1).astype(np.float64)

    print(f"\n输入形状: X={X_batch.shape}, y={y_batch.shape}")

    # NumPy
    print(f"\n【NumPy 前向传播 (batch)】")
    loss_np_b, cache_b = forward_numpy(X_batch, W1, b1, W2, b2, y_batch)

    print(f"\n【NumPy 反向传播 (batch)】")
    grads_np_b = backward_numpy(loss_np_b, cache_b, X_batch, y_batch, W2)

    # PyTorch
    print(f"\n【PyTorch autograd (batch)】")
    result_torch_b = forward_backward_torch(X_batch, W1, b1, W2, b2, y_batch)

    print(f"  loss = {result_torch_b['loss']:.6f} (NumPy loss = {loss_np_b:.6f})")

    # 对比
    numpy_dict_b = {n: grads_np_b[n] for n in names}
    torch_dict_b = {n: result_torch_b[n] for n in names}
    shapes_b = [str(grads_np_b[n].shape) for n in names]

    all_pass_b, max_err_b = compare_gradients(numpy_dict_b, torch_dict_b, names)

    # ── 绘制计算图 ──
    print(f"\n{'=' * 70}")
    print("绘制计算图")
    print("=" * 70)
    draw_nn_computation_graph(
        X_shape=f"(1,{INPUT_DIM})",
        W1_shape=f"({INPUT_DIM},{HIDDEN_DIM})",
        b1_shape=f"({HIDDEN_DIM},)",
        W2_shape=f"({HIDDEN_DIM},1)",
        b2_shape="(1,)",
        hidden_dim=f"({HIDDEN_DIM},)",
        input_dim=INPUT_DIM
    )

    # ── 总结 ──
    print(f"\n{'=' * 70}")
    print("实践 2 完成！")
    print("=" * 70)
    print(f"  测试 1 (单样本): max_err={max_err:.2e} {'✅' if all_pass else '❌'}")
    print(f"  测试 2 (batch={BATCH_SIZE}): max_err={max_err_b:.2e} {'✅' if all_pass_b else '❌'}")

    # 形状对比表
    print(f"\n{'─' * 65}")
    print("  形状对比：单样本 vs 小 batch")
    print(f"{'─' * 65}")
    print(f"  {'参数':8s} {'单样本 shape':16s} {'batch shape':16s}")
    print(f"  {'─'*42}")
    for n in names:
        print(f"  {n:8s} {str(grads_np[n].shape):16s} {str(grads_np_b[n].shape):16s}")
    print(f"\n  ⭐ 关键观察：")
    print(f"    - W1, W2, b1, b2 的形状在单样本和 batch 中完全相同")
    print(f"    - 只有中间结果的 batch 维度从 1 变为 {BATCH_SIZE}")
    print(f"    - BCE loss 的 reduction='mean' 意味着 batch 越大梯度越平滑")


if __name__ == "__main__":
    main()
