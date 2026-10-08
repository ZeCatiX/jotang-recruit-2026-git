"""
实践 1：简单计算链 f(x, y) = (xy + x²)²

任务：
  1. 把计算过程拆成步骤，画出计算图
  2. 用 NumPy 按计算图从后向前手算梯度
  3. 用 PyTorch autograd 验证结果一致
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
# 1. 前向传播：f(x, y) = (xy + x²)²
# ═══════════════════════════════════════════════════════════════
def forward_numpy(x, y):
    """
    把 f(x,y) = (xy + x²)² 拆成 4 个简单步骤：
        step1: a = x * y          # 乘法
        step2: b = x ** 2         # 平方
        step3: c = a + b          # 加法
        step4: f = c ** 2         # 平方
    返回中间值和最终结果。
    """
    a = x * y       # step1: 乘法
    b = x ** 2      # step2: 平方
    c = a + b       # step3: 加法
    f = c ** 2      # step4: 平方
    return {"x": x, "y": y, "a": a, "b": b, "c": c, "f": f}


# ═══════════════════════════════════════════════════════════════
# 2. 反向传播：按计算图从后向前计算梯度
# ═══════════════════════════════════════════════════════════════
def backward_numpy(vals):
    """
    根据前向保存的中间值，用链式法则从 f 向前计算 ∂f/∂x 和 ∂f/∂y。

    计算图（前向→反向）：
        f = c²        →  ∂f/∂c = 2c
        c = a + b     →  ∂c/∂a = 1,  ∂c/∂b = 1
        a = x * y     →  ∂a/∂x = y,  ∂a/∂y = x
        b = x²        →  ∂b/∂x = 2x

    链式法则：
        ∂f/∂x = ∂f/∂c · (∂c/∂a·∂a/∂x + ∂c/∂b·∂b/∂x)
              = 2c · (1·y + 1·2x)
              = 2c · (y + 2x)

        ∂f/∂y = ∂f/∂c · ∂c/∂a · ∂a/∂y
              = 2c · 1 · x
              = 2cx
    """
    x, y, a, b, c, f = vals["x"], vals["y"], vals["a"], vals["b"], vals["c"], vals["f"]

    # 从 f 开始：df/dc = 2c
    df_dc = 2 * c

    # c = a + b → dc/da = 1, dc/db = 1
    df_da = df_dc * 1
    df_db = df_dc * 1

    # a = xy → da/dx = y, da/dy = x
    df_dax = df_da * y   # ∂f/∂x 来自 a 的部分
    df_day = df_da * x   # ∂f/∂y

    # b = x² → db/dx = 2x
    df_dbx = df_db * 2 * x  # ∂f/∂x 来自 b 的部分

    # 汇总 ∂f/∂x = 来自 a 的部分 + 来自 b 的部分
    df_dx = df_dax + df_dbx
    df_dy = df_day

    return {"df_dx": df_dx, "df_dy": df_dy}


# ═══════════════════════════════════════════════════════════════
# 3. PyTorch autograd 验证
# ═══════════════════════════════════════════════════════════════
def forward_backward_torch(x_val, y_val):
    """用 PyTorch autograd 计算 f(x,y)=(xy+x²)² 的梯度"""
    x = torch.tensor(float(x_val), requires_grad=True)
    y = torch.tensor(float(y_val), requires_grad=True)

    # 前向
    a = x * y
    b = x ** 2
    c = a + b
    f = c ** 2

    # 反向
    f.backward()

    return f.item(), x.grad.item(), y.grad.item()


# ═══════════════════════════════════════════════════════════════
# 4. 数值梯度验证（有限差分法）
# ═══════════════════════════════════════════════════════════════
def numerical_gradient(x, y, eps=1e-5):
    """用中心差分法计算数值梯度作为第三方验证"""
    f_xy = (x * y + x ** 2) ** 2

    f_xp = ((x + eps) * y + (x + eps) ** 2) ** 2
    f_xm = ((x - eps) * y + (x - eps) ** 2) ** 2

    f_yp = (x * (y + eps) + x ** 2) ** 2
    f_ym = (x * (y - eps) + x ** 2) ** 2

    dxd = (f_xp - f_xm) / (2 * eps)
    dyd = (f_yp - f_ym) / (2 * eps)

    return dxd, dyd


# ═══════════════════════════════════════════════════════════════
# 5. 绘制计算图
# ═══════════════════════════════════════════════════════════════
def draw_computation_graph(vals, fig_name="fig01_计算图_简单函数.png"):
    """画出 f(x,y)=(xy+x²)² 的计算图"""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    x, y, a, b, c, f = vals["x"], vals["y"], vals["a"], vals["b"], vals["c"], vals["f"]

    # 节点定义：(标签, 值, x坐标, y坐标, 颜色)
    nodes = [
        ("x", f"x = {x:.2f}", 0.15, 0.75, "#E3F2FD"),
        ("y", f"y = {y:.2f}", 0.15, 0.25, "#E3F2FD"),
        ("a", f"a = x·y = {a:.2f}", 0.40, 0.55, "#FFF3E0"),
        ("b", f"b = x² = {b:.2f}", 0.40, 0.25, "#FFF3E0"),
        ("c", f"c = a+b = {c:.2f}", 0.65, 0.40, "#E8F5E9"),
        ("f", f"f = c² = {f:.2f}", 0.90, 0.40, "#FCE4EC"),
    ]

    # 连线
    edges = [
        (0, 2, f"∂a/∂x = y = {y:.2f}"),
        (1, 2, f"∂a/∂y = x = {x:.2f}"),
        (0, 3, f"∂b/∂x = 2x = {2*x:.2f}"),
        (2, 4, "∂c/∂a = 1"),
        (3, 4, "∂c/∂b = 1"),
        (4, 5, f"∂f/∂c = 2c = {2*c:.2f}"),
    ]

    node_pos = {}
    for label, text, nx, ny, color in nodes:
        ax.scatter(nx, ny, s=1200, color=color, edgecolors="#333", zorder=5, linewidth=2)
        ax.text(nx, ny, text, ha="center", va="center", fontsize=10, fontweight="bold", zorder=6)
        node_pos[label] = (nx, ny)

    for src_idx, dst_idx, edge_label in edges:
        sx, sy = node_pos[nodes[src_idx][0]]
        dx, dy = node_pos[nodes[dst_idx][0]]
        ax.annotate("", xy=(dx - 0.05, dy), xytext=(sx + 0.05, sy),
                    arrowprops=dict(arrowstyle="->", color="#666", lw=1.5,
                                    connectionstyle="arc3,rad=0"))
        # 梯度标注
        mx, my = (sx + dx) / 2, (sy + dy) / 2 + 0.06
        ax.text(mx, my, edge_label, ha="center", va="center", fontsize=7,
                color="#B71C1C", bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFF8E1", edgecolor="#FFB74D", linewidth=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("实践 1：f(x, y) = (xy + x²)²  计算图\n箭头=前向传播，红色标注=反向传播的局部梯度",
                 fontsize=13, pad=15)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 6. 绘制梯度对比图
# ═══════════════════════════════════════════════════════════════
def draw_gradient_comparison(results, fig_name="fig02_梯度对比_简单函数.png"):
    """绘制 NumPy / PyTorch / 数值梯度 的对比"""
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5))
    if len(results) == 1:
        axes = [axes]

    colors = ["#42A5F5", "#66BB6A", "#FFA726"]
    labels = ["NumPy 手算", "PyTorch", "数值差分"]

    for i, res in enumerate(results):
        ax = axes[i]
        vals_x = [res["numpy"]["df_dx"], res["torch"]["df_dx"], res["numerical"]["df_dx"]]
        vals_y = [res["numpy"]["df_dy"], res["torch"]["df_dy"], res["numerical"]["df_dy"]]

        # 分组柱状图：2组（∂f/∂x, ∂f/∂y），每组3个柱（3种方法）
        x_pos = np.arange(2)  # 0=∂f/∂x, 1=∂f/∂y
        width = 0.25

        # 每组3个柱：遍历组(g=0,1)，每组内遍历方法(m=0,1,2)
        for g in range(2):
            vals_group = vals_x if g == 0 else vals_y
            for m in range(3):
                xpos = x_pos[g] - width + m * width
                val = vals_group[m]
                ax.bar(xpos, val, width, color=colors[m], edgecolor="black", linewidth=0.5)
                ax.text(xpos, val + (2 if val >= 0 else -2), f"{val:.1f}",
                        ha="center", va="bottom" if val >= 0 else "top", fontsize=7)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(["∂f/∂x", "∂f/∂y"], fontsize=11)
        ax.set_ylabel("梯度值", fontsize=11)
        ax.set_title(f"x={res['x']:.1f}, y={res['y']:.1f}\n最大误差={res['max_err']:.2e}", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0, color="black", linewidth=0.5)

    # 添加方法图例
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i], label=labels[i]) for i in range(3)]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("实践 1：三种方法计算 f(x,y)=(xy+x²)² 的梯度对比\nNumPy 手算 ≈ PyTorch autograd ≈ 数值差分 → 结果一致",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 7. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("实践 1：简单计算链  f(x, y) = (xy + x²)²")
    print("=" * 70)

    # 选择几组 (x, y) 进行测试
    test_cases = [(2.0, 3.0), (-1.5, 0.5), (0.7, 4.2), (3.0, -2.0)]

    all_results = []

    for x_val, y_val in test_cases:
        print(f"\n{'─' * 60}")
        print(f"  x = {x_val},  y = {y_val}")
        print(f"{'─' * 60}")

        # ── NumPy 前向 + 反向 ──
        vals = forward_numpy(x_val, y_val)
        grads = backward_numpy(vals)

        print(f"\n【NumPy 前向传播】")
        print(f"  a = x*y = {vals['a']:.6f}")
        print(f"  b = x²  = {vals['b']:.6f}")
        print(f"  c = a+b = {vals['c']:.6f}")
        print(f"  f = c²  = {vals['f']:.6f}")

        print(f"\n【NumPy 反向传播】")
        print(f"  ∂f/∂x = 2c(y+2x) = {grads['df_dx']:.6f}")
        print(f"  ∂f/∂y = 2cx      = {grads['df_dy']:.6f}")

        # ── PyTorch autograd ──
        f_torch, dx_torch, dy_torch = forward_backward_torch(x_val, y_val)

        print(f"\n【PyTorch autograd】")
        print(f"  f  = {f_torch:.6f}")
        print(f"  ∂f/∂x = {dx_torch:.6f}")
        print(f"  ∂f/∂y = {dy_torch:.6f}")

        # ── 数值差分验证 ──
        dx_num, dy_num = numerical_gradient(x_val, y_val)

        print(f"\n【数值差分 (中心差商)】")
        print(f"  ∂f/∂x ≈ {dx_num:.6f}")
        print(f"  ∂f/∂y ≈ {dy_num:.6f}")

        # ── 对比 ──
        err_x = abs(grads["df_dx"] - dx_torch)
        err_y = abs(grads["df_dy"] - dy_torch)
        max_err = max(err_x, err_y)

        print(f"\n【对比结果】")
        print(f"  ∂f/∂x: NumPy = {grads['df_dx']:.10f},  PyTorch = {dx_torch:.10f},  误差 = {err_x:.2e}")
        print(f"  ∂f/∂y: NumPy = {grads['df_dy']:.10f},  PyTorch = {dy_torch:.10f},  误差 = {err_y:.2e}")
        print(f"  最大绝对误差: {max_err:.2e}")
        if max_err < 1e-5:
            print("  ✅ 结果一致！")
        else:
            print("  ❌ 结果不一致！")

        all_results.append({
            "x": x_val, "y": y_val,
            "numpy": grads,
            "torch": {"df_dx": dx_torch, "df_dy": dy_torch},
            "numerical": {"df_dx": dx_num, "df_dy": dy_num},
            "max_err": max_err,
        })

    # 绘制计算图（用第一组数据）
    print(f"\n{'=' * 70}")
    print("绘制计算图")
    print("=" * 70)
    vals_demo = forward_numpy(*test_cases[0])
    draw_computation_graph(vals_demo)

    # 绘制梯度对比
    draw_gradient_comparison(all_results)

    # 总结
    print(f"\n{'=' * 70}")
    print("实践 1 完成！")
    print("=" * 70)
    for r in all_results:
        status = "✅" if r["max_err"] < 1e-5 else "❌"
        print(f"  x={r['x']:.1f}, y={r['y']:.1f}: 最大误差={r['max_err']:.2e} {status}")


if __name__ == "__main__":
    main()
