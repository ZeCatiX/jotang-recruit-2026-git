"""
重绘 Transformer 结构图
基于论文 Figure 1，用 Matplotlib 画出简化版和详细版结构图
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════
# 颜色定义
# ═══════════════════════════════════════════════════════════════
COLORS = {
    "enc": "#E3F2FD",       # 蓝色 - Encoder
    "dec": "#E8F5E9",       # 绿色 - Decoder
    "attn": "#FFF3E0",      # 橙色 - Attention
    "ffn": "#F3E5F5",       # 紫色 - Feed-Forward
    "norm": "#FFEBEE",      # 红色 - LayerNorm
    "res": "#F5F5F5",       # 灰色 - Residual
    "pe": "#FFF8E1",        # 黄色 - Positional Encoding
    "embed": "#E0F2F1",     # 青色 - Embedding
    "add": "#FFFDE7",       # 浅黄 - Add
}


def draw_box(ax, x, y, w, h, text, color, fontsize=8, alpha=0.8, bold=False):
    """画一个圆角矩形"""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.02",
                         facecolor=color, edgecolor="#333", linewidth=1.2,
                         alpha=alpha, zorder=2)
    ax.add_patch(box)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
            fontweight=weight, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, style="->", color="#333", lw=1.5, ls="-"):
    """画箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, color=color, linewidth=lw,
                           linestyle=ls, mutation_scale=12,
                           shrinkA=0, shrinkB=0, zorder=1)
    ax.add_patch(arrow)


def draw_skip(ax, x_start, y_start, x_end, y_end, offset=0.06):
    """画残差连接（跳过的U型线）"""
    mid_x = (x_start + x_end) / 2
    draw_arrow(ax, x_start, y_start, mid_x, y_start, color="#C62828", lw=1.2, ls="--")
    draw_arrow(ax, mid_x, y_start, mid_x, y_end + offset, color="#C62828", lw=1.2, ls="--")
    draw_arrow(ax, mid_x, y_end + offset, x_end, y_end, color="#C62828", lw=1.2, ls="--")


# ═══════════════════════════════════════════════════════════════
# 1. 简化版 Transformer 结构图
# ═══════════════════════════════════════════════════════════════
def draw_simple_structure():
    """简化版：只展示 Encoder 和 Decoder 的顶层结构"""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.axis("off")

    # Encoder 列
    enc_x = 0.3
    draw_box(ax, enc_x, 0.85, 0.20, 0.05, "Encoder Embedding", COLORS["embed"])
    draw_arrow(ax, enc_x, 0.82, enc_x, 0.78)
    draw_box(ax, enc_x, 0.75, 0.20, 0.05, "+ Positional Encoding", COLORS["pe"])

    draw_arrow(ax, enc_x, 0.72, enc_x, 0.68)

    # Encoder 重复部分
    for i, y_base in enumerate([0.55, 0.35]):
        draw_box(ax, enc_x, y_base + 0.08, 0.18, 0.04, "Multi-Head Self-Attention", COLORS["attn"], 7)
        draw_box(ax, enc_x, y_base + 0.02, 0.18, 0.04, "Add & Norm", COLORS["norm"], 7)
        draw_box(ax, enc_x, y_base - 0.04, 0.18, 0.04, "Feed-Forward Network", COLORS["ffn"], 7)
        draw_box(ax, enc_x, y_base - 0.10, 0.18, 0.04, "Add & Norm", COLORS["norm"], 7)

        # 残差连接
        draw_skip(ax, enc_x - 0.09, y_base + 0.10, enc_x - 0.09, y_base - 0.02, 0.03)
        draw_skip(ax, enc_x - 0.09, y_base - 0.06, enc_x - 0.09, y_base - 0.12, 0.03)

        if i == 0:
            draw_arrow(ax, enc_x, 0.62, enc_x, 0.62)  # 重复标记
            ax.text(enc_x + 0.12, 0.55, "× N", fontsize=14, color="#C62828", fontweight="bold")

    draw_arrow(ax, enc_x, 0.25, enc_x, 0.20)

    # Decoder 列
    dec_x = 0.70
    draw_box(ax, dec_x, 0.85, 0.20, 0.05, "Decoder Embedding", COLORS["embed"])
    draw_arrow(ax, dec_x, 0.82, dec_x, 0.78)
    draw_box(ax, dec_x, 0.75, 0.20, 0.05, "+ Positional Encoding", COLORS["pe"])

    draw_arrow(ax, dec_x, 0.72, dec_x, 0.68)

    for i, y_base in enumerate([0.55, 0.35]):
        # Masked Self-Attention
        draw_box(ax, dec_x, y_base + 0.08, 0.18, 0.04, "Masked Self-Attention", COLORS["attn"], 7)
        draw_box(ax, dec_x, y_base + 0.02, 0.18, 0.04, "Add & Norm", COLORS["norm"], 7)

        # Cross-Attention
        draw_box(ax, dec_x, y_base - 0.04, 0.18, 0.04, "Cross-Attention (K,V from Enc)", COLORS["attn"], 7)
        draw_box(ax, dec_x, y_base - 0.10, 0.18, 0.04, "Add & Norm", COLORS["norm"], 7)

        draw_box(ax, dec_x, y_base - 0.16, 0.18, 0.04, "Feed-Forward Network", COLORS["ffn"], 7)
        draw_box(ax, dec_x, y_base - 0.22, 0.18, 0.04, "Add & Norm", COLORS["norm"], 7)

        draw_skip(ax, dec_x - 0.09, y_base + 0.10, dec_x - 0.09, y_base - 0.02, 0.03)
        draw_skip(ax, dec_x - 0.09, y_base - 0.06, dec_x - 0.09, y_base - 0.12, 0.03)
        draw_skip(ax, dec_x - 0.09, y_base - 0.14, dec_x - 0.09, y_base - 0.24, 0.03)

        if i == 0:
            ax.text(dec_x + 0.12, 0.55, "× N", fontsize=14, color="#C62828", fontweight="bold")

    draw_arrow(ax, dec_x, 0.12, dec_x, 0.08)
    draw_box(ax, dec_x, 0.05, 0.18, 0.04, "Linear + Softmax", COLORS["embed"], 8)

    # Encoder → Decoder 的 Cross-Attention 连接
    draw_arrow(ax, enc_x + 0.10, 0.45, dec_x - 0.10, 0.45, color="#1565C0", lw=2)
    ax.text((enc_x + dec_x) / 2, 0.47, "Encoder Output → Cross-Attention",
            fontsize=7, color="#1565C0", ha="center")

    # 标题
    ax.set_title("Transformer 结构图（简化版）\n基于《Attention Is All You Need》Figure 1 重绘",
                 fontsize=14, fontweight="bold", pad=10)

    fig.savefig(FIG_DIR / "fig01_transformer_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / 'fig01_transformer_structure.png'}")


# ═══════════════════════════════════════════════════════════════
# 2. 详细版：展示 Attention 内部结构
# ═══════════════════════════════════════════════════════════════
def draw_detailed_attention():
    """详细展示 Multi-Head Attention 和 Position-wise FFN 的内部结构"""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.axis("off")

    # === Multi-Head Attention 部分 ===
    x_start = 0.05

    # Q, K, V 输入
    draw_box(ax, x_start, 0.80, 0.08, 0.05, "Input", COLORS["embed"], 8)
    draw_arrow(ax, x_start, 0.77, x_start, 0.73)

    draw_box(ax, x_start, 0.70, 0.10, 0.04, "Linear (W_Q)", COLORS["attn"], 7)
    draw_box(ax, x_start + 0.13, 0.70, 0.10, 0.04, "Linear (W_K)", COLORS["attn"], 7)
    draw_box(ax, x_start + 0.26, 0.70, 0.10, 0.04, "Linear (W_V)", COLORS["attn"], 7)

    draw_arrow(ax, x_start, 0.72, x_start, 0.74)
    draw_arrow(ax, x_start + 0.13, 0.72, x_start + 0.13, 0.74)
    draw_arrow(ax, x_start + 0.26, 0.72, x_start + 0.26, 0.74)

    draw_box(ax, x_start, 0.74, 0.10, 0.04, "Q", COLORS["embed"], 8, bold=True)
    draw_box(ax, x_start + 0.13, 0.74, 0.10, 0.04, "K", COLORS["embed"], 8, bold=True)
    draw_box(ax, x_start + 0.26, 0.74, 0.10, 0.04, "V", COLORS["embed"], 8, bold=True)

    # Scaled Dot-Product
    draw_arrow(ax, x_start, 0.72, x_start + 0.08, 0.65)
    draw_arrow(ax, x_start + 0.13, 0.72, x_start + 0.08, 0.65)

    draw_box(ax, x_start + 0.13, 0.62, 0.14, 0.06, "Scaled Dot-Product\nAttention\nsoftmax(QK^T / √d_k) · V",
             COLORS["attn"], 7)

    draw_arrow(ax, x_start + 0.20, 0.62, x_start + 0.30, 0.62)
    draw_box(ax, x_start + 0.38, 0.62, 0.10, 0.05, "Multi-Head\nConcat + Linear(W_O)",
             COLORS["attn"], 7)

    # === Position-wise Feed-Forward ===
    y_ffn = 0.30
    draw_box(ax, x_start + 0.20, y_ffn + 0.08, 0.14, 0.05, "Linear 1\n(d_model → d_ff)",
             COLORS["ffn"], 7)
    draw_box(ax, x_start + 0.20, y_ffn, 0.14, 0.05, "ReLU",
             COLORS["ffn"], 7)
    draw_box(ax, x_start + 0.20, y_ffn - 0.08, 0.14, 0.05, "Linear 2\n(d_ff → d_model)",
             COLORS["ffn"], 7)

    draw_arrow(ax, x_start + 0.20, y_ffn + 0.05, x_start + 0.20, y_ffn + 0.03)
    draw_arrow(ax, x_start + 0.20, y_ffn - 0.03, x_start + 0.20, y_ffn - 0.05)

    # === Positional Encoding ===
    x_pe = 0.75
    draw_box(ax, x_pe, 0.80, 0.18, 0.05, "Input Embedding", COLORS["embed"], 8)
    draw_arrow(ax, x_pe, 0.77, x_pe, 0.73)

    draw_box(ax, x_pe, 0.70, 0.18, 0.05, "+ Positional Encoding\nsin(pos/10000^(2i/d_model))",
             COLORS["pe"], 7)

    # === LayerNorm + Residual ===
    x_norm = 0.92
    draw_box(ax, x_norm, 0.62, 0.10, 0.04, "LayerNorm", COLORS["norm"], 7)
    draw_box(ax, x_norm, 0.30, 0.10, 0.04, "LayerNorm", COLORS["norm"], 7)

    draw_arrow(ax, x_start + 0.43, 0.62, x_norm - 0.05, 0.62)
    draw_arrow(ax, x_start + 0.27, 0.30, x_norm - 0.05, 0.30)

    # 残差标记
    ax.text(x_norm, 0.68, "Add", fontsize=8, ha="center", color="#C62828")
    ax.text(x_norm, 0.36, "Add", fontsize=8, ha="center", color="#C62828")

    ax.set_title("Transformer 内部组件详解\nMulti-Head Attention | Position-wise FFN | Positional Encoding",
                 fontsize=14, fontweight="bold", pad=10)

    fig.savefig(FIG_DIR / "fig02_attention_detail.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / 'fig02_attention_detail.png'}")


# ═══════════════════════════════════════════════════════════════
# 3. ViT 结构图
# ═══════════════════════════════════════════════════════════════
def draw_vit_structure():
    """ViT (Vision Transformer) 结构图"""
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis("off")

    # 输入图像
    draw_box(ax, 0.08, 0.60, 0.10, 0.06, "Image\n(224×224)", COLORS["embed"], 8)
    draw_arrow(ax, 0.13, 0.60, 0.18, 0.60)

    # Patch Embedding
    draw_box(ax, 0.23, 0.60, 0.12, 0.06, "Patch Embedding\n(16×16 patches → 196 tokens)",
             COLORS["attn"], 7)
    draw_arrow(ax, 0.29, 0.60, 0.34, 0.60)

    # + [CLS] token + Position Embedding
    draw_box(ax, 0.39, 0.60, 0.12, 0.06, "+ [CLS] +\nPosition Embedding",
             COLORS["pe"], 7)
    draw_arrow(ax, 0.45, 0.60, 0.50, 0.60)

    # Encoder Stack
    for i in range(3):
        x = 0.55 + i * 0.12
        draw_box(ax, x, 0.60, 0.10, 0.06, "Encoder\nBlock", COLORS["enc"], 7)
        if i < 2:
            draw_arrow(ax, x + 0.05, 0.60, x + 0.10, 0.60)

    ax.text(0.67, 0.68, "× L (e.g. 12)", fontsize=10, color="#C62828", fontweight="bold")

    draw_arrow(ax, 0.79, 0.60, 0.84, 0.60)

    # [CLS] output → Classifier
    draw_box(ax, 0.89, 0.60, 0.12, 0.06, "[CLS] Token →\nLinear Classifier",
             COLORS["embed"], 7)

    # 说明
    ax.text(0.50, 0.25,
            "ViT 关键差异：\n"
            "• 图像先切成 patch（不是词嵌入）\n"
            "• 每个 patch 线性投影为 token\n"
            "• 加 [CLS] token 用于分类\n"
            "• 只用 Encoder（无 Decoder）\n"
            "• 需要预训练（如 JFT-300M）才能在小数据集上工作",
            fontsize=10, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#FFF8E1", edgecolor="#FFB74D"))

    ax.set_title("ViT (Vision Transformer) 结构图\n与原始 Transformer 的差异",
                 fontsize=14, fontweight="bold", pad=10)

    fig.savefig(FIG_DIR / "fig03_vit_structure.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / 'fig03_vit_structure.png'}")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    draw_simple_structure()
    draw_detailed_attention()
    draw_vit_structure()
    print("\n所有结构图绘制完成！")
