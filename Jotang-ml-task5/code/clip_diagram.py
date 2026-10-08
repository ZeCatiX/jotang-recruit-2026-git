"""
绘制 CLIP 结构图

图 1：CLIP 整体架构（两个编码器 + 对比学习 + 特征空间对齐）
图 2：N×N 对比学习示意（正样本对 vs 负样本对）
图 3：Zero-shot 分类流程
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

FIG_DIR = Path(__file__).parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False

# ── 颜色方案 ──
C_IMG = "#4A90D9"      # 图片（蓝色）
C_TXT = "#E74C3C"      # 文本（红色）
C_ENC = "#F39C12"      # 编码器（橙色）
C_FEAT = "#27AE60"     # 特征（绿色）
C_SIM = "#8E44AD"      # 相似度（紫色）
C_LOSS = "#C0392B"     # 损失（深红）
C_POS = "#2ECC71"      # 正样本（绿色）
C_NEG = "#E74C3C"      # 负样本（红色）


def draw_box(ax, x, y, w, h, text, color="white", edgecolor="black",
            fontsize=10, fontweight="normal", text_color="black", alpha=1.0,
            boxstyle="round,pad=0.3"):
    """绘制圆角矩形框"""
    box = FancyBboxPatch((x, y), w, h, boxstyle=boxstyle,
                         facecolor=color, edgecolor=edgecolor, linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight, color=text_color, alpha=alpha)


def draw_arrow(ax, x1, y1, x2, y2, color="gray", linewidth=1.5, style="->", label="", alpha=1.0):
    """绘制箭头"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                            color=color, linewidth=linewidth, mutation_scale=15, alpha=alpha)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx, my + 0.15, label, ha="center", va="bottom", fontsize=8, color=color)


# ═══════════════════════════════════════════════════════════════
# 图 1：CLIP 整体架构
# ═══════════════════════════════════════════════════════════════
def plot_clip_architecture(fig_name="fig05_clip_architecture.png"):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ── 左侧：图片路径 ──
    draw_box(ax, 0.5, 6.0, 2.5, 1.2, "输入图片\n(224×224)", color=C_IMG, alpha=0.3,
             fontsize=10, text_color="black")
    draw_arrow(ax, 1.75, 6.0, 1.75, 5.3)
    draw_box(ax, 0.5, 4.0, 2.5, 1.3, "图像编码器\n(Vision Encoder)\nResNet-50 / ViT",
             color=C_ENC, alpha=0.3, fontsize=9)
    draw_arrow(ax, 1.75, 4.0, 1.75, 3.3)
    draw_box(ax, 0.5, 2.0, 2.5, 1.3, "投影到共享空间\n(Projection)\n输出: (N, 512)",
             color=C_FEAT, alpha=0.3, fontsize=9)

    # ── 右侧：文本路径 ──
    draw_box(ax, 11.0, 6.0, 2.5, 1.2, "输入文本\n(自然语言描述)", color=C_TXT, alpha=0.3,
             fontsize=10, text_color="black")
    draw_arrow(ax, 12.25, 6.0, 12.25, 5.3)
    draw_box(ax, 11.0, 4.0, 2.5, 1.3, "文本编码器\n(Text Encoder)\nTransformer",
             color=C_ENC, alpha=0.3, fontsize=9)
    draw_arrow(ax, 12.25, 4.0, 12.25, 3.3)
    draw_box(ax, 11.0, 2.0, 2.5, 1.3, "投影到共享空间\n(Projection)\n输出: (N, 512)",
             color=C_FEAT, alpha=0.3, fontsize=9)

    # ── 中间：相似度计算 ──
    draw_arrow(ax, 3.0, 2.65, 6.0, 2.65, color=C_SIM, linewidth=2)
    draw_arrow(ax, 11.0, 2.65, 8.0, 2.65, color=C_SIM, linewidth=2)
    draw_box(ax, 6.0, 1.8, 2.0, 1.7, "余弦相似度\n$cos(\\vec{f}_{img}, \\vec{f}_{txt})$",
             color=C_SIM, alpha=0.2, fontsize=10)

    # ── 下方：对比损失 ──
    draw_arrow(ax, 7.0, 1.8, 7.0, 1.0, color=C_LOSS, linewidth=2)
    draw_box(ax, 4.5, 0.0, 5.0, 1.0, "InfoNCE 对比损失\n$\\mathcal{L} = -\\log \\frac{e^{cos(f_i, t_i) / \\tau}}{\\sum_j e^{cos(f_i, t_j) / \\tau}}$",
             color=C_LOSS, alpha=0.15, fontsize=9)

    # ── 顶部：训练数据 ──
    draw_box(ax, 4.5, 7.2, 5.0, 0.7, "训练数据：互联网上的 4 亿 图片-文本对",
             color="#34495E", alpha=0.8, fontsize=10, text_color="white",
             boxstyle="round,pad=0.2")

    # 从训练数据到两个编码器
    draw_arrow(ax, 5.5, 7.2, 1.75, 6.0, color="#34495E", linewidth=1.5, style="-|>")
    draw_arrow(ax, 8.5, 7.2, 12.25, 6.0, color="#34495E", linewidth=1.5, style="-|>")

    # ── 标题 ──
    ax.set_title("CLIP 整体架构：对比学习对齐图像和文本特征空间",
                 fontsize=14, fontweight="bold", pad=20)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 图 2：N×N 对比学习
# ═══════════════════════════════════════════════════════════════
def plot_contrastive_learning(fig_name="fig06_contrastive_learning.png"):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    N = 4

    # ── 左侧：图片 ──
    img_labels = ["图片 1\n(猫)", "图片 2\n(狗)", "图片 3\n(铁塔)", "图片 4\n(披萨)"]
    txt_labels = ["文本 1\n(猫)", "文本 2\n(狗)", "文本 3\n(铁塔)", "文本 4\n(披萨)"]

    for i in range(N):
        y = 8.0 - i * 1.8
        draw_box(ax, 0.3, y - 0.4, 1.5, 0.8, img_labels[i],
                 color=C_IMG, alpha=0.3, fontsize=8)

    # ── 右侧：文本 ──
    for j in range(N):
        y = 8.0 - j * 1.8
        draw_box(ax, 8.2, y - 0.4, 1.5, 0.8, txt_labels[j],
                 color=C_TXT, alpha=0.3, fontsize=8)

    # ── 中间：N×N 配对 ──
    # 对角线 = 正样本，非对角线 = 负样本
    for i in range(N):
        for j in range(N):
            y_img = 8.0 - i * 1.8
            y_txt = 8.0 - j * 1.8

            if i == j:
                # 正样本：绿色，粗线
                draw_arrow(ax, 1.8, y_img, 8.2, y_txt, color=C_POS, linewidth=2.5, style="-|>")
                ax.text(5.0, (y_img + y_txt)/2 + 0.1, "正样本对",
                        ha="center", fontsize=8, color=C_POS, fontweight="bold")
            else:
                # 负样本：红色，细线
                draw_arrow(ax, 1.8, y_img, 8.2, y_txt, color=C_NEG, linewidth=0.5,
                           style="-|>", alpha=0.2)

    # ── 标注 ──
    ax.text(5.0, 9.5, "N×N 对比学习：1 个正样本对 + N-1 个负样本对",
            fontsize=12, fontweight="bold", ha="center")
    ax.text(5.0, 9.0, "（绿色粗线=正样本，红色细线=负样本）",
            fontsize=10, ha="center", color="gray")

    # 图例
    draw_box(ax, 0.3, 0.3, 4.5, 0.5, "[正] 正样本对: 提高相似度 cos(f_i, t_i)",
             color=C_POS, alpha=0.15, fontsize=9)
    draw_box(ax, 5.2, 0.3, 4.5, 0.5, "[负] 负样本对: 降低相似度 cos(f_i, t_j)",
             color=C_NEG, alpha=0.15, fontsize=9)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 图 3：Zero-shot 分类
# ═══════════════════════════════════════════════════════════════
def plot_zeroshot_classification(fig_name="fig07_zeroshot_classification.png"):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # ── 输入图片 ──
    draw_box(ax, 0.3, 2.3, 1.5, 1.4, "输入图片\n(猫)", color=C_IMG, alpha=0.3, fontsize=10)
    draw_arrow(ax, 1.8, 3.0, 2.8, 3.0, linewidth=2)
    draw_box(ax, 2.8, 2.3, 1.5, 1.4, "图像编码器\n→ 图像特征\n$f_{img}$",
             color=C_FEAT, alpha=0.3, fontsize=9)

    # ── 候选类别 ──
    classes = ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a tree"]
    draw_box(ax, 6.5, 2.3, 2.0, 1.4, "候选类别\n(类名文本)", color=C_TXT, alpha=0.2,
             fontsize=10, text_color="black")

    for i, cls in enumerate(classes):
        y = 4.5 - i * 0.7
        draw_box(ax, 9.5, y - 0.25, 3.5, 0.5, cls, color=C_TXT, alpha=0.1, fontsize=8)

    # 类别 → 文本编码器
    draw_arrow(ax, 8.5, 3.0, 9.5, 3.85, color=C_ENC, linewidth=1)
    draw_arrow(ax, 8.5, 3.0, 9.5, 3.15, color=C_ENC, linewidth=1)
    draw_arrow(ax, 8.5, 3.0, 9.5, 2.45, color=C_ENC, linewidth=1)
    draw_arrow(ax, 8.5, 3.0, 9.5, 1.75, color=C_ENC, linewidth=1)

    # 文本编码器 → 文本特征
    for i in range(4):
        y = 4.5 - i * 0.7
        draw_arrow(ax, 13.0, y, 13.5, y, color=C_FEAT, linewidth=1)
        draw_box(ax, 13.5, y - 0.15, 0.5, 0.3, "→", color=C_FEAT, alpha=0.3, fontsize=8)

    # 相似度计算
    draw_box(ax, 6.5, 0.3, 2.0, 1.0, "计算余弦相似度\n$cos(f_{img}, f_{cls})$",
             color=C_SIM, alpha=0.2, fontsize=9)

    # 从图像特征到相似度
    draw_arrow(ax, 4.3, 3.0, 7.5, 1.3, color=C_SIM, linewidth=1.5)

    # 输出：分类结果
    draw_arrow(ax, 8.5, 0.8, 9.5, 0.8, color=C_LOSS, linewidth=2)
    draw_box(ax, 9.5, 0.3, 3.5, 1.0, "输出概率分布\n[0.82, 0.05, 0.08, 0.05]\n→ 类别: cat (82%)",
             color=C_LOSS, alpha=0.15, fontsize=9)

    ax.set_title("Zero-shot 分类：无需训练，用类名文本直接分类",
                 fontsize=13, fontweight="bold", pad=15)

    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


if __name__ == "__main__":
    print("=" * 60)
    print("绘制 CLIP 结构图")
    print("=" * 60)

    plot_clip_architecture()
    plot_contrastive_learning()
    plot_zeroshot_classification()

    print("\n所有图绘制完成！")
