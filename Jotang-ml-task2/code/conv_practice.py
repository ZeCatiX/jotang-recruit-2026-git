"""
实践 1：卷积核
───────────────────────────────────────────────────────────────
1. 图片表示：像素、RGB、灰度图 vs 彩色图、HWC vs CHW
2. 手写二维卷积（不调用任何现成卷积库）
3. 四种卷积核：均值模糊 / 高斯模糊 / 锐化 / Sobel 边缘检测
4. valid vs same padding 对比
5. 输出尺寸公式
"""
import numpy as np
import torch
from PIL import Image
from pathlib import Path
from typing import Tuple

# ── 设置字体和输出路径 ──────────────────────────────────────────────────
FIG_DIR = Path(__file__).parent.parent / "figures"
DATA_DIR = Path(__file__).parent.parent / "data"
FIG_DIR.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════════════
# 1. 图片表示
# ═══════════════════════════════════════════════════════════════════════
def part1_image_representation():
    """像素、RGB、灰度 vs 彩色、HWC vs CHW"""
    print("=" * 70)
    print("实践 1.1  图片表示")
    print("=" * 70)

    img_path = DATA_DIR / "scene.png"
    img = Image.open(img_path)
    print(f"原图: {img_path.name}, size={img.size}, mode={img.mode}")

    # Pillow → NumPy
    arr = np.array(img)
    print(f"\nNumPy 数组:")
    print(f"  shape  = {arr.shape}  →  (H={arr.shape[0]}, W={arr.shape[1]}, C={arr.shape[2]})")
    print(f"  dtype  = {arr.dtype}")
    print(f"  min    = {arr.min()},  max = {arr.max()}")
    print(f"  一个像素 (y=270, x=480) 的 RGB: {arr[270, 480]}")
    print(f"  各通道范围: R=[{arr[:,:,0].min()},{arr[:,:,0].max()}]  "
          f"G=[{arr[:,:,1].min()},{arr[:,:,1].max()}]  "
          f"B=[{arr[:,:,2].min()},{arr[:,:,2].max()}]")

    # HWC vs CHW
    print(f"\nHWC vs CHW 对比:")
    print(f"  NumPy / Pillow 使用 HWC: (H, W, C) = {arr.shape}")
    chw = np.transpose(arr, (2, 0, 1))
    print(f"  PyTorch 使用 CHW:      (C, H, W) = {chw.shape}")
    print(f"  → 对神经网络来说，CHW 让卷积核直接按通道操作，无需转置")

    # 灰度图
    gray = img.convert("L")
    gray_arr = np.array(gray)
    print(f"\n灰度图:")
    print(f"  shape = {gray_arr.shape}  →  只有 (H, W)，没有通道维")
    print(f"  dtype = {gray_arr.dtype}")
    print(f"  每个像素只有一个亮度值 0-255，信息量只有彩色图的 1/3")

    # Pillow → Tensor
    tensor = torch.from_numpy(arr).float() / 255.0
    print(f"\nPyTorch Tensor:")
    print(f"  shape = {tensor.shape}")
    print(f"  dtype = {tensor.dtype}")
    print(f"  min = {tensor.min():.4f},  max = {tensor.max():.4f}")
    print(f"  像素 (270, 480) 归一化后: {tensor[270, 480].tolist()}")

    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(img)
    axes[0].set_title("原图 (RGB, 彩色)")
    axes[0].axis("off")

    axes[1].imshow(gray_arr, cmap="gray")
    axes[1].set_title("灰度图 (1 通道, 亮度)")
    axes[1].axis("off")

    # 展示单个像素的 RGB
    px = arr[270, 480]
    im = np.ones((1, 1, 3), dtype=np.uint8) * px
    axes[2].imshow(im, extent=[0, 1, 0, 1])
    axes[2].set_title(f"单个像素 RGB={tuple(px)}")
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    fig.suptitle("实践 1.1：图片表示——像素、RGB、灰度 vs 彩色", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_图片表示.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n已保存: {FIG_DIR / 'fig01_图片表示.png'}")

    return arr


# ═══════════════════════════════════════════════════════════════════════
# 2. 手写二维卷积
# ═══════════════════════════════════════════════════════════════════════
def conv2d_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    手写二维卷积（valid 模式，无填充）。
    支持灰度图 (H, W) 或彩色图 (H, W, C)。
    kernel 可以是 2D (kh, kw) 或 3D (kh, kw, C)。
    """
    # 如果是灰度图，加一个通道维
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    H, W, C = image.shape

    # 如果 kernel 是 2D，扩展为单通道
    if kernel.ndim == 2:
        kh, kw = kernel.shape
        kernel = np.stack([kernel] * C, axis=2)  # (kh, kw, C)

    kh, kw, kc = kernel.shape

    out_h = H - kh + 1
    out_w = W - kw + 1

    # 初始化输出：与图片同结构，但尺寸缩小
    out = np.zeros((out_h, out_w, C), dtype=np.float64)

    # 逐位置滑动卷积核
    for i in range(out_h):
        for j in range(out_w):
            # 提取当前窗口的数据块
            patch = image[i:i+kh, j:j+kw, :]  # shape: (kh, kw, C)
            # 逐元素相乘后求和 → 输出一个通道值
            out[i, j, :] = np.sum(patch * kernel, axis=(0, 1))

    return out


def conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    手写二维卷积（same 模式，零填充）。
    输出尺寸与输入相同。
    """
    # 如果 kernel 是 2D，先扩展
    if kernel.ndim == 2:
        if image.ndim == 2:
            kernel = np.stack([kernel] * 1, axis=2)
        else:
            kernel = np.stack([kernel] * image.shape[2], axis=2)

    kh, kw, _ = kernel.shape
    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # 零填充
    if image.ndim == 2:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)),
                        mode="constant", constant_values=0.0)
    else:
        padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                        mode="constant", constant_values=0.0)
    return conv2d_valid(padded, kernel)


def conv2d_with_stride(
    image: np.ndarray, kernel: np.ndarray, stride: int = 1
) -> np.ndarray:
    """
    手写二维卷积，支持 stride（步长）。
    valid 模式。
    """
    if image.ndim == 2:
        image = image[:, :, np.newaxis]

    H, W, C = image.shape

    if kernel.ndim == 2:
        kh, kw = kernel.shape
        kernel = np.stack([kernel] * C, axis=2)

    kh, kw, _ = kernel.shape

    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((out_h, out_w, C), dtype=np.float64)

    for i in range(out_h):
        for j in range(out_w):
            patch = image[i*stride:(i*stride+kh), j*stride:(j*stride+kw), :]
            out[i, j, :] = np.sum(patch * kernel, axis=(0, 1))

    return out


def output_size_formula(H, W, K, stride, padding):
    """输出尺寸公式: out = floor((H - K + 2*P) / S) + 1"""
    return (H - K + 2 * padding) // stride + 1


# ═══════════════════════════════════════════════════════════════════════
# 3. 四种卷积核
# ═══════════════════════════════════════════════════════════════════════
def generate_kernels():
    """生成四种卷积核，每个支持单通道（自动广播到 3 通道）"""

    # 1. 均值模糊 3×3
    mean_kernel_2d = np.ones((3, 3), dtype=np.float64) / 9
    mean_kernel = np.stack([mean_kernel_2d] * 3, axis=2)  # (3, 3, 3)

    # 2. 高斯模糊 5×5
    sigma = 1.0
    size = 5
    axis = np.arange(-size // 2, size // 2 + 1)
    g = np.exp(-axis ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    gaussian_2d = np.outer(g, g)
    gaussian_2d /= gaussian_2d.sum()
    gaussian_kernel = np.stack([gaussian_2d] * 3, axis=2)

    # 3. 锐化 3×3
    sharpen_2d = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float64)
    sharpen_kernel = np.stack([sharpen_2d] * 3, axis=2)

    # 4. Sobel 边缘检测
    sobel_x_2d = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
    sobel_y_2d = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
    sobel_x = np.stack([sobel_x_2d] * 3, axis=2)
    sobel_y = np.stack([sobel_y_2d] * 3, axis=2)

    return {
        "mean_3x3": (mean_kernel, mean_kernel_2d, "均值模糊 3×3"),
        "gaussian_5x5": (gaussian_kernel, gaussian_2d, "高斯模糊 5×5"),
        "sharpen_3x3": (sharpen_kernel, sharpen_2d, "锐化 3×3"),
        "sobel_x": (sobel_x, sobel_x_2d, "Sobel-X"),
        "sobel_y": (sobel_y, sobel_y_2d, "Sobel-Y"),
    }


def clamp_and_save(result: np.ndarray, save_path, title: str, ax=None):
    """将结果裁剪到 [0, 255] 并保存或显示"""
    if result.ndim == 2:
        # 灰度图
        result = np.clip(result, 0, 255)
        im = result.astype(np.uint8)
    else:
        # 彩色图
        result = np.clip(result, 0, 255)
        im = result.astype(np.uint8)

    if ax is None:
        plt.figure()
        plt.imshow(im)
        plt.title(title)
        plt.axis("off")
    else:
        ax.imshow(im)
        ax.set_title(title)
        ax.axis("off")

    return im


# ═══════════════════════════════════════════════════════════════════════
# 4. 主流程
# ═══════════════════════════════════════════════════════════════════════
def part2_convolution(img_arr: np.ndarray):
    """展示四种卷积核的效果"""
    print("\n" + "=" * 70)
    print("实践 1.2  四种卷积核处理同一张图片")
    print("=" * 70)

    kernels = generate_kernels()

    # 先转灰度图做卷积（卷积核按单通道设计，灰度图更清晰）
    gray = np.array(Image.open(DATA_DIR / "scene.png").convert("L"))
    print(f"\n使用灰度图: shape={gray.shape}")

    # 可视化四种卷积核的效果
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 原图
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("① 原图 (灰度)")
    axes[0, 0].axis("off")

    # 均值模糊
    mean_result = conv2d_valid(gray, kernels["mean_3x3"][1])
    axes[0, 1].imshow(np.clip(mean_result, 0, 255).astype(np.uint8), cmap="gray")
    axes[0, 1].set_title("② 均值模糊 3×3\n(每个像素取邻域平均值)")
    axes[0, 1].axis("off")

    # 高斯模糊
    gauss_result = conv2d_valid(gray, kernels["gaussian_5x5"][1])
    axes[0, 2].imshow(np.clip(gauss_result, 0, 255).astype(np.uint8), cmap="gray")
    axes[0, 2].set_title("③ 高斯模糊 5×5\n(中心权重大，边缘权重小)")
    axes[0, 2].axis("off")

    # 锐化
    sharpen_result = conv2d_valid(gray, kernels["sharpen_3x3"][1])
    axes[1, 0].imshow(np.clip(sharpen_result, 0, 255).astype(np.uint8), cmap="gray")
    axes[1, 0].set_title("④ 锐化 3×3\n(中心增强，邻域抑制)")
    axes[1, 0].axis("off")

    # Sobel 边缘检测
    sx = conv2d_valid(gray, kernels["sobel_x"][1])
    sy = conv2d_valid(gray, kernels["sobel_y"][1])
    sobel_combined = np.sqrt(sx ** 2 + sy ** 2)
    axes[1, 1].imshow(np.clip(sobel_combined / sobel_combined.max() * 255, 0, 255).astype(np.uint8), cmap="gray")
    axes[1, 1].set_title("⑤ Sobel 边缘检测\n(横向 + 纵向梯度合成)")
    axes[1, 1].axis("off")

    # 卷积核展示
    ax = axes[1, 2]
    kernel_names = ["均值 3×3", "高斯 5×5", "锐化 3×3", "Sobel-X", "Sobel-Y"]
    kernel_vals = [k[1] for k in kernels.values()]

    # 显示四个主要卷积核
    gs = ax.get_gridspec()
    for idx, (name, val) in enumerate(zip(["均值 3×3", "高斯 5×5", "锐化 3×3"],
                                           [kernels["mean_3x3"][1],
                                            kernels["gaussian_5x5"][1],
                                            kernels["sharpen_3x3"][1]])):
        sub_ax = ax.get_subplots() if hasattr(ax, "get_subplots") else None
        # 简单展示
        sub = fig.add_axes([0.67 + (idx % 2) * 0.16, 0.08 + (idx // 2) * 0.12,
                            0.13, 0.1])
        sub.imshow(val, cmap="hot", vmin=-1, vmax=1)
        sub.set_title(name, fontsize=8)
        sub.axis("off")
    # 不占位，删除原 axes
    ax.axis("off")

    fig.suptitle("实践 1.2：四种卷积核的效果对比", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_四种卷积核.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 打印输出尺寸
    print(f"\n各卷积核的输出尺寸:")
    H, W = gray.shape
    for name, (kernel_3ch, kernel_1ch, label) in kernels.items():
        out = conv2d_valid(gray, kernel_1ch)
        print(f"  {label:15s} 核大小={kernel_1ch.shape[:2]}  "
              f"输出: {out.shape}  有效范围: [{out.min():.1f}, {out.max():.1f}]")

    print(f"\n已保存: {FIG_DIR / 'fig02_四种卷积核.png'}")

    return mean_result, gauss_result, sharpen_result, sobel_combined


def part3_padding_comparison(img_arr: np.ndarray):
    """valid vs same padding 对比"""
    print("\n" + "=" * 70)
    print("实践 1.3  valid vs same padding")
    print("=" * 70)

    gray = np.array(Image.open(DATA_DIR / "scene.png").convert("L"))
    H, W = gray.shape
    kernel = kernels["mean_3x3"][1] if 'kernels' in dir() else None

    kernels = generate_kernels()
    kernel = kernels["sharpen_3x3"][1]  # 用锐化核，效果明显

    # valid
    out_valid = conv2d_valid(gray, kernel)
    # same
    out_same = conv2d_same(gray, kernel)

    print(f"\n输入尺寸: {gray.shape} (H={H}, W={W})")
    print(f"卷积核: {kernel.shape[0]}×{kernel.shape[1]}")
    print(f"  valid 输出: {out_valid.shape}  →  {(H-3+1), (W-3+1)}")
    print(f"  same  输出: {out_same.shape}  →  与输入相同 {(H, W)}")

    # 公式验证
    print(f"\n输出尺寸公式: out = floor((H - K + 2P) / S) + 1")
    for K, P, S in [(3, 0, 1), (3, 1, 1), (5, 2, 1), (3, 1, 2), (3, 0, 2)]:
        out = output_size_formula(H, W, K, S, P)
        print(f"  K={K}, P={P}, S={S} → out = ({H}-{K}+2×{P})/{S}+1 = {out}")

    # 可视化对比
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title(f"原图 {H}×{W}")
    axes[0].axis("off")

    axes[1].imshow(np.clip(out_valid, 0, 255).astype(np.uint8), cmap="gray")
    axes[1].set_title(f"valid (无填充)\n{(out_valid.shape[1])}×{out_valid.shape[0]}\n边缘丢失 1 像素")
    axes[1].axis("off")

    axes[2].imshow(np.clip(out_same, 0, 255).astype(np.uint8), cmap="gray")
    axes[2].set_title(f"same (零填充 P=1)\n{out_same.shape[1]}×{out_same.shape[0]}\n边缘用 0 填充")
    axes[2].axis("off")

    fig.suptitle("实践 1.3：valid vs same padding 对比", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_padding对比.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 展示边缘像素差异
    print(f"\n边缘像素对比 (左上角 5×5 区域):")
    print("  原图:")
    print(f"  {gray[:5, :5]}")
    print(f"  valid (输出缩小，无对应像素):")
    print(f"  {np.round(out_valid[:5, :5], 1)}")
    print(f"  same (有边缘像素，受零填充影响):")
    print(f"  {np.round(out_same[:5, :5], 1)}")

    print(f"\n已保存: {FIG_DIR / 'fig03_padding对比.png'}")


def part4_color_convolution(img_arr: np.ndarray):
    """彩色图卷积 + 数值裁剪"""
    print("\n" + "=" * 70)
    print("实践 1.4  彩色图卷积与数值裁剪")
    print("=" * 70)

    gray = np.array(Image.open(DATA_DIR / "scene.png").convert("L"))
    kernels = generate_kernels()

    # 用锐化核处理彩色图
    sharpen_3ch = kernels["sharpen_3x3"][0]  # (3, 3, 3)
    result = conv2d_valid(img_arr, sharpen_3ch)

    print(f"\n彩色图锐化:")
    print(f"  输入: shape={img_arr.shape}, dtype={img_arr.dtype}, range=[{img_arr.min()}, {img_arr.max()}]")
    print(f"  输出: shape={result.shape}, dtype={result.dtype}, range=[{result.min():.1f}, {result.max():.1f}]")
    print(f"  → 输出范围超出 [0, 255]，需要裁剪")
    print(f"  → 负值 {int(np.sum(result < 0))} 个像素，超 255 的 {int(np.sum(result > 255))} 个像素")

    # 裁剪前后对比
    clipped = np.clip(result, 0, 255).astype(np.uint8)
    print(f"\n裁剪后: range=[{clipped.min()}, {clipped.max()}], dtype={clipped.dtype}")

    # 高斯模糊对彩色图的影响
    gauss_3ch = kernels["gaussian_5x5"][0]
    gauss_result = conv2d_valid(img_arr, gauss_3ch)
    gauss_clipped = np.clip(gauss_result, 0, 255).astype(np.uint8)
    print(f"\n高斯模糊: 输出 range=[{gauss_result.min():.1f}, {gauss_result.max():.1f}]")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].imshow(img_arr)
    axes[0, 0].set_title("原图 (RGB)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(clipped)
    axes[0, 1].set_title("锐化 (裁剪到 [0,255])")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(gauss_clipped)
    axes[0, 2].set_title("高斯模糊 5×5")
    axes[0, 2].axis("off")

    # 未裁剪 vs 裁剪
    axes[1, 0].imshow(np.clip(result / result.max() * 255, 0, 255).astype(np.uint8))
    axes[1, 0].set_title(f"锐化 (线性归一化)\n显示范围 [{result.min():.0f}, {result.max():.0f}]")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(result, 0, 255).astype(np.uint8))
    axes[1, 1].set_title("锐化 (裁剪到 [0,255])\n丢失了部分细节")
    axes[1, 1].axis("off")

    # 展示数值分布
    ax = axes[1, 2]
    ax.hist(result.ravel(), bins=100, range=(0, 300), color="steelblue", alpha=0.7)
    ax.set_xlabel("像素值")
    ax.set_ylabel("频数")
    ax.set_title("锐化后像素值分布\n(有值超出 [0,255])")
    ax.axvline(0, color="red", ls="--", label="0 (下界)")
    ax.axvline(255, color="green", ls="--", label="255 (上界)")
    ax.legend()

    fig.suptitle("实践 1.4：彩色图卷积与数值裁剪", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig04_彩色图卷积与裁剪.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n已保存: {FIG_DIR / 'fig04_彩色图卷积与裁剪.png'}")


def main():
    img_arr = part1_image_representation()
    part2_convolution(img_arr)
    part3_padding_comparison(img_arr)
    part4_color_convolution(img_arr)

    print("\n" + "=" * 70)
    print("实践 1 完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()