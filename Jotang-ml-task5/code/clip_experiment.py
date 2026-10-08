"""
CLIP 图文配对实验

任务 2：让 CLIP 配对图片和文字

流程：
  1. 加载 OpenAI CLIP (RN50) 预训练权重
  2. 提取 4 张图片的特征 + 4 段描述的特征
  3. L2 归一化
  4. 计算 4×4 余弦相似度矩阵
  5. 画热力图，检查配对是否正确
  6. 加入误导性描述（颜色/数量/动作错误），观察相似度变化
"""
import os
import json
import numpy as np
import torch
import clip
from PIL import Image
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ── 路径 ──
BASE_DIR = Path(__file__).parent.parent
IMG_DIR = BASE_DIR / "images"
FIG_DIR = BASE_DIR / "figures"
RES_DIR = BASE_DIR / "results"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RES_DIR.mkdir(parents=True, exist_ok=True)

# ── 中文字体 ──
FONT_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
fm.fontManager.addfont(FONT_PATH)
plt.rcParams["font.family"] = "Noto Sans SC"
plt.rcParams["axes.unicode_minus"] = False


# ═══════════════════════════════════════════════════════════════
# 1. 加载 CLIP 模型
# ═══════════════════════════════════════════════════════════════
def load_clip_model(model_name="RN50", device="cpu"):
    """加载 CLIP 预训练模型和 tokenizer"""
    print(f"加载 CLIP 模型: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    print(f"  设备: {device}")
    print(f"  图像编码器输出维度: {model.visual.output_dim}")
    print(f"  文本编码器输出维度: {model.transformer.get_output_embeddings().out_features if hasattr(model.transformer, 'get_output_embeddings') else '见下文'}")
    return model, preprocess, clip.tokenize


# ═══════════════════════════════════════════════════════════════
# 2. 提取特征
# ═══════════════════════════════════════════════════════════════
@torch.no_grad()
def extract_features(model, preprocess, tokenizer, image_paths, texts, device="cpu"):
    """
    提取图像特征和文本特征
    
    参数:
        image_paths: 图片路径列表
        texts: 文本描述列表
    
    返回:
        image_features: (N, D) 图像特征 (L2 归一化后)
        text_features: (M, D) 文本特征 (L2 归一化后)
    """
    # ── 图像特征 ──
    images = []
    for p in image_paths:
        img = Image.open(p).convert("RGB")
        img = preprocess(img)  # resize + normalize
        images.append(img)
    image_batch = torch.stack(images).to(device)  # (N, 3, 224, 224)
    
    image_features = model.encode_image(image_batch)  # (N, D)
    
    # L2 归一化
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # ── 文本特征 ──
    text_tokens = tokenizer(texts, context_length=77, truncate=True).to(device)  # (M, 77)
    text_features = model.encode_text(text_tokens)  # (M, D)
    
    # L2 归一化
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return image_features.cpu().numpy(), text_features.cpu().numpy()


# ═══════════════════════════════════════════════════════════════
# 3. 计算相似度矩阵
# ═══════════════════════════════════════════════════════════════
def compute_similarity_matrix(image_features, text_features, temperature=None):
    """
    计算余弦相似度矩阵
    
    由于特征已 L2 归一化，点积 = 余弦相似度
    
    参数:
        image_features: (N, D) L2 归一化后的图像特征
        text_features: (M, D) L2 归一化后的文本特征
        temperature: 可选温度系数（论文中的可学习参数，默认 0.07 即 1/14.2）
    
    返回:
        similarity: (N, M) 相似度矩阵
    """
    # 点积 = 余弦相似度（因为已归一化）
    similarity = image_features @ text_features.T  # (N, M)
    
    # 应用温度系数（如果需要）
    if temperature is not None:
        similarity = similarity / temperature
    
    return similarity


# ═══════════════════════════════════════════════════════════════
# 4. 绘制热力图
# ═══════════════════════════════════════════════════════════════
def plot_similarity_heatmap(similarity, image_labels, text_labels, 
                            correct_pairs=None, title="", fig_name="heatmap.png"):
    """
    绘制相似度热力图
    
    参数:
        similarity: (N, M) 相似度矩阵
        image_labels: 图像标签列表
        text_labels: 文本标签列表
        correct_pairs: 正确配对的索引列表 [(img_idx, txt_idx), ...]
    """
    n_img = len(image_labels)
    n_txt = len(text_labels)
    
    fig, ax = plt.subplots(figsize=(max(8, 3 * n_txt + 2), max(6, 3 * n_img + 2)))
    
    # 绘制热力图
    im = ax.imshow(similarity, cmap="RdYlGn", aspect="auto", vmin=-0.2, vmax=0.4)
    fig.colorbar(im, ax=ax, label="余弦相似度", fraction=0.046, pad=0.04)
    
    # 设置轴标签
    ax.set_xticks(range(n_txt))
    ax.set_yticks(range(n_img))
    ax.set_xticklabels(text_labels, fontsize=9, rotation=30, ha="right")
    ax.set_yticklabels(image_labels, fontsize=9)
    ax.set_xlabel("文本描述", fontsize=11)
    ax.set_ylabel("图片", fontsize=11)
    
    # 标注每个格子的数值
    for i in range(n_img):
        for j in range(n_txt):
            val = similarity[i, j]
            color = "white" if val < 0 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=8, color=color)
    
    # 标注正确配对
    if correct_pairs:
        for img_idx, txt_idx in correct_pairs:
            ax.text(txt_idx, img_idx, "★", ha="center", va="center",
                    fontsize=14, color="blue", fontweight="bold")
    
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fig_name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / fig_name}")


# ═══════════════════════════════════════════════════════════════
# 5. 分析配对结果
# ═══════════════════════════════════════════════════════════════
def analyze_matching(similarity, image_labels, text_labels):
    """
    分析每张图的最佳匹配文本
    """
    n_img = similarity.shape[0]
    results = []
    all_correct = True
    
    print(f"\n{'─' * 70}")
    print("配对分析：每张图最相似的文本描述")
    print(f"{'─' * 70}")
    
    for i in range(n_img):
        row = similarity[i]
        best_idx = row.argmax()
        best_sim = row[best_idx]
        second_idx = np.argsort(row)[-2]
        second_sim = row[second_idx]
        margin = best_sim - second_sim
        
        # 判断是否匹配正确（假设第 i 张图对应第 i 个文本）
        is_correct = (best_idx == i)
        
        results.append({
            "image": image_labels[i],
            "best_match": text_labels[best_idx],
            "best_similarity": float(best_sim),
            "second_best": text_labels[second_idx],
            "second_similarity": float(second_sim),
            "margin": float(margin),
            "correct": bool(is_correct)
        })
        
        status = "✅ 正确" if is_correct else "❌ 错误"
        print(f"  {image_labels[i]:20s} → 最匹配: {text_labels[best_idx]:30s} "
              f"(sim={best_sim:.4f}, margin={margin:.4f}) {status}")
        
        if not is_correct:
            all_correct = False
    
    print(f"\n  {'✅ 全部配对正确！' if all_correct else '❌ 存在配对错误'}")
    return results, all_correct


# ═══════════════════════════════════════════════════════════════
# 6. 主函数
# ═══════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("CLIP 图文配对实验")
    print("=" * 70)
    
    # ── 加载模型 ──
    model, preprocess, tokenizer = load_clip_model("RN50", device="cpu")
    
    # ── 准备数据和描述 ──
    # 4 张图片 + 正确的英文描述
    image_paths = [
        IMG_DIR / "img1_cat.jpg",
        IMG_DIR / "img2_dog.jpg",
        IMG_DIR / "img3_eiffel.jpg",
        IMG_DIR / "img4_pizza.jpg",
    ]
    image_labels = ["Cat", "Dog", "Eiffel Tower", "Pizza"]
    
    correct_texts = [
        "a domestic cat sitting on a couch",
        "a dog walking in a park",
        "the Eiffel Tower in Paris at sunset",
        "a pizza with cheese and toppings",
    ]
    
    # 验证图片存在
    for p, label in zip(image_paths, image_labels):
        exists = p.exists()
        size = p.stat().st_size if exists else 0
        print(f"  {'✅' if exists else '❌'} {label:15s}: {p.name} ({size/1024:.0f} KB)")
    
    print(f"\n图片数量: {len(image_paths)}")
    print(f"描述数量: {len(correct_texts)}")
    
    # ═══════════════════════════════════════════════════════════
    # 实验 1：正确描述配对
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("实验 1：4 张图片 + 4 条正确描述")
    print(f"{'=' * 70}")
    
    image_features, text_features = extract_features(
        model, preprocess, tokenizer, image_paths, correct_texts, device="cpu"
    )
    
    print(f"\n图像特征形状: {image_features.shape}  (N_images, D)")
    print(f"文本特征形状: {text_features.shape}  (N_texts, D)")
    print(f"特征维度 D = {image_features.shape[1]}")
    
    # 验证 L2 归一化
    img_norms = np.linalg.norm(image_features, axis=1)
    txt_norms = np.linalg.norm(text_features, axis=1)
    print(f"\nL2 归一化验证:")
    print(f"  图像特征范数: {img_norms}  (应全部为 1.0)")
    print(f"  文本特征范数: {txt_norms}  (应全部为 1.0)")
    
    # 计算相似度矩阵
    similarity = compute_similarity_matrix(image_features, text_features)
    print(f"\n相似度矩阵形状: {similarity.shape}")
    print(f"\n相似度矩阵:")
    print("         ", "  ".join(f"{t[:12]:>12s}" for t in correct_texts))
    for i, label in enumerate(image_labels):
        row = "  ".join(f"{s:>12.4f}" for s in similarity[i])
        print(f"  {label:12s}: {row}")
    
    # 检查配对
    results, all_correct = analyze_matching(similarity, image_labels, correct_texts)
    
    # 绘制热力图
    correct_pairs = [(i, i) for i in range(4)]
    plot_similarity_heatmap(
        similarity, image_labels, correct_texts,
        correct_pairs=correct_pairs,
        title="实验 1：正确描述配对 (★=正确配对)",
        fig_name="fig01_similarity_heatmap_correct.png"
    )
    
    # ═══════════════════════════════════════════════════════════
    # 实验 2：加入误导性描述
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("实验 2：加入误导性描述（颜色/数量/动作错误）")
    print(f"{'=' * 70}")
    
    # 误导性描述：与正确描述相似，但关键内容错误
    misleading_texts = [
        "a tabby dog sitting on a couch",          # cat → dog (错误动物)
        "a cat walking in a park",                 # dog → cat (错误动物)
        "the Statue of Liberty in New York",       # Eiffel Tower → Statue of Liberty (错误地标)
        "a hamburger on a plate",                  # pizza → hamburger (错误食物)
    ]
    
    print("\n误导性描述对比:")
    for i, (correct, misleading) in enumerate(zip(correct_texts, misleading_texts)):
        print(f"  图片 {i+1} ({image_labels[i]}):")
        print(f"    正确:   {correct}")
        print(f"    错误:   {misleading}")
    
    # 用误导性描述计算相似度
    image_features2, text_features2 = extract_features(
        model, preprocess, tokenizer, image_paths, misleading_texts, device="cpu"
    )
    similarity_wrong = compute_similarity_matrix(image_features2, text_features2)
    
    # 分析配对
    results_wrong, all_correct_wrong = analyze_matching(
        similarity_wrong, image_labels, misleading_texts
    )
    
    # 绘制误导性描述热力图
    plot_similarity_heatmap(
        similarity_wrong, image_labels, misleading_texts,
        correct_pairs=None,
        title="实验 2：误导性描述配对 (★=正确配对，应为对角线)",
        fig_name="fig02_similarity_heatmap_misleading.png"
    )
    
    # ═══════════════════════════════════════════════════════════
    # 实验 3：对比正确 vs 误导性描述
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("实验 3：正确 vs 误导性描述对比")
    print(f"{'=' * 70}")
    
    # 混合正确和错误描述，看 CLIP 能否区分
    mixed_texts = correct_texts + misleading_texts  # 8 条描述
    image_features3, text_features3 = extract_features(
        model, preprocess, tokenizer, image_paths, mixed_texts, device="cpu"
    )
    similarity_mixed = compute_similarity_matrix(image_features3, text_features3)
    
    mixed_labels = [f"{t[:20]}✓" for t in correct_texts] + \
                   [f"{t[:20]}✗" for t in misleading_texts]
    
    print(f"\n混合描述相似度矩阵 ({similarity_mixed.shape[0]}×{similarity_mixed.shape[1]}):")
    print("         ", "  ".join(f"{l:>20s}" for l in mixed_labels))
    for i, label in enumerate(image_labels):
        row = "  ".join(f"{s:>20.4f}" for s in similarity_mixed[i])
        print(f"  {label:12s}: {row}")
    
    # 分析：每张图的最佳匹配是正确描述还是错误描述？
    print(f"\n{'─' * 70}")
    print("每张图的最佳匹配分析（正确 vs 错误）")
    print(f"{'─' * 70}")
    
    discrimination_results = []
    for i in range(4):
        row = similarity_mixed[i]
        best_idx = row.argmax()
        best_sim = row[best_idx]
        
        # 正确描述在前 4 个位置
        is_correct_best = best_idx < 4
        
        # 正确描述 vs 错误描述的最大值
        correct_max = row[:4].max()
        wrong_max = row[4:].max()
        margin = correct_max - wrong_max
        
        best_label = mixed_texts[best_idx]
        best_type = "正确" if best_idx < 4 else "错误"
        
        discrimination_results.append({
            "image": image_labels[i],
            "best_match": best_label,
            "best_type": best_type,
            "correct_max": float(correct_max),
            "wrong_max": float(wrong_max),
            "margin": float(margin),
            "discriminated": bool(is_correct_best)
        })
        
        status = "✅ 正确区分" if is_correct_best else "❌ 被误导"
        print(f"  {image_labels[i]:15s}: 最佳={best_type} ({best_sim:.4f}), "
              f"正确max={correct_max:.4f}, 错误max={wrong_max:.4f}, "
              f"margin={margin:.4f} {status}")
    
    correct_disc = sum(1 for r in discrimination_results if r["discriminated"])
    print(f"\n  {'✅' if correct_disc == 4 else '❌'} CLIP 正确区分: {correct_disc}/4")
    
    # 绘制对比热力图
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：正确描述
    im1 = axes[0].imshow(similarity, cmap="RdYlGn", vmin=-0.2, vmax=0.4)
    axes[0].set_xticks(range(4))
    axes[0].set_yticks(range(4))
    axes[0].set_xticklabels([t[:15] for t in correct_texts], fontsize=8, rotation=20, ha="right")
    axes[0].set_yticklabels(image_labels, fontsize=9)
    axes[0].set_title("正确描述配对\n(对角线应为高相似度)", fontsize=11, fontweight="bold")
    fig.colorbar(im1, ax=axes[0], fraction=0.046)
    for i in range(4):
        for j in range(4):
            axes[0].text(j, i, f"{similarity[i,j]:.2f}", ha="center", va="center", fontsize=7)
    
    # 右图：误导性描述
    im2 = axes[1].imshow(similarity_wrong, cmap="RdYlGn", vmin=-0.2, vmax=0.4)
    axes[1].set_xticks(range(4))
    axes[1].set_yticks(range(4))
    axes[1].set_xticklabels([t[:15] for t in misleading_texts], fontsize=8, rotation=20, ha="right")
    axes[1].set_yticklabels(image_labels, fontsize=9)
    axes[1].set_title("误导性描述配对\n(正确配对不在对角线)", fontsize=11, fontweight="bold")
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    for i in range(4):
        for j in range(4):
            axes[1].text(j, i, f"{similarity_wrong[i,j]:.2f}", ha="center", va="center", fontsize=7)
    
    plt.suptitle("CLIP 图文配对：正确描述 vs 误导性描述", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_comparison_correct_vs_misleading.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n已保存: {FIG_DIR / 'fig03_comparison_correct_vs_misleading.png'}")
    
    # ═══════════════════════════════════════════════════════════
    # 实验 4：温度系数实验
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("实验 4：温度系数 (temperature) 对 softmax 分布的影响")
    print(f"{'=' * 70}")
    
    # 用第一个 batch 的相似度数据
    raw_sim = similarity[0]  # 第一张图对所有文本的相似度
    
    print(f"\n第一张图 (Cat) 对 4 条描述的原始相似度:")
    for j, txt in enumerate(correct_texts):
        print(f"  {raw_sim[j]:.4f}  {txt[:40]}")
    
    print(f"\n不同温度系数下的 softmax 分布:")
    print(f"{'温度':>6s}  {'softmax 值':>50s}")
    print(f"{'─' * 70}")
    
    temperatures = [0.01, 0.05, 0.07, 0.10, 0.20, 0.50, 1.00]
    temp_results = []
    
    for temp in temperatures:
        scaled = raw_sim / temp
        exp_vals = np.exp(scaled - np.max(scaled))
        softmax_vals = exp_vals / np.sum(exp_vals)
        
        # 计算熵（衡量分布的均匀程度）
        entropy = -np.sum(softmax_vals * np.log2(softmax_vals + 1e-12))
        max_prob = softmax_vals.max()
        
        temp_results.append({
            "temperature": temp,
            "softmax_values": softmax_vals.tolist(),
            "entropy": float(entropy),
            "max_probability": float(max_prob),
        })
        
        vals_str = ", ".join(f"{v:.3f}" for v in softmax_vals)
        print(f"  {temp:>4.2f}  [{vals_str}]  (max={max_prob:.4f}, H={entropy:.3f})")
    
    # 绘制温度系数影响的可视化
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 图 1：softmax 分布随温度变化
    temps_arr = np.array(temperatures)
    max_probs = [r["max_probability"] for r in temp_results]
    entropies = [r["entropy"] for r in temp_results]
    
    axes[0].plot(temps_arr, max_probs, 'ro-', linewidth=2, markersize=8)
    axes[0].set_xlabel("温度系数 τ", fontsize=11)
    axes[0].set_ylabel("最大概率", fontsize=11)
    axes[0].set_title("最大概率 vs 温度", fontsize=11, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].annotate('τ=0.07\n(CLIP默认)', xy=(0.07, max_probs[2]),
                     xytext=(0.15, max_probs[2]+0.02),
                     arrowprops=dict(arrowstyle="->", color="blue"),
                     fontsize=9, color="blue")
    
    # 图 2：熵随温度变化
    axes[1].plot(temps_arr, entropies, 'bs-', linewidth=2, markersize=8)
    axes[1].set_xlabel("温度系数 τ", fontsize=11)
    axes[1].set_ylabel("熵 (bits)", fontsize=11)
    axes[1].set_title("熵 vs 温度", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    
    # 图 3：softmax 分布堆叠图
    all_softmax = np.array([r["softmax_values"] for r in temp_results])
    im = axes[2].imshow(all_softmax, cmap="YlOrRd", aspect="auto")
    axes[2].set_xticks(range(4))
    axes[2].set_xticklabels([f"T{j+1}" for j in range(4)], fontsize=9)
    axes[2].set_yticks(range(len(temperatures)))
    axes[2].set_yticklabels([f"τ={t:.2f}" for t in temperatures], fontsize=9)
    axes[2].set_title("softmax 分布变化\n(行=温度, 列=文本)", fontsize=11, fontweight="bold")
    fig.colorbar(im, ax=axes[2], fraction=0.046)
    
    plt.suptitle("温度系数 τ 对 softmax 分布的影响", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_temperature_effect.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已保存: {FIG_DIR / 'fig04_temperature_effect.png'}")
    
    # ═══════════════════════════════════════════════════════════
    # 保存结果到 JSON
    # ═══════════════════════════════════════════════════════════
    results_data = {
        "model": "RN50",
        "feature_dim": int(image_features.shape[1]),
        "experiment1_correct_matching": {
            "image_labels": image_labels,
            "text_labels": correct_texts,
            "similarity_matrix": similarity.tolist(),
            "results": results,
            "all_correct": all_correct,
        },
        "experiment2_misleading": {
            "misleading_texts": misleading_texts,
            "similarity_matrix": similarity_wrong.tolist(),
            "results": results_wrong,
        },
        "experiment3_discrimination": {
            "mixed_texts": mixed_texts,
            "similarity_matrix": similarity_mixed.tolist(),
            "results": discrimination_results,
            "correctly_discriminated": correct_disc,
        },
        "experiment4_temperature": {
            "temperatures": temperatures,
            "results": temp_results,
        },
    }
    
    with open(RES_DIR / "clip_experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {RES_DIR / 'clip_experiment_results.json'}")
    
    # ── 总结 ──
    print(f"\n{'=' * 70}")
    print("实验总结")
    print(f"{'=' * 70}")
    print(f"  模型: CLIP RN50")
    print(f"  特征维度: {image_features.shape[1]}")
    print(f"  实验 1 (正确配对): {'✅ 全部正确' if all_correct else '❌ 存在错误'}")
    print(f"  实验 2 (误导性描述): {'⚠️ 被误导' if not all_correct_wrong else '✅ 仍正确'}")
    print(f"  实验 3 (正确 vs 错误区分): {correct_disc}/4 正确区分")
    print(f"  实验 4 (温度系数): 已展示 τ 从 0.01 到 1.00 的 softmax 变化")
    
    # 特征形状示例
    print(f"\n  特征示例:")
    print(f"    图像特征: {image_features.shape} → 归一化后范数 = {img_norms}")
    print(f"    文本特征: {text_features.shape} → 归一化后范数 = {txt_norms}")
    print(f"    相似度矩阵: {similarity.shape} → 点积 = 余弦相似度")


if __name__ == "__main__":
    main()
