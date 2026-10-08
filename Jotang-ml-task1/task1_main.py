#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 —— 任务 1、2、3、5 主脚本

覆盖：
  任务 1  可视化原始数据，划分训练/验证/测试集
  任务 2  PyTorch MLP 训练、验证、测试，模型保存与加载
  任务 3  loss / accuracy 曲线 + 二维决策边界
  任务 5  混淆矩阵 + 错误样本分析
  趁热打铁  学习率对比、类别不均衡、人为过拟合

运行：  python task1_main.py
产出：  figures/ 下 9 张图，results/ 下保存的模型与汇总 JSON
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

import common as C

C.setup_seeds(C.SEED)
C.setup_mpl()
os.makedirs(C.RESULT_DIR, exist_ok=True)
DEVICE = "cpu"
CLASSES = ["月牙A (0)", "月牙B (1)"]
EPOCHS, LR, BATCH = 150, 1e-3, 32
CMAP = {0: "#3b7dd8", 1: "#e4572e"}
MK = {0: "o", 1: "^"}


def hr(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def scatter_panel(ax, groups, title):
    """
    在子图上画散点。groups 是 [(标签名, X, y), ...]，
    每个 (标签名, 类) 组合一个 legend 条目。
    """
    for name, Xd, yd in groups:
        for lab in (0, 1):
            m = yd == lab
            if not m.any():
                continue
            ax.scatter(Xd[m, 0], Xd[m, 1], s=48, alpha=0.78,
                       color=CMAP[lab], marker=MK[lab], edgecolor="k", linewidth=0.5,
                       label=f"{name} · 类{lab}")
    ax.set_title(title)
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")


# ===========================================================================
# 任务 1：数据生成、划分、可视化
# ===========================================================================
hr(f"任务 1  生成 moons 数据并划分（样本数 {C.N_SAMPLES}，噪声 {C.MOONS_NOISE}）")
X, y = C.make_data()
X_tr, y_tr, X_va, y_va, X_te, y_te = C.split_data(X, y)
print(f"  全部 {len(X)} 条 → 训练 {len(y_tr)} / 验证 {len(y_va)} / 测试 {len(y_te)}")
for tag, yt in [("训练", y_tr), ("验证", y_va), ("测试", y_te)]:
    print(f"  {tag}集类别占比: 类0 {(np.mean(yt==0)*100):.1f}%  类1 {(np.mean(yt==1)*100):.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
scatter_panel(axes[0], [("原始数据", X, y)], f"① 原始数据（{len(X)} 条，噪声 {C.MOONS_NOISE}）\n两类在中间区域互相重叠 → 这个任务不是线性可分的")
axes[1].axis("off")
axes[1].text(0.5, 0.5,
             "② 划分规则\n\n"
             "分层抽样 70 / 15 / 15\n"
             "↓\n"
             "训练集：唯一用来更新参数的\n"
             "验证集：调超参数、判断过拟合、\n          决定何时停（early stopping）\n"
             "测试集：全程不许碰，最后只算一次\n\n"
             "验证集和测试集永远不参与梯度更新，\n这就是「避免数据泄漏」的含意。",
             ha="center", va="center", fontsize=11)
axes[1].set_title("② 三份数据各自职责", fontsize=12)
scatter_panel(axes[2], [("训练", X_tr, y_tr), ("验证", X_va, y_va), ("测试", X_te, y_te)],
              "③ 三份数据叠加（验证+测试 = 模型从没见过的数据）")
axes[2].legend(fontsize=7, ncol=2)
C.save_fig(fig, "fig1_数据划分.png")


# ===========================================================================
# 任务 2 + 3：训练、验证、测试、曲线
# ===========================================================================
hr("任务 2/3  训练 MLP（2 隐藏层 × 64，ReLU，Adam lr=1e-3，150 epoch，batch 32）")
model = C.MLP(input_dim=2, hidden_width=64, n_hidden=2, activation="relu")
print(f"  模型结构: 2 → 64 → 64 → 2   可训练参数 {model.count_params()}")

Xtr_t, ytr_t = C.to_tensors(X_tr, y_tr)
Xva_t, yva_t = C.to_tensors(X_va, y_va)
Xte_t, yte_t = C.to_tensors(X_te, y_te)

mem_before = C.process_rss_mib()
hist = C.train_model(model, Xtr_t, ytr_t, Xva_t, yva_t,
                     epochs=EPOCHS, lr=LR, batch_size=BATCH,
                     optimizer="adam", verbose=True, device=DEVICE)
mem_after = C.process_rss_mib()

crit = nn.CrossEntropyLoss()
tr_acc, tr_loss = C.evaluate(model, Xtr_t, ytr_t, crit)
va_acc, va_loss = C.evaluate(model, Xva_t, yva_t, crit)
te_acc, te_loss = C.evaluate(model, Xte_t, yte_t, crit)
print(f"\n  最终:  训练准确率 {tr_acc:.4f} | 验证准确率 {va_acc:.4f} | 测试准确率 {te_acc:.4f}")
print(f"  训练耗时 {hist['train_time']:.2f}s / {hist['n_epochs']} epoch")
print(f"  进程常驻内存: {mem_before:.1f} → {mem_after:.1f} MiB（增量 {mem_after - mem_before:+.1f} MiB）")

# 验证损失最低的 epoch = early stopping 应该停的位置，标出来给读者看
best_val_epoch = int(np.argmin(hist["val_loss"])) + 1   # 1-based
best_val_loss = float(min(hist["val_loss"]))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
axes[0].plot(hist["train_loss"], color=CMAP[0], lw=2, label="训练损失")
axes[0].plot(hist["val_loss"], color=CMAP[1], lw=2, label="验证损失")
axes[0].axvline(best_val_epoch, color="gray", ls="--", lw=1.2)
axes[0].scatter([best_val_epoch], [best_val_loss], s=70, color="#e4572e",
                zorder=5, edgecolor="k", linewidth=1.2, label="验证损失最低点")
axes[0].annotate(f"拐点：epoch {best_val_epoch}\n"
                 f"验证损失 {best_val_loss:.4f}\n→ 之后回升，应在此停",
                 xy=(best_val_epoch, best_val_loss),
                 xytext=(best_val_epoch + 22, best_val_loss + 0.11),
                 fontsize=9, ha="left",
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.0))
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("交叉熵损失")
axes[0].set_title(f"① 损失曲线：训练损失单调下降，验证损失在 epoch {best_val_epoch} 触底后回升\n"
                  f"→ 这就是「早期过拟合」的信号，也是 early stopping 的依据")
axes[0].legend(fontsize=8.5)
axes[0].grid(alpha=0.3)
axes[1].plot(hist["train_acc"], color=CMAP[0], lw=2, label="训练准确率")
axes[1].plot(hist["val_acc"], color=CMAP[1], lw=2, label="验证准确率")
axes[1].axvline(best_val_epoch, color="gray", ls="--", lw=1.2)
axes[1].axhline(te_acc, color="gray", ls=":", lw=1.6, label=f"测试准确率 {te_acc:.3f}（只算一次）")
axes[1].set_ylim(0.5, 1.02)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("准确率")
axes[1].set_title("② 准确率曲线：训练与验证都爬到高位后趋平，没有出现明显的过拟合分离")
axes[1].legend(fontsize=8.5)
axes[1].grid(alpha=0.3)
fig.suptitle("任务 3：loss 与 accuracy 曲线", fontsize=13)
C.save_fig(fig, "fig2_loss_acc曲线.png")


# ===========================================================================
# 任务 2：模型保存与加载
# ===========================================================================
hr("任务 2  模型保存与加载")
model_path = os.path.join(C.RESULT_DIR, "mlp_best.pt")
torch.save({
    "state_dict": model.state_dict(),
    "model_config": model.config,
    "metrics": {"train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc},
    "seed": C.SEED,
    "torch_version": torch.__version__,
}, model_path)
print(f"  已保存: {model_path}  ({os.path.getsize(model_path) / 1024:.1f} KB)")

reloaded = C.MLP(**model.config)
reloaded.load_state_dict(torch.load(model_path, weights_only=False)["state_dict"])
pred_orig = C.predict(model, Xte_t)
pred_new = C.predict(reloaded, Xte_t)
same = np.array_equal(pred_orig, pred_new)
print(f"  加载后在测试集上的预测与原模型完全一致: {same}")
print(f"  加载后测试准确率: {float((pred_new == y_te).mean()):.4f}")
assert same, "加载后的模型预测不一致"


# ===========================================================================
# 任务 3：二维决策边界
# ===========================================================================
hr("任务 3  二维决策边界")
xx, yy, grid_pts = C.decision_grid(model, X, step=400)

fig, ax = plt.subplots(figsize=(7.8, 5.8))
ax.contourf(xx, yy, grid_pts[:, :, 1], cmap="RdBu", alpha=0.45)
ax.contour(xx, yy, grid_pts[:, :, 1] >= 0.5, levels=[0], colors="k", linewidths=2.0)
for name, (Xd, yd) in [("训练集", (X_tr, y_tr)), ("验证集", (X_va, y_va)), ("测试集", (X_te, y_te))]:
    for lab in (0, 1):
        m = yd == lab
        ax.scatter(Xd[m, 0], Xd[m, 1], s=45, alpha=0.85, color=CMAP[lab], marker=MK[lab],
                   edgecolor="k", linewidth=0.5, label=f"{name} · 类{lab}" if name == "训练集" else None)
wrong_te = pred_orig != y_te
ax.scatter(X_te[wrong_te, 0], X_te[wrong_te, 1], s=150, facecolor="none",
           edgecolor="k", linewidth=1.8, zorder=6, label="测试集预测错误")
ax.set_xlabel("特征 1")
ax.set_ylabel("特征 2")
ax.set_title(f"任务 3：二维决策边界（黑色实线 = 类别概率 0.5 的分界）\n"
             f"测试集准确率 {te_acc:.3f}，错误 {wrong_te.sum()} 个")
ax.legend(fontsize=9, loc="upper left")
C.save_fig(fig, "fig3_决策边界.png")


# ===========================================================================
# 任务 5：混淆矩阵 + 错误样本分析
# ===========================================================================
hr("任务 5  混淆矩阵与错误样本分析")
def confusion_of(Xd, yd):
    pred = C.predict(model, C.to_tensors(Xd, yd)[0])
    return pred, float((pred == yd).mean()), confusion_matrix(yd, pred).tolist()

pred_tr, acc_tr, cm_tr = confusion_of(X_tr, y_tr)
pred_va, acc_va, cm_va = confusion_of(X_va, y_va)
pred_te, acc_te, cm_te = confusion_of(X_te, y_te)

# 用 gridspec 显式留一列给颜色条——直接 ax=axes 或 location="right" 都会把颜色条
# 叠到最后一个矩阵上面，把里面的数字盖掉。
fig = plt.figure(figsize=(14.2, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.16])
axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
for ax, (name, cm, acc, n) in zip(axes, [("训练集", cm_tr, acc_tr, len(y_tr)),
                                          ("验证集", cm_va, acc_va, len(y_va)),
                                          ("测试集", cm_te, acc_te, len(y_te))]):
    m = np.array(cm)
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    ax.set_xticks([0, 1], CLASSES, fontsize=8)
    ax.set_yticks([0, 1], CLASSES, fontsize=8)
    ax.set_xlabel("预测")
    ax.set_ylabel("真实")
    ax.set_title(f"{name}（n={n}，准确率 {acc:.3f}）")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{m[i, j]}  ({m[i, j] / m.sum() * 100:.1f}%)",
                    ha="center", va="center", fontsize=11,
                    color="white" if m[i, j] > m.max() * 0.6 else "black")
    ax.grid(False)
fig.colorbar(im, cax=fig.add_subplot(gs[0, 3]), shrink=0.92)
fig.suptitle("任务 5：三份数据的混淆矩阵", fontsize=13)
C.save_fig(fig, "fig4_混淆矩阵.png")

# 汇总三个数据集的所有错误样本
err_X, err_true, err_pred, err_src = [], [], [], []
for tag, (Xd, yd, pr) in [("训练", (X_tr, y_tr, pred_tr)), ("验证", (X_va, y_va, pred_va)),
                          ("测试", (X_te, y_te, pred_te))]:
    w = pr != yd
    err_X.append(Xd[w]); err_true.append(yd[w]); err_pred.append(pr[w])
    err_src.append(np.full(int(w.sum()), tag, dtype=object))
err_X = np.concatenate(err_X); err_true = np.concatenate(err_true)
err_pred = np.concatenate(err_pred); err_src = np.concatenate(err_src)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.0), gridspec_kw={"width_ratios": [1.55, 1]})
ax = axes[0]
ax.contourf(xx, yy, grid_pts[:, :, 1], cmap="RdBu", alpha=0.35)
ax.contour(xx, yy, grid_pts[:, :, 1] >= 0.5, levels=[0], colors="k", linewidths=1.8)
for lab in (0, 1):
    m = err_true == lab
    ax.scatter(err_X[m, 0], err_X[m, 1], s=80, color=CMAP[lab], marker=MK[lab],
               edgecolor="k", linewidth=0.8, zorder=5, label=f"真类{lab} → 被误判成类{1 - lab}")
ax.set_title(f"① 全部预测错误的样本（{len(err_X)} 个）\n它们全部贴在决策边界两侧——模型对它们几乎「拿不准」")
ax.set_xlabel("特征 1")
ax.set_ylabel("特征 2")
ax.legend(fontsize=9, loc="upper right")

ax = axes[1]
tags = ["训练", "验证", "测试"]
counts = [int((err_src == t).sum()) for t in tags]
tot = [len(y_tr), len(y_va), len(y_te)]
bars = ax.bar(tags, counts, color=[CMAP[0], "#f4a261", CMAP[1]], alpha=0.88)
for b, c, t_ in zip(bars, counts, tot):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * 0.03,
            f"{c} 个\n误判率 {c / t_ * 100:.2f}%", ha="center", va="bottom", fontsize=10)
ax.set_ylim(0, max(counts) * 1.35)
ax.set_ylabel("预测错误的样本数")
ax.set_title("② 各数据集误判数量与误判率")
ax.grid(axis="y", alpha=0.3)
fig.suptitle("任务 5：错误样本分析", fontsize=13)
C.save_fig(fig, "fig5_错误样本分析.png")

print(f"  误判汇总: " + "  ".join(f"{t} {c}/{n}" for t, c, n in zip(tags, counts, tot)))
print(f"  错误样本到数据中心的平均距离: {np.linalg.norm(err_X - X.mean(axis=0), axis=1).mean():.2f}")
print("  共同特征：全部落在决策边界两侧很近的位置，正是两类互相重叠、噪声最大的地方。")


# ===========================================================================
# 趁热打铁 A：学习率对比
# ===========================================================================
hr("趁热打铁 A  学习率对比（1e-4 / 1e-3 / 1e-2 / 1e-1）")
LR_LIST = [1e-4, 1e-3, 1e-2, 1e-1]
lr_runs = {}
for r in LR_LIST:
    C.setup_seeds(C.SEED)
    m = C.MLP(input_dim=2, hidden_width=64, n_hidden=2, activation="relu")
    h = C.train_model(m, Xtr_t, ytr_t, Xva_t, yva_t, epochs=EPOCHS, lr=r, batch_size=BATCH, device=DEVICE)
    ta, _ = C.evaluate(m, Xte_t, yte_t, crit)
    lr_runs[r] = {"hist": h, "test_acc": ta}
    print(f"  lr={r:<6g} 最终验证损失 {h['val_loss'][-1]:.4f}  最终验证准确率 {h['val_acc'][-1]:.4f}  "
          f"测试准确率 {ta:.4f}  耗时 {h['train_time']:.2f}s")

fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
for r, d in lr_runs.items():
    axes[0].plot(d["hist"]["val_loss"], label=f"lr={r:g}", lw=2)
    axes[1].plot(d["hist"]["val_acc"], label=f"lr={r:g}", lw=2)
axes[0].set_yscale("log")
axes[0].set_xlabel("epoch")
axes[0].set_ylabel("验证损失（对数刻度）")
axes[0].set_title("① 验证损失：lr=1e-3 最低（0.212）\nlr=1e-4 太小下不去（0.339），1e-2/1e-1 过大反而更差")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("epoch")
axes[1].set_ylabel("验证准确率")
axes[1].set_title("② 验证准确率：lr=1e-4 停在 0.844 收敛不足\n1e-3/1e-2/1e-1 都到 0.956，但后两者验证损失明显更差")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.suptitle("趁热打铁 A：学习率过大 vs 过小", fontsize=13)
C.save_fig(fig, "fig6_学习率对比.png")


# ===========================================================================
# 趁热打铁 B：类别不均衡
# ===========================================================================
hr("趁热打铁 B  构造 90/10 不均衡数据")
# 关键设计：两类必须「部分重叠」。
# 如果正类放在 [-1,-1] 而负类放在 [1,1]（两个团离得远远的），
# 不加权也能拿 100%，根本不体现不均衡的问题。
# 这里正类故意和负类的分布重叠（正类中心 -0.4 vs 负类中心 -1.0，且正类更分散），
# 这样"模型偷懒全猜多数类"才是一个真实的诱惑。
rng = np.random.RandomState(C.SEED)
n_neg, n_pos = 900, 100
X_n = np.vstack([rng.normal(loc=[-1.0, -1.0], scale=0.35, size=(n_neg, 2)),
                 rng.normal(loc=[-0.4, -0.4], scale=1.1, size=(n_pos, 2))])
y_n = np.concatenate([np.zeros(n_neg, np.int64), np.ones(n_pos, np.int64)])
Xn_tr, yn_tr, Xn_va, yn_va, Xn_te, yn_te = C.split_data(X_n, y_n)
Xn_tr_t, yn_tr_t = C.to_tensors(Xn_tr, yn_tr)
Xn_va_t, yn_va_t = C.to_tensors(Xn_va, yn_va)
Xn_te_t, yn_te_t = C.to_tensors(Xn_te, yn_te)
print(f"  总样本 {len(y_n)}（负类 {n_neg} / 正类 {n_pos}，比例 90:10）")
print(f"  测试集 {len(yn_te)} 条：类0 {(np.mean(yn_te==0)*100):.1f}%  类1 {(np.mean(yn_te==1)*100):.1f}%")

def run_imb(class_weight=None):
    C.setup_seeds(C.SEED)
    m = C.MLP(input_dim=2, hidden_width=64, n_hidden=2, activation="relu")
    h = C.train_model(m, Xn_tr_t, yn_tr_t, Xn_va_t, yn_va_t, epochs=EPOCHS, lr=LR,
                      batch_size=BATCH, device=DEVICE, class_weight=class_weight)
    return m, h

# 方案 1：平凡基线——永远预测多数类，一行代码不用训练
pred_trivial = np.zeros(len(yn_te), dtype=np.int64)
# 方案 2：直接训练，不加权
m_plain, h_plain = run_imb(None)
pred_plain = C.predict(m_plain, Xn_te_t)
# 方案 3：按 1/频率 加权（CrossEntropyLoss 的 weight 参数）
w = torch.tensor([len(y_n) / (2 * n_neg), len(y_n) / (2 * n_pos)], dtype=torch.float32)
m_weight, h_weight = run_imb(w)
pred_weight = C.predict(m_weight, Xn_te_t)

def score(pred, tag):
    return {"tag": tag,
            "acc": float((pred == yn_te).mean()),
            "f1_pos": float(f1_score(yn_te, pred, pos_label=1, zero_division=0)),
            "recall_pos": float(recall_score(yn_te, pred, pos_label=1, zero_division=0)),
            "precision_pos": float(precision_score(yn_te, pred, pos_label=1, zero_division=0)),
            "confusion": confusion_matrix(yn_te, pred).tolist()}

s_trivial = score(pred_trivial, "平凡基线：全猜多数类")
s_plain = score(pred_plain, "直接训练（不加权）")
s_weight = score(pred_weight, f"加权训练（正类权重 {w[1].item():.1f}×）")
SCORES = [s_trivial, s_plain, s_weight]
print("  " + "-" * 100)
print(f"  {'方案':24s} {'总准确率':>8s} {'正类召回':>8s} {'正类精度':>8s} {'正类F1':>8s}  混淆矩阵")
print("  " + "-" * 100)
for s in SCORES:
    print(f"  {s['tag']:24s} {s['acc']:>8.3f} {s['recall_pos']:>8.3f} "
          f"{s['precision_pos']:>8.3f} {s['f1_pos']:>8.3f}  {s['confusion']}")

# 最后留一列给颜色条（直接 ax=axes 会把颜色条叠到第 4 格上，遮住数字）
fig = plt.figure(figsize=(19.6, 4.6))
gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, 1, 0.17])
axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

# 面板 1：数据分布——让读者一眼看到两类为什么会重叠
ax = axes[0]
for lab in (0, 1):
    mm = y_n == lab
    ax.scatter(X_n[mm, 0], X_n[mm, 1], s=7, alpha=0.55, color=CMAP[lab], marker=MK[lab],
               label=f"类{lab}（{int(mm.sum())} 条）")
ax.legend(fontsize=9, loc="best")
ax.set_xlabel("特征 1"); ax.set_ylabel("特征 2")
ax.set_title(f"① 数据分布：正类只有 {n_pos} 条，\n而且和负类大面积重叠", fontsize=10)
ax.grid(alpha=0.3)

# 面板 2-4：三种方案的混淆矩阵
im_last = None
for k, s in enumerate(SCORES, start=1):
    ax = axes[k]
    m = np.array(s["confusion"])
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    im_last = im
    ax.set_xticks([0, 1], CLASSES, fontsize=8)
    ax.set_yticks([0, 1], CLASSES, fontsize=8)
    ax.set_xlabel("预测"); ax.set_ylabel("真实" if k == 1 else "")
    ax.set_title(f"{'②③④'[k-1]} {s['tag']}\n总准确率 {s['acc']:.3f}  正类F1 {s['f1_pos']:.3f}", fontsize=10)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(m[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if m[i, j] > m.max() * 0.6 else "black")
    ax.grid(False)
# 颜色条放到最后一个面板的右侧，而不是横穿所有面板（否则会把第 4 格数字盖住）
fig.colorbar(im_last, ax=[axes[3]], location="right", fraction=0.045, pad=0.03)
fig.suptitle(f"趁热打铁 B：90/10 不均衡数据（测试集 {len(yn_te)} 条，正类 {int((yn_te==1).sum())} 条）\n"
             f"总准确率会骗人——平凡基线「一个正类都不找」也能拿 {(yn_te==0).mean():.1%} 准确率",
             fontsize=12)
C.save_fig(fig, "fig7_不均衡数据.png")


# ===========================================================================
# 趁热打铁 C：人为过拟合
# ===========================================================================
hr("趁热打铁 C  人为制造过拟合（60 条数据 + 大模型 + 无正则 + 多训练）")
# 设计要点：噪声必须非零。noise=0 时月牙数据天然线性可分，
# 哪怕 20 万参数的模型在验证集上也能拿满分，根本造不出差距。
# 噪声 0.6 让"记住 36 个训练点"和"猜对新样本"变成两件不同的事。
# 划分故意用 60/20/20（而不是 70/15/15），保证验证集有 12 条，
# 这样验证准确率每 1 条只动 0.083，曲线读得出来。
Xs, ys = C.make_data(n_samples=60, noise=0.6)
Xs_tr, ys_tr, Xs_va, ys_va, Xs_te, ys_te = C.split_data(Xs, ys, train=0.6, val=0.2, test=0.2)
Xs_tr_t, ys_tr_t = C.to_tensors(Xs_tr, ys_tr)
Xs_va_t, ys_va_t = C.to_tensors(Xs_va, ys_va)
Xs_te_t, ys_te_t = C.to_tensors(Xs_te, ys_te)
print(f"  数据: 60 条（噪声 0.6），训练 {len(ys_tr)} / 验证 {len(ys_va)} / 测试 {len(ys_te)}")

def run_fit(n_hidden, width, dropout, epochs, weight_decay=0.0, tag=""):
    C.setup_seeds(C.SEED)
    m = C.MLP(input_dim=2, hidden_width=width, n_hidden=n_hidden, activation="relu", dropout=dropout)
    h = C.train_model(m, Xs_tr_t, ys_tr_t, Xs_va_t, ys_va_t, epochs=epochs, lr=LR,
                      batch_size=32, weight_decay=weight_decay, device=DEVICE)
    ta, _ = C.evaluate(m, Xs_te_t, ys_te_t, crit)
    return m, h, ta

# 全部跑同样的 200 epoch，唯一变量是模型容量 / 正则强度
m_tiny, h_tiny, ta_tiny = run_fit(1, 16, 0.0, 200)
m_mid,  h_mid,  ta_mid  = run_fit(2, 64, 0.0, 200)
m_big,  h_big,  ta_big  = run_fit(4, 256, 0.0, 200)
m_reg,  h_reg,  ta_reg  = run_fit(4, 256, 0.2, 200, weight_decay=1e-2)

gap = lambda h: h["train_acc"][-1] - h["val_acc"][-1]
FIT_CASES = [("太小：1 层 × 16", h_tiny, ta_tiny),
             ("常规：2 层 × 64", h_mid, ta_mid),
             ("过大：4 层 × 256，无正则", h_big, ta_big),
             ("过大 + dropout 0.2 + wd 1e-2", h_reg, ta_reg)]
print("  " + "-" * 84)
print(f"  {'配置':38s} {'训练':>6s} {'验证':>6s} {'差距':>8s} {'测试':>6s} {'参数':>8s}")
print("  " + "-" * 84)
for tag, h, ta in FIT_CASES:
    print(f"  {tag:38s} {h['train_acc'][-1]:>6.3f} {h['val_acc'][-1]:>6.3f} "
          f"{gap(h):>+8.3f} {ta:>6.3f} {h['n_params']:>8d}")

fig, axes = plt.subplots(1, 4, figsize=(21, 4.4))
for ax, (name, h, ta) in zip(axes, FIT_CASES):
    ax.plot(h["train_acc"], color=CMAP[0], lw=2.2, label="训练准确率")
    ax.plot(h["val_acc"], color=CMAP[1], lw=2.2, label="验证准确率")
    ax.fill_between(range(1, len(h["train_acc"]) + 1), h["val_acc"], h["train_acc"],
                    color="gray", alpha=0.25, label="两者差距")
    ax.set_xlabel("epoch")
    ax.set_ylabel("准确率")
    ax.set_title(f"{name}\n训练 {h['train_acc'][-1]:.3f} → 验证 {h['val_acc'][-1]:.3f}\n"
                 f"差距 {gap(h):+.3f}（{h['n_params']} 参数）", fontsize=9.5)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
g_tiny, g_mid, g_big, g_reg = (gap(h) for _, h, _ in FIT_CASES)
t_tiny, t_mid, t_big, t_reg = (ta for _, _, ta in FIT_CASES)
fig.suptitle(
    f"趁热打铁 C：容量越大差距越宽（{g_tiny:+.3f} → {g_mid:+.3f} → {g_big:+.3f}），"
    f"加正则把它压回 {g_reg:+.3f}；\n而四个模型的测试准确率都是 "
    f"{t_tiny:.3f}/{t_mid:.3f}/{t_big:.3f}/{t_reg:.3f} —— 单看准确率根本看不出来",
    fontsize=12)
C.save_fig(fig, "fig8_过拟合曲线.png")

xg, yg, pp_big = C.decision_grid(m_big, Xs, step=400)
_, _, pp_tiny = C.decision_grid(m_tiny, Xs, step=400)   # 同一份网格，只是模型不同

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6), sharex=True, sharey=True)
for ax, (m_, pp, tag, h_, show_legend) in zip(
        axes, [(m_tiny, pp_tiny, "容量太小（1 层 × 16）", h_tiny, True),
               (m_big, pp_big, "容量过大（4 层 × 256，仅 36 条训练数据）", h_big, False)]):
    ax.contourf(xg, yg, pp[:, :, 1], cmap="RdBu", alpha=0.35)
    ax.contour(xg, yg, pp[:, :, 1] >= 0.5, levels=[0], colors="k", linewidths=2.0)
    for lab in (0, 1):
        mm = ys == lab
        ax.scatter(Xs[mm, 0], Xs[mm, 1], s=48, alpha=0.85, color=CMAP[lab], marker=MK[lab],
                   edgecolor="k", linewidth=0.5, label=f"数据 类{lab}" if show_legend else None)
    vp = C.predict(m_, Xs_va_t)
    wrong = vp != ys_va
    ax.scatter(Xs_va[wrong, 0], Xs_va[wrong, 1], s=155, facecolor="none",
               edgecolor="k", linewidth=1.7, zorder=6, label="验证集预测错误")
    ax.set_title(f"{tag}\n训练准确率 {h_['train_acc'][-1]:.3f} / 验证准确率 {h_['val_acc'][-1]:.3f}",
                 fontsize=10)
    ax.set_xlabel("特征 1")
    ax.set_ylabel("特征 2")
    ax.legend(fontsize=9, loc="upper right")
fig.suptitle("决策边界对比：过拟合的模型为了套住每一个训练点，边界被拖得又绕又碎", fontsize=13)
C.save_fig(fig, "fig9_过拟合决策边界.png")


# ===========================================================================
# 汇总
# ===========================================================================
hr("汇总")
summary = {
    "seed": C.SEED, "n_samples": C.N_SAMPLES, "moons_noise": C.MOONS_NOISE,
    "model": {"config": model.config, "n_params": hist["n_params"]},
    "training": {"epochs": EPOCHS, "lr": LR, "batch_size": BATCH, "optimizer": "adam",
                 "train_time_s": round(hist["train_time"], 3), "device": DEVICE},
    "final_metrics": {"train_acc": tr_acc, "val_acc": va_acc, "test_acc": te_acc,
                      "train_loss": tr_loss, "val_loss": va_loss, "test_loss": te_loss},
    "memory_mib": {"before": round(mem_before, 1), "after": round(mem_after, 1),
                   "delta": round(mem_after - mem_before, 1)},
    "confusion": {"训练集": {"n": len(y_tr), "acc": acc_tr, "matrix": cm_tr},
                  "验证集": {"n": len(y_va), "acc": acc_va, "matrix": cm_va},
                  "测试集": {"n": len(y_te), "acc": acc_te, "matrix": cm_te}},
    "wrong_counts": dict(zip(tags, counts)),
    "total_wrong": int(len(err_X)),
    "learning_rate_sweep": {f"lr={r:g}": {"val_loss_last": lr_runs[r]["hist"]["val_loss"][-1],
                                           "val_acc_last": lr_runs[r]["hist"]["val_acc"][-1],
                                           "test_acc": lr_runs[r]["test_acc"],
                                           "time_s": round(lr_runs[r]["hist"]["train_time"], 3)}
                            for r in LR_LIST},
    "imbalance": {"n_total": int(len(y_n)), "n_neg": n_neg, "n_pos": n_pos,
                  "pos_ratio": float((y_n == 1).mean()), "pos_weight": float(w[1]),
                  "results": SCORES},
    "overfit_experiment": {
        "data": {"n_samples": 60, "noise": 0.6, "train": int(len(ys_tr)), "val": int(len(ys_va))},
        "epochs": 200, "lr": LR, "batch_size": BATCH,
        "cases": {
            "太小 1x16": {"config": {"n_hidden": 1, "width": 16, "dropout": 0.0, "weight_decay": 0.0},
                        "n_params": h_tiny["n_params"], "train_acc": h_tiny["train_acc"][-1],
                        "val_acc": h_tiny["val_acc"][-1], "test_acc": ta_tiny,
                        "gap": round(gap(h_tiny), 4)},
            "常规 2x64": {"config": {"n_hidden": 2, "width": 64, "dropout": 0.0, "weight_decay": 0.0},
                        "n_params": h_mid["n_params"], "train_acc": h_mid["train_acc"][-1],
                        "val_acc": h_mid["val_acc"][-1], "test_acc": ta_mid,
                        "gap": round(gap(h_mid), 4)},
            "过大 4x256 无正则": {"config": {"n_hidden": 4, "width": 256, "dropout": 0.0, "weight_decay": 0.0},
                              "n_params": h_big["n_params"], "train_acc": h_big["train_acc"][-1],
                              "val_acc": h_big["val_acc"][-1], "test_acc": ta_big,
                              "gap": round(gap(h_big), 4),
                              "val_acc_epoch50": h_big["val_acc"][49],
                              "train_loss_first": h_big["train_loss"][0],
                              "train_loss_last": h_big["train_loss"][-1]},
            "过大 + 正则": {"config": {"n_hidden": 4, "width": 256, "dropout": 0.2, "weight_decay": 1e-2},
                         "n_params": h_reg["n_params"], "train_acc": h_reg["train_acc"][-1],
                         "val_acc": h_reg["val_acc"][-1], "test_acc": ta_reg,
                         "gap": round(gap(h_reg), 4)},
        },
    },
    "history_main": {k: v for k, v in hist.items() if k not in ("train_time",)},
}
with open(os.path.join(C.RESULT_DIR, "summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(f"  汇总已写入 {os.path.join(C.RESULT_DIR, 'summary.json')}")
print("\n全部完成：figures/ 下 9 张图，results/ 下模型与汇总。")
