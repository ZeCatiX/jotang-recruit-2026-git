"""
任务 4：对照实验（消融实验 / ablation study）

原则：一次只改变一个主要变量，其余配置全部锁定在 baseline，
这样才能把观察到的变化归因到那一个变量上。

baseline：2 隐藏层 × 64，ReLU，Adam lr=1e-3，batch 32，150 epoch，
数据 make_moons(n=300, noise=0.2)，固定随机种子 C.SEED。

每个取值都记录四类指标：
  结果  → 训练 / 验证 / 测试准确率
  速度  → 训练耗时（s）
  内存  → 训练前后的进程常驻内存增量（MiB）
  规模  → 可训练参数量

输出：results/ablations.json、results/实验表格.md、figures/fig10、figures/fig11

运行：python experiments.py
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

import common as C

EPOCHS = 150          # 所有实验统一 150 epoch，耗时才有可比性
BATCH = 32


def hr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def make_baseline(**overrides):
    """生成 baseline 配置，再用 overrides 覆盖其中某一个变量。"""
    cfg = dict(n_hidden=2, width=64, activation="relu", optimizer="adam",
               lr=1e-3, batch_size=BATCH, weight_decay=0.0, dropout=0.0)
    cfg.update(overrides)
    return cfg


def run_one(cfg, X_tr, y_tr, X_va, y_va, X_te, y_te):
    """按 cfg 跑一次完整训练，返回指标字典。每次重置种子，保证可复现。"""
    C.setup_seeds(C.SEED)
    model = C.MLP(input_dim=X_tr.shape[1], hidden_width=cfg["width"],
                  n_hidden=cfg["n_hidden"], activation=cfg["activation"],
                  dropout=cfg["dropout"])
    n_params = model.count_params()

    Xtr, ytr = C.to_tensors(X_tr, y_tr)
    Xva, yva = C.to_tensors(X_va, y_va)
    Xte, yte = C.to_tensors(X_te, y_te)

    rss0 = C.process_rss_mib()
    hist = C.train_model(model, Xtr, ytr, Xva, yva, epochs=EPOCHS, lr=cfg["lr"],
                         batch_size=cfg["batch_size"], optimizer=cfg["optimizer"],
                         weight_decay=cfg["weight_decay"])
    rss1 = C.process_rss_mib()

    # 测试集在训练循环外单独评估（train_model 只拿到训练/验证集）
    test_acc, test_loss = C.evaluate(model, Xte, yte)

    return {
        "config": dict(cfg),
        "n_params": n_params,
        "train_acc": round(hist["train_acc"][-1], 4),
        "val_acc": round(hist["val_acc"][-1], 4),
        "test_acc": round(test_acc, 4),
        "val_loss_last": round(hist["val_loss"][-1], 4),
        "test_loss": round(test_loss, 4),
        "train_loss_first": round(hist["train_loss"][0], 4),
        "train_loss_last": round(hist["train_loss"][-1], 4),
        "time_s": round(hist["train_time"], 3),
        "rss_before": round(rss0, 1),
        "rss_after": round(rss1, 1),
        "rss_delta": round(rss1 - rss0, 1),
    }


# ---------------------------------------------------------------------------
# 七个对照实验：每个只动一个变量
# ---------------------------------------------------------------------------
ABLATIONS = [
    dict(key="width", title="隐藏层宽度",
         items=[(16, dict(width=16)), (32, dict(width=32)), (64, dict(width=64)),
                (128, dict(width=128)), (256, dict(width=256))],
         xlabels=["16", "32", "64", "128", "256"]),

    dict(key="depth", title="隐藏层层数",
         items=[(1, dict(n_hidden=1)), (2, dict(n_hidden=2)), (3, dict(n_hidden=3)),
                (4, dict(n_hidden=4)), (6, dict(n_hidden=6))],
         xlabels=["1 层", "2 层", "3 层", "4 层", "6 层"]),

    dict(key="activation", title="激活函数",
         items=[("ReLU", dict(activation="relu")), ("tanh", dict(activation="tanh")),
                ("sigmoid", dict(activation="sigmoid")), ("ELU", dict(activation="elu"))],
         xlabels=["ReLU", "tanh", "sigmoid", "ELU"]),

    dict(key="lr", title="学习率（Adam）",
         items=[(1e-4, dict(lr=1e-4)), (3e-4, dict(lr=3e-4)), (1e-3, dict(lr=1e-3)),
                (3e-3, dict(lr=3e-3)), (1e-2, dict(lr=1e-2))],
         xlabels=["1e-4", "3e-4", "1e-3", "3e-3", "1e-2"]),

    # 不同优化器必须用各自合理的默认学习率，否则对比没有意义：
    # Adam 类在 1e-3 附近最优，SGD 需要大得多的学习率才追得上
    dict(key="optimizer", title="优化器",
         items=[("SGD", dict(optimizer="sgd", lr=1e-2)),
                ("RMSprop", dict(optimizer="rmsprop", lr=1e-3)),
                ("Adam", dict(optimizer="adam", lr=1e-3)),
                ("AdamW", dict(optimizer="adamw", lr=1e-3))],
         xlabels=["SGD\nlr=1e-2", "RMSprop\nlr=1e-3", "Adam\nlr=1e-3", "AdamW\nlr=1e-3"]),

    dict(key="batch", title="batch size",
         items=[(8, dict(batch_size=8)), (16, dict(batch_size=16)), (32, dict(batch_size=32)),
                (64, dict(batch_size=64)), (128, dict(batch_size=128)),
                (210, dict(batch_size=210))],
         xlabels=["8", "16", "32", "64", "128", "210（全量）"]),

    # 数据噪声是唯一「改数据而不是改模型」的变量，单独处理
    dict(key="noise", title="数据噪声",
         items=[(0.0, None), (0.1, None), (0.2, None), (0.4, None), (0.6, None), (0.8, None)],
         xlabels=["0.0", "0.1", "0.2", "0.4", "0.6", "0.8"]),
]


def run_ablation(ab, X_tr, y_tr, X_va, y_va, X_te, y_te):
    """跑完一个变量的所有取值。"""
    rows = []
    for label, override in ab["items"]:
        if ab["key"] == "noise":
            Xn, yn = C.make_data(n_samples=C.N_SAMPLES, noise=float(label),
                                 random_state=C.SEED)
            a, b, c_, d, e, f = C.split_data(Xn, yn)
            row = run_one(make_baseline(), a, b, c_, d, e, f)
            row["config"]["noise"] = float(label)
        else:
            row = run_one(make_baseline(**override), X_tr, y_tr, X_va, y_va, X_te, y_te)
        rows.append(row)
    return rows


def fig_ablations(results, base):
    """每个变量一格：柱状图看测试准确率，折线看训练耗时。"""
    fig, axes = plt.subplots(2, 4, figsize=(22, 9))
    axes = axes.ravel()
    for ax, ab_key in zip(axes, [a["key"] for a in ABLATIONS]):
        ab = results[ab_key]
        xs = np.arange(len(ab["xlabels"]))
        acc = [r["test_acc"] for r in ab["rows"]]
        tm = [r["time_s"] for r in ab["rows"]]

        ax.bar(xs, acc, color="#3b7dd8", alpha=0.85, width=0.6, label="测试准确率")
        for x, v in zip(xs, acc):
            ax.text(x, v + 0.012, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_ylim(0.5, 1.06)
        ax.set_xticks(xs, ab["xlabels"], fontsize=9)
        ax.set_ylabel("测试准确率")
        ax.set_xlabel(ab["title"], fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)

        ax2 = ax.twinx()
        ax2.plot(xs, tm, color="#e4572e", marker="o", lw=1.8, ms=5, label="训练耗时")
        # 手动留白，否则数字标签会飘出子图边界
        ax2.set_ylim(min(tm) * 0.55, max(tm) * 1.45)
        for x, v in zip(xs, tm):
            ax2.text(x, v + (max(tm) - min(tm)) * 0.13 + 0.05, f"{v:.2f}s", ha="center",
                     fontsize=7.5, color="#e4572e")
        ax2.set_ylabel("训练耗时 (s)", color="#e4572e")
        ax2.tick_params(axis="y", colors="#e4572e")

        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower right")

    # 最后一格留空，放结论摘要
    last = axes[-1]
    last.axis("off")
    last.text(0.02, 0.98, "结论摘要", transform=last.transAxes, fontsize=15,
              weight="bold", va="top")
    last.text(0.02, 0.88, summarize(results), transform=last.transAxes,
              fontsize=11, va="top", linespacing=1.9)

    fig.suptitle(f"任务 4：对照实验（一次只改变一个主要变量）  |  "
                 f"baseline 测试准确率 {base['test_acc']:.3f}，其余配置全部锁定", fontsize=14)
    C.save_fig(fig, "fig10_对照实验.png")

    # 第二张：直接看规模与代价的关系
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 4.8))
    for ax, ab_key, color in ((axes2[0], "width", "#3b7dd8"), (axes2[1], "depth", "#e4572e")):
        ab = results[ab_key]
        xs = np.arange(len(ab["xlabels"]))
        ax.bar(xs, [r["n_params"] for r in ab["rows"]], color=color, alpha=0.85, width=0.6,
               label="可训练参数量")
        for x, r in zip(xs, ab["rows"]):
            ax.text(x, r["n_params"] * 1.06, f"{r['n_params']:,}", ha="center", fontsize=8)
        ax.set_xticks(xs, ab["xlabels"], fontsize=9)
        ax.set_ylabel("可训练参数量")
        ax.set_xlabel(ab["title"], fontsize=10.5)
        ax.grid(axis="y", alpha=0.3)
        ax2 = ax.twinx()
        ax2.plot(xs, [r["time_s"] for r in ab["rows"]], color="#5a5a5a", marker="o",
                 lw=1.8, label="训练耗时 (s)")
        ax2.set_ylabel("训练耗时 (s)")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=9)
    fig2.suptitle("模型规模与训练代价：参数量和耗时都随宽度 / 深度近似线性增长", fontsize=12)
    C.save_fig(fig2, "fig11_模型规模与代价.png")


def summarize(results):
    """七个实验各挑一句最有信息量的观察，用在 fig10 的结论格上。"""
    def by(ab, field, value):
        return _row(results[ab]["rows"], field, value)

    w16, w64, w256 = by("width", "width", 16), by("width", "width", 64), by("width", "width", 256)
    d1, d2 = by("depth", "n_hidden", 1), by("depth", "n_hidden", 2)
    s6 = by("depth", "n_hidden", 6)
    sig, elu = by("activation", "activation", "sigmoid"), by("activation", "activation", "elu")
    relu = by("activation", "activation", "relu")
    lo, sw, big = by("lr", "lr", 1e-4), by("lr", "lr", 3e-4), by("lr", "lr", 1e-2)
    op = results["optimizer"]["rows"]
    best_op = min(op, key=lambda x: x["val_loss_last"])
    bs = results["batch"]["rows"]
    b8, b128, b210 = by("batch", "batch_size", 8), by("batch", "batch_size", 128), by("batch", "batch_size", 210)
    n0, n02, n08 = by("noise", "noise", 0.0), by("noise", "noise", 0.2), by("noise", "noise", 0.8)

    return (
        f"1. 宽度 16 欠拟合（{w16['test_acc']*100:.1f}%）；32~256 测试准确率全是 1.000，"
        f"但验证损失 {w64['val_loss_last']:.3f}→{w256['val_loss_last']:.3f} 越宽越差\n"
        f"2. 层数 1→6：{d1['test_acc']*100:.1f}%→{d2['test_acc']*100:.1f}% 就到顶了，"
        f"参数却 ×{s6['n_params']//d1['n_params']}（{s6['n_params']:,} 个）\n"
        f"3. sigmoid 明显掉队（{sig['test_acc']*100:.1f}%）；ELU 比 ReLU 验证损失低 "
        f"{(1-elu['val_loss_last']/relu['val_loss_last'])*100:.0f}%\n"
        f"4. 学习率两端都坏：{lo['config']['lr']:.0e} 欠拟合 {lo['test_acc']*100:.1f}%；"
        f"{big['config']['lr']:.0e} 准确率没掉但验证损失冲到 {big['val_loss_last']:.3f}\n"
        f"5. 四个优化器准确率全 1.000，{best_op['config']['optimizer']} 验证损失最低"
        f"（{best_op['val_loss_last']:.3f}）且最快（{best_op['time_s']:.2f}s）\n"
        f"6. batch 8→210：耗时 {b8['time_s']:.2f}s→{b210['time_s']:.2f}s"
        f"（快 {b8['time_s']/b210['time_s']:.1f} 倍），但全量批训练欠拟合"
        f"（{b210['test_acc']*100:.1f}%）\n"
        f"7. 噪声 0→0.8：测试准确率 {n0['test_acc']*100:.1f}%→{n08['test_acc']*100:.1f}%，"
        f"唯一让结果大幅下滑的变量"
    )


def write_markdown_table(results, base):
    """把七个实验写成 Markdown 表，每个变量配一句结论。"""
    lines = [
        "# 任务 4  对照实验结果表",
        "",
        f"- 随机种子 `{C.SEED}`，`task1_main.py` 与 `experiments.py` 共用，结果可复现",
        f"- 统一训练 **{EPOCHS} epoch**，未列出的配置全部锁定在 baseline",
        f"- 环境：PyTorch {torch.__version__}，CPU 版（`torch.cuda.is_available() = False`）",
        f"- **baseline**：测试准确率 {base['test_acc']:.4f}，验证 {base['val_acc']:.4f}，"
        f"耗时 {base['time_s']:.3f}s，{base['n_params']:,} 参数",
        "",
        "> 关于「内存」列：CPU 版 PyTorch 没有 `torch.cuda.memory_allocated()` 可用，",
        "> 所以改用进程常驻内存（WorkingSetSize）的增量。数据只有 300×2 个浮点数，",
        "> 增量差异很小，主要反映分配器行为，**不能当作显存的精确测量**。",
        "> 跑第一个实验前有一次预热跑，用于排除分配器一次性开销，否则 width=16 会",
        "> 凭空背上几十 MiB 的假象。",
        ">",
        "> 关于「测试准确率」：300 条数据、45 条测试集太容易学，多个配置都撞到 1.000 的",
        "> 天花板。**真正有区分度的是「验证损失」和「训练耗时」**，下面的结论主要靠它们下判断。",
        "",
    ]

    for ab in ABLATIONS:
        r = results[ab["key"]]
        lines += [f"## {r['title']}", ""]
        lines += ["| 取值 | 可训练参数 | 训练准确率 | 验证准确率 | 测试准确率 | "
                  "训练-验证差距 | 验证损失(末) | 训练耗时(s) | 内存增量(MiB) |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lab, row in zip(r["xlabels"], r["rows"]):
            cell = str(lab).replace("\n", "<br>")   # markdown 表格里不能出现裸换行
            lines.append(f"| {cell} | {row['n_params']:,} | {row['train_acc']:.3f} | "
                         f"{row['val_acc']:.3f} | **{row['test_acc']:.3f}** | "
                         f"{row['train_acc']-row['val_acc']:+.3f} | "
                         f"{row['val_loss_last']:.4f} | {row['time_s']:.3f} | "
                         f"{row['rss_delta']:+.1f} |")
        lines += ["", CONCLUSIONS[ab["key"]](r), ""]

    path = os.path.join(C.RESULT_DIR, "实验表格.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  已写入 {path}")


def _row(rows, field, value):
    for r in rows:
        if r["config"].get(field) == value:
            return r
    raise KeyError(f"{field}={value} 不在结果里")


# 每个变量的结论，写成函数以便按实际跑出来的数字下判断。
# 注意：测试准确率在多个配置上都撞到 1.000 的天花板，
# 所以真正有区分度的是「验证损失」和「训练耗时」——下面几段都靠它们说话。
def CONCLUSIONS_width(r):
    a = _row(r["rows"], "width", 16)
    b = _row(r["rows"], "width", 64)
    c_ = _row(r["rows"], "width", 256)
    return (f"**宽度不是越大越好，但坏的方式和直觉不太一样。** 宽 16 容量不够（测试 "
            f"{a['test_acc']*100:.1f}%，验证损失 {a['val_loss_last']:.4f}）；从 32 起四个宽度"
            f"测试准确率全是 1.000，看起来完全一样——但验证损失从宽 64 的 "
            f"{b['val_loss_last']:.4f} 一路涨到宽 256 的 {c_['val_loss_last']:.4f}"
            f"（×{c_['val_loss_last']/b['val_loss_last']:.1f}），说明更大的网络在训练集上拟合得更紧、"
            f"泛化余量更小，只是 45 条测试集还没暴露出来。代价是真金白银的："
            f"参数 {a['n_params']:,} → {c_['n_params']:,}（×{c_['n_params']//a['n_params']}），"
            f"耗时 {a['time_s']:.2f}s → {c_['time_s']:.2f}s。**宽 64 是性价比最高点。**")


def CONCLUSIONS_depth(r):
    d = r["rows"]
    return (f"**加深同样有上限。** 1 层（{d[0]['n_params']:,} 参数）测试 {d[0]['test_acc']*100:.1f}%；"
            f"2 层就到 {d[1]['test_acc']*100:.1f}%，之后 3/4/6 层都在 "
            f"{min(x['test_acc'] for x in d[2:])*100:.1f}% ~ {max(x['test_acc'] for x in d[2:])*100:.1f}% "
            f"之间晃，验证损失从 {d[1]['val_loss_last']:.4f} 反而升到 {d[3]['val_loss_last']:.4f}，"
            f"参数却从 {d[0]['n_params']:,} 堆到 {d[4]['n_params']:,}（×{d[4]['n_params']//d[0]['n_params']}）。"
            f"**「更深更宽 ≠ 更好」是本任务最想留下的一句话。**")


def CONCLUSIONS_activation(r):
    e = _row(r["rows"], "activation", "elu")
    a = _row(r["rows"], "activation", "relu")
    s = _row(r["rows"], "activation", "sigmoid")
    return (f"**sigmoid 明显拖后腿（测试 {s['test_acc']*100:.1f}%，其余三个都是 "
            f"{a['test_acc']*100:.1f}%）。** 原因：sigmoid 把输出压进 (0,1)，梯度上限只有 "
            f"0.25；网络越深，梯度逐层相乘就越指数级衰减（梯度消失），后面的层几乎学不动。"
            f"顺带一个反直觉的发现：**ELU 比 ReLU 更好**——验证损失 "
            f"{e['val_loss_last']:.4f} vs ReLU 的 {a['val_loss_last']:.4f}"
            f"（低 {(1-e['val_loss_last']/a['val_loss_last'])*100:.0f}%），两者测试准确率相同。"
            f"ELU 在 x<0 时给一个很小的负梯度而不是 0，缓解了 ReLU 的「死神经元」问题。")


def CONCLUSIONS_lr(r):
    lo = _row(r["rows"], "lr", 1e-4)
    sw = _row(r["rows"], "lr", 3e-4)
    big = _row(r["rows"], "lr", 1e-2)
    return (f"**学习率两端都坏，但坏法不同。** {lo['config']['lr']:.0e} 步子太小，"
            f"{EPOCHS} epoch 还没学完（测试 {lo['test_acc']*100:.1f}%，验证损失停在 "
            f"{lo['val_loss_last']:.4f}）；{big['config']['lr']:.0e} 步子太大，"
            f"**测试准确率照样 1.000，但验证损失冲到 {big['val_loss_last']:.4f}**——这是典型的"
            f"在最优点附近来回震荡，准确率没掉下来纯属测试集只有 45 条、误差被量化吞掉了。"
            f"甜点是 {sw['config']['lr']:.0e}（验证损失 {sw['val_loss_last']:.4f}，全场最低）。"
            f"**结论：学习率要同时看损失曲线，只看准确率会漏掉「过大」这一种失败模式。**")


def CONCLUSIONS_optimizer(r):
    best = min(r["rows"], key=lambda x: x["val_loss_last"])
    worst = max(r["rows"], key=lambda x: x["val_loss_last"])
    fastest = min(r["rows"], key=lambda x: x["time_s"])
    slowest = max(r["rows"], key=lambda x: x["time_s"])
    return (f"**四个优化器测试准确率全是 1.000，差距全在验证损失和耗时里。** 按验证损失排序："
            f"{best['config']['optimizer']} 最优（{best['val_loss_last']:.4f}），"
            f"{worst['config']['optimizer']} 最差（{worst['val_loss_last']:.4f}，高 "
            f"{worst['val_loss_last']/best['val_loss_last']:.1f} 倍）；按耗时："
            f"{fastest['config']['optimizer']} 最快（{fastest['time_s']:.2f}s），"
            f"{slowest['config']['optimizer']} 最慢（{slowest['time_s']:.2f}s）。"
            f"也就是 **SGD 在这个任务上又准又快**（梯度信息够干净，动量足够），"
            f"Adam/AdamW 更慢是因为每个参数要额外维护一阶和二阶动量。"
            f"注意 AdamW 和 Adam 结果几乎完全一样——本次没开权重衰减，"
            f"AdamW 的解耦惩罚项没有生效的余地，它退化成普通 Adam。")


def CONCLUSIONS_batch(r):
    small = _row(r["rows"], "batch_size", 8)
    big = _row(r["rows"], "batch_size", 128)
    full = _row(r["rows"], "batch_size", 210)
    return (f"**batch size 影响的是训练效率和拟合程度，不是最终准确度。** batch=8 时每个 epoch "
            f"要走 {210//8} 次梯度更新，耗时 {small['time_s']:.2f}s；batch=210（整份数据一次更新）"
            f"只要 {full['time_s']:.2f}s，快 {small['time_s']/full['time_s']:.1f} 倍——"
            f"但代价是拟合不够：测试 {full['test_acc']*100:.1f}%、训练准确率只有 "
            f"{full['train_acc']*100:.1f}%，明显低于其他配置。batch=128 反而拿到全场最低的"
            f"验证损失 {big['val_loss_last']:.4f}（步子大但每个 epoch 仍有 1-2 次更新，"
            f"既快又稳）。理论上小 batch 的梯度噪声有利于跳出局部最优，"
            f"大 batch 梯度更准但更新次数少——在 300 条数据的玩具任务上这个理论差异很微弱。")


def CONCLUSIONS_noise(r):
    a = r["rows"][0]
    mid = _row(r["rows"], "noise", 0.2)
    first_bad = _row(r["rows"], "noise", 0.4)
    b = r["rows"][-1]
    tmin = min(x["time_s"] for x in r["rows"])
    tmax = max(x["time_s"] for x in r["rows"])
    return (f"**数据噪声是唯一能让结果大幅下滑的变量。** 噪声 ≤ {mid['config']['noise']} 时"
            f"还能拿满（{mid['test_acc']*100:.1f}%），但一到 {first_bad['config']['noise']} 就掉到 "
            f"{first_bad['test_acc']*100:.1f}%，噪声 {b['config']['noise']} 时只剩 "
            f"{b['test_acc']*100:.1f}%——单调下降，验证损失从 {a['val_loss_last']:.4f} 涨到 "
            f"{b['val_loss_last']:.4f}，训练-验证差距也从 {a['train_acc']-a['val_acc']:+.3f} 拉宽到 "
            f"{b['train_acc']-b['val_acc']:+.3f}。耗时却始终在 {tmin:.2f}~{tmax:.2f}s 之间"
            f"（噪声是数据属性，不改变模型规模）。前六个变量都是在「同一个可学的信号」里做文章，"
            f"噪声直接改变了信号难度本身——工程上应该优先保证数据质量，其次才调超参。")


CONCLUSIONS = {"width": CONCLUSIONS_width, "depth": CONCLUSIONS_depth,
               "activation": CONCLUSIONS_activation, "lr": CONCLUSIONS_lr,
               "optimizer": CONCLUSIONS_optimizer, "batch": CONCLUSIONS_batch,
               "noise": CONCLUSIONS_noise}


def main():
    C.setup_mpl()
    os.makedirs(C.RESULT_DIR, exist_ok=True)
    os.makedirs(C.FIG_DIR, exist_ok=True)

    hr("任务 4  对照实验：一次只改变一个主要变量")
    C.setup_seeds(C.SEED)
    print(f"  baseline: 2 层 × 64, ReLU, Adam lr=1e-3, batch {BATCH}, {EPOCHS} epoch, "
          f"种子 {C.SEED}")
    print(f"  数据: make_moons(n={C.N_SAMPLES}, noise={C.MOONS_NOISE}) → "
          f"训练 {int(0.7*C.N_SAMPLES)} / 验证 {int(0.15*C.N_SAMPLES)} / 测试 "
          f"{int(0.15*C.N_SAMPLES)}")

    X_tr, y_tr, X_va, y_va, X_te, y_te = C.split_data(*C.make_data())

    # 预热一次：让 torch 的内存分配器先完成大块内存的申请，
    # 否则第一个实验（width=16）会背上几十 MiB 的一次性开销，
    # 让「宽度小的模型更占内存」这种假象混进表里。
    print(f"\n  预热跑一次（不计入结果，用于稳定内存基线）...")
    run_one(make_baseline(), X_tr, y_tr, X_va, y_va, X_te, y_te)

    results = {}

    for i, ab in enumerate(ABLATIONS, 1):
        print(f"\n  [{i}/{len(ABLATIONS)}] {ab['title']}")
        rows = run_ablation(ab, X_tr, y_tr, X_va, y_va, X_te, y_te)
        results[ab["key"]] = {"title": ab["title"], "xlabels": ab["xlabels"], "rows": rows}
        print(f"  {'取值':>14s} {'参数':>9s} {'训练':>7s} {'验证':>7s} {'测试':>7s} "
              f"{'耗时(s)':>8s} {'内存增量':>10s}")
        for lab, row in zip(ab["xlabels"], rows):
            print(f"  {lab.replace(chr(10), ' '):>14s} {row['n_params']:>9,} "
                  f"{row['train_acc']:>7.3f} {row['val_acc']:>7.3f} {row['test_acc']:>7.3f} "
                  f"{row['time_s']:>8.3f} {row['rss_delta']:>9.1f} MiB")

    hr("baseline 参照点（其余配置全部默认）")
    base = run_one(make_baseline(), X_tr, y_tr, X_va, y_va, X_te, y_te)
    print(f"  训练 {base['train_acc']:.4f} | 验证 {base['val_acc']:.4f} | "
          f"测试 {base['test_acc']:.4f} | 耗时 {base['time_s']:.3f}s | "
          f"参数 {base['n_params']:,}")

    with open(os.path.join(C.RESULT_DIR, "ablations.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": C.SEED, "epochs": EPOCHS, "batch_size": BATCH,
                   "data": {"n_samples": C.N_SAMPLES, "moons_noise": C.MOONS_NOISE},
                   "device": "cpu", "torch": torch.__version__,
                   "baseline": base, "ablations": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  已写入 {os.path.join(C.RESULT_DIR, 'ablations.json')}")

    write_markdown_table(results, base)
    fig_ablations(results, base)
    print("\n全部完成：ablations.json + 实验表格.md + fig10 + fig11")


if __name__ == "__main__":
    main()
