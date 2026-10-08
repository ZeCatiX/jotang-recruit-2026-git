#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 1 —— 简单神经网络：共享模块

提供：随机种子、数据生成与划分、MLP 模型定义、优化器构造、
训练循环、评估函数、绘图与内存度量工具。

被 task1_main.py 和 experiments.py 共同引用。
"""

import copy
import os
import platform
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib import font_manager
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

# ---------------------------------------------------------------------------
# 全局常量：种子固定，保证学长能复现
# ---------------------------------------------------------------------------
SEED = 0
N_SAMPLES = 300          # 主实验样本数
MOONS_NOISE = 0.2        # make_moons 的噪声强度
FIG_DIR = "figures"
RESULT_DIR = "results"

_ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "elu": nn.ELU}


# ---------------------------------------------------------------------------
# 随机种子
# ---------------------------------------------------------------------------
def setup_seeds(seed=SEED):
    """固定 Python / NumPy / PyTorch 的随机源，保证实验可复现。"""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "manual_seed_all"):      # 老版本 PyTorch 还有这个接口
        torch.manual_seed_all(seed)
    # GPU 上才有意义；本机是 CPU 版 PyTorch，这两行只是为完整性保留
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


# ---------------------------------------------------------------------------
# 绘图基础设置（中文字体）
# ---------------------------------------------------------------------------
def setup_mpl():
    """注册中文字体，避免图上的中文变成方块。"""
    matplotlib.use("Agg")
    path = r"C:\Windows\Fonts\Noto Sans SC.ttf" if os.path.exists(r"C:\Windows\Fonts\Noto Sans SC.ttf") else None
    if path:
        font_manager.fontManager.addfont(path)
        plt.rcParams["font.family"] = "Noto Sans SC"
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 110
    plt.rcParams["figure.autolayout"] = True


def save_fig(fig, name):
    """保存图片到 figures/ 并关闭 figure。"""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  已保存图片: {path}")
    return path


# ---------------------------------------------------------------------------
# 数据
# ---------------------------------------------------------------------------
def make_data(n_samples=N_SAMPLES, noise=MOONS_NOISE, random_state=SEED):
    """生成 moons 二分类数据。返回 (X, y)，X 形状 (n, 2)。"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64)


def split_data(X, y, train=0.7, val=0.15, test=0.15, stratified=True):
    """
    按 70/15/15 分层划分为训练/验证/测试集。

    stratified=True 时保证三份数据里的类别比例一致——分类任务必须这么做，
    否则可能出现"训练集全是一类"这种让实验失效的切分。
    """
    parts = [train, val, test]
    X_tr, X_rest, y_tr, y_rest = train_test_split(
        X, y, test_size=(val + test), stratify=y if stratified else None, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_rest, y_rest, test_size=test / (val + test), stratify=y_rest if stratified else None, random_state=SEED
    )
    return X_tr, y_tr, X_va, y_va, X_te, y_te


def to_tensors(X, y):
    """numpy 数组转 torch 张量，float32 / int64。"""
    return torch.from_numpy(np.asarray(X, dtype=np.float32)), torch.from_numpy(np.asarray(y, dtype=np.int64))


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """
    多层感知机：n_hidden 个隐藏层，每层 hidden_width 个神经元，后接输出层。

    结构： Linear-ReLU[-Dropout] × n_hidden → Linear(output_dim)
    输出层不加激活函数，交给 CrossEntropyLoss 内部处理（它要求喂 logits）。
    """

    def __init__(self, input_dim=2, hidden_width=64, n_hidden=2, activation="relu",
                 dropout=0.0, output_dim=2):
        super().__init__()
        act_cls = _ACTIVATIONS[activation.lower()]
        layers, prev = [], input_dim
        for _ in range(n_hidden):
            layers += [nn.Linear(prev, hidden_width), act_cls()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden_width
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
        self.config = dict(input_dim=input_dim, hidden_width=hidden_width, n_hidden=n_hidden,
                           activation=activation.lower(), dropout=dropout, output_dim=output_dim)

    def forward(self, x):
        return self.net(x)

    def count_params(self):
        """可训练参数个数——显存/内存占用的主要决定因素。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def make_optim(model, name="adam", lr=1e-3, momentum=0.9, weight_decay=0.0):
    n = name.lower()
    if n == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if n == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if n == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"未知优化器: {name}")


# ---------------------------------------------------------------------------
# 评估
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, X, y, crit=None):
    """在给定数据集上算准确率和平均损失。返回 (acc, loss)。"""
    model.eval()
    out = model(X)
    loss = crit(out, y).item() if crit is not None else 0.0
    acc = (out.argmax(dim=1) == y).float().mean().item()
    return acc, loss


def predict(model, X):
    """返回预测标签。"""
    model.eval()
    with torch.no_grad():
        return model(X).argmax(dim=1).numpy()


def decision_grid(model, X, step=400, pad=0.35):
    """
    在特征平面上密集采样，返回 (xx, yy, probs)。

    probs 形状 (step, step, C)，是模型在网格点上的类别概率，
    用来画二维决策边界（probs[:, :, 1] >= 0.5 就是分界线）。
    """
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, step),
                         np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, step))
    pts = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()].astype(np.float32))
    with torch.no_grad():
        probs = torch.softmax(model(pts), dim=1).numpy().reshape(xx.shape + (-1,))
    return xx, yy, probs


# ---------------------------------------------------------------------------
# 训练循环
# ---------------------------------------------------------------------------
def train_model(model, X_tr, y_tr, X_va, y_va,
                epochs=150, lr=1e-3, batch_size=32, optimizer="adam",
                weight_decay=0.0, patience=None, verbose=False, device="cpu",
                class_weight=None):
    """
    标准训练循环：小 batch 随机打乱 → zero_grad → forward → backward → step。

    训练集准确率按"每个 epoch 结束后在完整训练集上前向一次"来算（而不是各
    batch 前向结果取平均），这样得到的是真实的全集准确率，代价是每个 epoch
    多一次前向，数据量小的时候可以忽略。

    class_weight 传给 CrossEntropyLoss，用于类别不均衡的加权训练；为 None 时不加权。

    返回 history 字典，包含 train_loss / val_loss / train_acc / val_acc / lr，
    以及 train_time（秒）和 n_params。
    """
    crit = nn.CrossEntropyLoss(weight=class_weight)
    opt = make_optim(model, optimizer, lr=lr, weight_decay=weight_decay)
    model.to(device)
    X_tr, y_tr, X_va, y_va = X_tr.to(device), y_tr.to(device), X_va.to(device), y_va.to(device)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": []}
    n = X_tr.shape[0]
    best_val, best_state, bad = -1.0, None, 0

    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

        history["train_loss"].append(evaluate(model, X_tr, y_tr, crit)[1])
        history["train_acc"].append(evaluate(model, X_tr, y_tr, crit)[0])
        history["val_loss"].append(evaluate(model, X_va, y_va, crit)[1])
        history["val_acc"].append(evaluate(model, X_va, y_va, crit)[0])
        history["lr"].append(opt.param_groups[0]["lr"])

        if verbose and (ep % 25 == 0 or ep == 1):
            print(f"    epoch {ep:4d} | 训练损失 {history['train_loss'][-1]:.4f} "
                  f"训练准确率 {history['train_acc'][-1]:.3f} | "
                  f"验证损失 {history['val_loss'][-1]:.4f} 验证准确率 {history['val_acc'][-1]:.3f}")

        if patience:
            if history["val_acc"][-1] > best_val:
                best_val = history["val_acc"][-1]
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    print(f"    提前停止于 epoch {ep}（最佳验证准确率 {best_val:.3f}）")
                    model.load_state_dict(best_state)
                    break

    history["train_time"] = time.perf_counter() - t0
    history["n_params"] = model.count_params()
    history["n_epochs"] = len(history["train_loss"])
    model.to("cpu")
    return history


# ---------------------------------------------------------------------------
# 内存度量
# ---------------------------------------------------------------------------
def process_rss_mib():
    """
    返回当前进程的常驻内存 (MiB)。

    本机是 CPU 版 PyTorch，torch.cuda.memory_allocated() 不可用，
    所以用进程的 WorkingSetSize 作为显存/内存占用的替代指标。
    纯 Windows 实现，不依赖 psutil。
    """
    if platform.system() != "Windows":
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    import ctypes
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t)]
    cnt = PROCESS_MEMORY_COUNTERS()
    cnt.cb = ctypes.sizeof(cnt)
    # 必须显式声明 restype/argtypes：GetCurrentProcess 返回 64 位句柄，
    # 不声明的话 ctypes 按 c_int 截断，后面的调用会静默失败。
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    handle = k32.GetCurrentProcess()
    psapi = ctypes.windll.psapi
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p,
                                           ctypes.POINTER(PROCESS_MEMORY_COUNTERS),
                                           ctypes.c_ulong]
    if psapi.GetProcessMemoryInfo(handle, ctypes.byref(cnt), cnt.cb):
        return cnt.WorkingSetSize / (1024.0 * 1024.0)
    return float("nan")
