#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
招新 Task 0 —— 任务 3：开发环境自检

按要求验证三件事：
  1. 输出 Python 与 PyTorch 的版本；
  2. 完成一次 PyTorch 张量运算；
  3. 检查 torch.cuda.is_available()。

直接运行：
    python env_check.py
"""

import os
import platform
import subprocess
import sys

import numpy as np
import torch


def section(title):
    print()
    print("=" * 66)
    print(title)
    print("=" * 66)


section("1. 版本信息（题目要求：输出 Python 与 PyTorch 版本）")
print("Python 版本 :", sys.version.split()[0])
print("可执行文件  :", sys.executable)
print("操作系统    :", platform.platform())
print()
print("PyTorch 版本:", torch.__version__)
print("numpy 版本  :", np.__version__)
print("编译所用 CUDA 版本  :", torch.version.cuda, "(None 表示 CPU 版 wheel)")

# 当前处于哪个 conda 环境
env_name = os.environ.get("CONDA_DEFAULT_ENV") or os.environ.get("CONDA_PREFIX") or "（非 conda 环境）"
print("当前 conda 环境:", env_name)

section("2. 一次 PyTorch 张量运算")
# 造一个 2x3 的张量，做加法、逐元素乘法、求和，再把它当成全连接层输出算一次平均
x = torch.tensor([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
w = torch.tensor([[0.5, 1.0],
                  [-1.0, 0.0],
                  [0.5, 2.0]])

out = torch.matmul(x, w)      # (2,3) @ (3,2) -> (2,2)
print("x =", x.tolist())
print("w =", w.tolist())
print("torch.matmul(x, w) =")
print(out)
print("形状         :", tuple(out.shape))
print("逐元素平方和 :", float(torch.sum(out ** 2)))
print("按行求和     :", out.sum(dim=1).tolist())

# 和 NumPy 对照，确认两种张量实现结果一致
# 踩坑：NumPy 默认 dtype 是 float64，而 torch.tensor(...) 默认是 float32。
# 直接 torch.from_numpy(np_out) 会得到 Double 张量，allclose 会报
# "Float did not match Double"，需要先 .astype(np.float32)。
np_out = (np.array(x.tolist()) @ np.array(w.tolist())).astype(np.float32)
print("与 NumPy 结果一致:", bool(torch.allclose(out, torch.from_numpy(np_out))))

# 顺手演示一次自动求导：Task 1 之后的训练流程第一步就是 .backward()
a = torch.tensor(3.0, requires_grad=True)
loss = (a * a - 6 * a + 5)
loss.backward()
# 用 .detach().item() 而不是 float(loss)：对带 requires_grad 的张量直接取 float
# 会触发 UserWarning: Converting a tensor with requires_grad=True to a scalar...
print(f"\n自动求导：L(a) = a² - 6a + 5，a = 3.0")
print(f"  损失值      loss = {loss.detach().item()}")
print(f"  梯度 dL/da   = {a.grad.item()}    （解析解 2a - 6 = 0，一致）")

section("3. torch.cuda.is_available()")
available = torch.cuda.is_available()
print("torch.cuda.is_available() =", available)
if available:
    print("显卡名称:", torch.cuda.get_device_name(0))
    print("显存总量 :", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1), "GiB")
    t = torch.ones(2, 2, device="cuda") @ torch.eye(2, device="cuda")
    print("GPU 上的一次矩阵乘法:", t.tolist())
else:
    print()
    print("→ 当前 PyTorch 是 CPU 版（torch.__version__ 以 +cpu 结尾，torch.version.cuda 为 None），")
    print("  所以这里一定是 False。这与本机有没有显卡无关。")
    print("→ 本机实际有 RTX 5060，nvidia-smi 能正常识别（驱动 592.01 / CUDA 13.1），")
    print("  但当前 conda 环境里装的是 pip 默认的 CPU wheel，没有链接 CUDA。")
    print()
    print("换成 GPU 版只需在环境里装带 CUDA 后缀的 wheel，例如：")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128")
    print("（具体用哪个 cuXXX，去 pytorch.org 的 Get Started 页面按显卡和 Python 版本查）")

    # 用 nvidia-smi 佐证本机确实有可用的 NVIDIA 显卡
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if smi.returncode == 0:
            print("\nnvidia-smi 输出（证明显卡和驱动都在，只是 PyTorch 没接上）：")
            for line in smi.stdout.strip().splitlines():
                print("  ", line)
    except Exception as exc:
        print("\nnvidia-smi 未能执行:", exc)

print()
print("环境自检完成。")
