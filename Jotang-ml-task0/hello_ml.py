#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
招新 Task 0 —— 机器学习入门：Python 热身（题目要求 2）

包含四个部分：
  1. 用列表 / 字典记录姓名与成绩
  2. 写函数计算平均分、找出最高分
  3. 用 NumPy 创建两个形状合适的矩阵并完成矩阵乘法
  4. 打印计算结果，以及输入 / 输出矩阵的形状

矩阵乘法的例子刻意写成"神经网络全连接层"的形状：
    特征向量 (n, 3) @ 权重 (3, 2) = 输出 (n, 2)
对应笔记里"如何用矩阵乘法表示全连接层前向传播"那一部分。

直接运行：
    python hello_ml.py
"""

from statistics import mean

import numpy as np
import torch


def hr(title):
    """打印一行小标题，让输出分区更清楚。"""
    print()
    print("=" * 64)
    print(title)
    print("=" * 64)


# --------------------------------------------------------------------------
# 1. 用字典记录姓名与成绩（列表存分数，字典做 姓名 -> 分数 的映射）
# --------------------------------------------------------------------------
scores = [92, 78, 85, 96, 71, 88]
students = {"张伟": 92, "李娜": 78, "王强": 85, "赵敏": 96, "刘洋": 71, "陈思": 88}


# --------------------------------------------------------------------------
# 2. 计算平均分 + 找最高分
# --------------------------------------------------------------------------
def mean_of(nums):
    """返回一组分数的平均值，保留两位小数。"""
    return round(sum(nums) / len(nums), 2)


def top_of(mapping):
    """返回分数最高的 (姓名, 分数)。"""
    return max(mapping.items(), key=lambda item: item[1])


# --------------------------------------------------------------------------
# 3. NumPy 矩阵乘法
# --------------------------------------------------------------------------
# X: 3 个样本 x 4 个特征 —— 相当于神经网络里一个 batch 的输入
X = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [2.0, 1.0, 0.0, 1.0],
        [3.0, -1.0, 1.0, 0.0],
    ]
)

# W: 4 个输入特征 -> 2 个输出特征 —— 相当于全连接层的权重矩阵（不含偏置）
W = np.array(
    [
        [0.5, -1.0],
        [0.0, 2.0],
        [-1.0, 0.5],
        [1.5, -0.5],
    ]
)

# 全连接层前向传播：Y = X @ W
Y = X @ W

# 说明：下面第 4 部分会故意让两个形状不匹配的矩阵相乘，用来演示
# 神经网络里"上一层输出特征数必须等于下一层权重的输入特征数"这条约束。


hr("1/4. 姓名与成绩")
print("列表（只看分数） :", scores)
print("字典（姓名->分数）:", students)

hr("2/4. 平均分与最高分")
avg = mean_of(scores)
top_name, top_score = top_of(students)
print(f"平均分: {avg}")
print(f"最高分: {top_name}  {top_score} 分")
print(f"最低分: {min(students.values())}   "
      f"分数标准差: {round(float(np.std(scores)), 2)}")

hr("3/4. NumPy 矩阵乘法（形状 m×n 与 n×p 才能相乘）")
print(f"X.shape = {X.shape}   内容:\n{X}")
print(f"W.shape = {W.shape}   内容:\n{W}")

hr("4/4. 乘积与输出形状")
print(f"Y = X @ W,  Y.shape = {Y.shape}")
print(Y)
print(f"\n输出矩阵 Y 的形状: {Y.shape}  "
      f"(等于 X 的行数 {X.shape[0]} × W 的列数 {W.shape[1]})")

print("\n形状对不上时的报错（前向传播里的维度约束）:")
try:
    _ = np.zeros((3, 4)) @ np.zeros((5, 2))   # 中间维 4 != 5
except ValueError as exc:
    print("ValueError:", exc)

hr("顺手用 PyTorch 演示一遍同样的事")
print(f"torch 版本: {torch.__version__}")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"本次运行设备: {device}")

# 显式指定 dtype=torch.float32：numpy 的默认 dtype 是 float64，
# 直接 torch.tensor(X) 会得到 float64 张量，两边比 allclose 时会因为类型不同而报错。
tx = torch.tensor(X, dtype=torch.float32)
tw = torch.tensor(W, dtype=torch.float32)
ty = torch.matmul(tx, tw)
print(f"torch 矩阵乘法 ty.shape = {tuple(ty.shape)}, dtype = {ty.dtype}")
print(ty)
print(f"与 NumPy 结果是否一致: {torch.allclose(ty, torch.tensor(Y, dtype=torch.float32))}")
