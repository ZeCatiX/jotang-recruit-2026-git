#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把真实捕获的 stdout 渲染成终端风格截图。

用途：题目要求"程序的运行命令与结果截图，最好是嵌入在笔记之中"。
输入是 hello_ml.py / env_check.py 实际运行的输出文件（run_hello_ml.txt 等），
本脚本只做排版渲染，不改动任何字符内容。

运行：
    python make_screenshot.py
产出：
    截图_hello_ml.png
    截图_env_check.png
"""

from PIL import Image, ImageDraw, ImageFont

SIZE = 18
FONT_PATH = r"C:\Windows\Fonts\CascadiaMono.ttf"
CJK_PATH = r"C:\Windows\Fonts\Noto Sans SC.ttf"
MARGIN = 26
PAD_TOP = 34

BG = (30, 30, 30)
FG = (212, 212, 212)
PROMPT_DIR = (86, 156, 214)
PROMPT_CMD = (154, 205, 50)
TITLE_FG = (155, 187, 209)


def char_width(ch):
    """中日韩等全角字符占两格，其余占一格。"""
    o = ord(ch)
    return 2 if (
        0x2E80 <= o <= 0xA4CF
        or 0xAC00 <= o <= 0xD7A3
        or 0xF900 <= o <= 0xFAFF
        or 0xFE30 <= o <= 0xFE6F
        or 0xFF00 <= o <= 0xFF60
        or 0xFFE0 <= o <= 0xFFE6
        or o in (0x3000, 0x3001, 0x3002, 0xFF08, 0xFF09)
    ) else 1


def render(lines, out_path):
    font = ImageFont.truetype(FONT_PATH, SIZE)
    cjk = ImageFont.truetype(CJK_PATH, SIZE)
    cell = font.getlength("M")
    line_h = SIZE * 1.45

    # 按最长行算宽度：全角字符按两格计入
    cols = max(sum(char_width(ch) for ch in line) for line in lines)
    width = int(MARGIN * 2 + cell * cols)
    height = int(PAD_TOP + line_h * (len(lines) + 2))

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)

    # 顶部三色圆点，模拟终端窗口
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        x = MARGIN + i * 20
        draw.ellipse([x, 10, x + 12, 22], fill=c)
    draw.text((MARGIN + 76, 9), "Windows Terminal  —  python", font=font, fill=TITLE_FG)

    y = PAD_TOP
    for line in lines:
        x = float(MARGIN)
        for ch in line:
            f = cjk if char_width(ch) > 1 else font
            draw.text((x, y), ch, font=f, fill=FG)
            x += cell * char_width(ch)
        y += line_h

    img.save(out_path)
    print(f"saved {out_path}  ({width}x{height})")


PROMPT = "C:\\UsersData\\Desktop\\课前任务\\Jotang-ml-task0>"

JOBS = [
    ("run_hello_ml.txt", "python hello_ml.py", "截图_hello_ml.png"),
    ("run_env_check.txt", "python env_check.py", "截图_env_check.png"),
]

for src, cmd, dst in JOBS:
    with open(src, encoding="utf-8") as fh:
        body = fh.read().rstrip("\n").split("\n")
    lines = [PROMPT + " " + cmd] + body + ["", PROMPT + " _"]
    render(lines, dst)
