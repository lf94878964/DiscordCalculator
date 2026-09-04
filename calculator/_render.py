# -*- coding: utf-8 -*-
"""
_render.py — 把 SymPy 算式 / 結果渲染成 PNG 圖片，供 Discord 訊息附圖使用。

視覺風格：
    - 黑底白字（搭配 Discord 深色主題，透明背景在深色容器上黑字幾乎看不到）
    - 字型改用 matplotlib mathtext 的 "cm"（Computer Modern）字型集，
      這是真正 LaTeX 預設會用的字型，斜體變數、正體數字，效果跟一般
      線上 LaTeX 產生器（例如 quicklatex）的排版一致，而不是預設的
      DejaVu Sans（那個看起來就是普通字體，不像 LaTeX）。

限制說明：
    matplotlib 的 mathtext（非 usetex 模式）只支援「一般數學排版」的子集，
    並不支援 \\begin{matrix} 這類 LaTeX 環境。因為容器環境通常沒有安裝
    完整的 TeX 發行版（texlive），這裡選擇不依賴 usetex=True，而是：
      - 純量結果：直接用 sympy.latex() 產生的 LaTeX 交給 mathtext 畫（
        支援 \\frac、\\sqrt、上下標、希臘字母等常見語法，足以應付本計算機
        會用到的函數）
      - 向量 / 矩陣結果：改成「表格排版＋手繪中括號」的方式呈現每個元素
        （每個元素本身仍然是用 mathtext 個別渲染），視覺上等同矩陣。

等式（lhs = rhs）呈現方式：
    因為 lhs / rhs 可能是「純量 + 純量」「矩陣 + 矩陣」甚至以後可能混合，
    與其硬把兩者塞進同一個 mathtext 字串（矩陣做不到），這裡改用「先各自
    渲染成獨立 PNG（黑底白字），再用 Pillow 水平拼接、中間插入等號」的
    方式組合，這樣純量／矩陣都能共用同一套等式排版邏輯，也能維持同一張
    圖是「無縫的黑色背景」。
"""

from __future__ import annotations

import io
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sympy import Matrix, MatrixBase, latex as sympy_latex
matplotlib.rcParams["mathtext.fontset"] = "cm"
matplotlib.rcParams["font.family"] = "serif"

_BG_COLOR = "black"
_FG_COLOR = "white"


def render_scalar_image(expr: Any, fontsize: int = 26) -> io.BytesIO:
    """把純量 / 一般算式渲染成 PNG（黑底白字，回傳 BytesIO，游標已歸零）。"""
    try:
        tex = sympy_latex(expr)
    except Exception:
        tex = str(expr)

    fig = plt.figure(figsize=(0.1, 0.1), facecolor=_BG_COLOR)
    try:
        fig.text(0.5, 0.5, f"${tex}$", fontsize=fontsize, ha="center", va="center",
                  color=_FG_COLOR)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                    pad_inches=0.3, facecolor=_BG_COLOR, transparent=False)
    except Exception:
        plt.close(fig)
        fig = plt.figure(figsize=(0.1, 0.1), facecolor=_BG_COLOR)
        fig.text(0.5, 0.5, str(expr), fontsize=fontsize, ha="center", va="center",
                  color=_FG_COLOR)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                    pad_inches=0.3, facecolor=_BG_COLOR, transparent=False)
    finally:
        plt.close(fig)
    buf.seek(0)
    return buf


def render_matrix_image(mat: MatrixBase, fontsize: int = 22) -> io.BytesIO:
    rows, cols = mat.shape
    cell_w, cell_h = 1.6, 0.9
    pad = 0.5
    fig_w = cols * cell_w + pad * 2
    fig_h = rows * cell_h + pad * 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.axis("off")
    ax.set_xlim(0, cols * cell_w + pad * 2)
    ax.set_ylim(0, rows * cell_h + pad * 2)

    for i in range(rows):
        for j in range(cols):
            try:
                tex = sympy_latex(mat[i, j])
                content = f"${tex}$"
            except Exception:
                content = str(mat[i, j])
            x = pad + j * cell_w + cell_w / 2
            y = pad + (rows - 1 - i) * cell_h + cell_h / 2
            ax.text(x, y, content, ha="center", va="center", fontsize=fontsize,
                     color=_FG_COLOR)

    top = pad + rows * cell_h
    bottom = pad
    left = pad * 0.5
    right = cols * cell_w + pad * 1.5
    tick = 0.15

    ax.plot([left, left], [bottom, top], color=_FG_COLOR, linewidth=2.2)
    ax.plot([left, left + tick], [bottom, bottom], color=_FG_COLOR, linewidth=2.2)
    ax.plot([left, left + tick], [top, top], color=_FG_COLOR, linewidth=2.2)

    ax.plot([right, right], [bottom, top], color=_FG_COLOR, linewidth=2.2)
    ax.plot([right - tick, right], [bottom, bottom], color=_FG_COLOR, linewidth=2.2)
    ax.plot([right - tick, right], [top, top], color=_FG_COLOR, linewidth=2.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.2, facecolor=_BG_COLOR, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def render_matrix_text_image(cells, fontsize: int = 22) -> io.BytesIO:
    rows = len(cells)
    cols = len(cells[0]) if rows else 0
    cell_w, cell_h = 2.0, 0.9
    pad = 0.5
    fig_w = max(cols, 1) * cell_w + pad * 2
    fig_h = max(rows, 1) * cell_h + pad * 2

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)
    ax.axis("off")
    ax.set_xlim(0, cols * cell_w + pad * 2)
    ax.set_ylim(0, rows * cell_h + pad * 2)

    for i in range(rows):
        for j in range(cols):
            x = pad + j * cell_w + cell_w / 2
            y = pad + (rows - 1 - i) * cell_h + cell_h / 2
            ax.text(x, y, str(cells[i][j]), ha="center", va="center",
                     fontsize=fontsize, color=_FG_COLOR, family="serif")

    top = pad + rows * cell_h
    bottom = pad
    left = pad * 0.5
    right = cols * cell_w + pad * 1.5
    tick = 0.15

    ax.plot([left, left], [bottom, top], color=_FG_COLOR, linewidth=2.2)
    ax.plot([left, left + tick], [bottom, bottom], color=_FG_COLOR, linewidth=2.2)
    ax.plot([left, left + tick], [top, top], color=_FG_COLOR, linewidth=2.2)

    ax.plot([right, right], [bottom, top], color=_FG_COLOR, linewidth=2.2)
    ax.plot([right - tick, right], [bottom, bottom], color=_FG_COLOR, linewidth=2.2)
    ax.plot([right - tick, right], [top, top], color=_FG_COLOR, linewidth=2.2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.2, facecolor=_BG_COLOR, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def render_expr_image(expr: Any) -> io.BytesIO:
    if isinstance(expr, str):
        return render_literal_text_image(expr)
    if isinstance(expr, MatrixBase):
        return render_matrix_image(expr)
    if isinstance(expr, list):
        return render_matrix_text_image(expr)
    return render_scalar_image(expr)



def _render_equals_sign(fontsize: int = 26) -> Image.Image:
    fig = plt.figure(figsize=(0.1, 0.1), facecolor=_BG_COLOR)
    fig.text(0.5, 0.5, "$=$", fontsize=fontsize, ha="center", va="center",
              color=_FG_COLOR)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.15, facecolor=_BG_COLOR, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def render_literal_text_image(text: str, fontsize: int = 26) -> io.BytesIO:
    fig = plt.figure(figsize=(0.1, 0.1), facecolor=_BG_COLOR)
    fig.text(0.5, 0.5, text, fontsize=fontsize, ha="center", va="center",
              color=_FG_COLOR, family="serif", style="italic")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=220, bbox_inches="tight",
                pad_inches=0.3, facecolor=_BG_COLOR, transparent=False)
    plt.close(fig)
    buf.seek(0)
    return buf


def _to_image(value: Any) -> Image.Image:
    if isinstance(value, str):
        buf = render_literal_text_image(value)
    else:
        buf = render_expr_image(value)
    return Image.open(buf).convert("RGB")


def render_equation_image(lhs: Optional[Any], rhs: Any, fontsize: int = 26) -> io.BytesIO:
    if lhs is None:
        return render_expr_image(rhs)

    lhs_img = _to_image(lhs)
    eq_img = _render_equals_sign(fontsize)
    rhs_img = _to_image(rhs)

    imgs = [lhs_img, eq_img, rhs_img]
    gap = 26
    total_w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    max_h = max(im.height for im in imgs)

    canvas = Image.new("RGB", (total_w, max_h), _BG_COLOR)
    x = 0
    for im in imgs:
        y = (max_h - im.height) // 2
        canvas.paste(im, (x, y))
        x += im.width + gap

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    buf.seek(0)
    return buf