# -*- coding: utf-8 -*-
"""
calculator — /calculator（計算機）指令套件。

模組配置：
    _engine.py  — 數學運算核心（SymPy 解析、安全限制、逐步化簡邏輯）
    _render.py  — 把算式渲染成 LaTeX 風格的 PNG 圖片
    _view.py    — Discord Components V2 結果顯示介面（頁籤、分頁按鈕）
    command.py  — 斜線指令本體（含 `setup()`，供 discord.py 的
                  `bot.load_extension("calculator.command")` 使用）
"""