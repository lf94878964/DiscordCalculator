# -*- coding: utf-8 -*-
"""
_engine.py — /計算機 指令的數學運算核心

設計重點：
1. 全部透過 SymPy 完成運算（精確運算優先，必要時才 .evalf()）
2. 自訂 SymPy 未內建的函數（H(n,k) 重複組合、少見三角函數等）
3. 三道安全防護，避免使用者輸入造成程式卡死或記憶體爆炸：
   a. 前置正規表示式檢查（數字上限 / 次方上限）→ 最快、成本最低
   b. 個別函數內建的引數上限檢查（factorial / comb / perm / H / isqrt）
   c. 使用獨立子行程（multiprocessing）執行實際運算，並用 timeout 強制
      終止行程 → 就算前兩道防線被繞過（例如巢狀冪運算），也不會卡住主流程
4. 使用 AST 白名單驗證，避免使用者透過屬性存取（例如 (1).__class__...）
   逃逸出我們允許的函數集合

注意：由於是把使用者輸入的字串丟給 SymPy 的表達式解析器處理，
無法做到 100% 資安等級的沙盒隔離，建議搭配「最小權限的執行環境」
（例如限制記憶體 / 停用網路的容器）一起使用。
"""

from __future__ import annotations

import ast
import functools
import multiprocessing as mp
import re
from decimal import Decimal, ROUND_DOWN, getcontext
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import sympy
from sympy import (
    Abs, E, Matrix, MatrixBase, oo, pi,
    acos, acosh, acot, acsc, asec, asin, asinh, atan, atanh,
    binomial, ceiling, cos, cosh, cot, csc, erf, exp, factorial,
    floor, gamma, log as sympy_log, sec, simplify, sin, sinh, sqrt, tan, tanh,
)

tau = 2 * pi
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)



class CalcError(Exception):
    """114514"""



MAX_NUMBER = 10 ** 6          
MAX_EXPONENT = 1000            
MAX_FACT_ARG = 500              
MAX_MATRIX_DIM = 12             
COMPUTE_TIMEOUT = 6.0            

LARGE_INT_THRESHOLD = 2 ** 31    
DECIMAL_DIGITS = 12


_NUM_PATTERN = re.compile(r"\d+(\.\d+)?")
_POW_PATTERN = re.compile(r"(\*\*|\^)\s*(\d+(\.\d+)?)")
_POW_TOWER_PATTERN = re.compile(r"(\*\*|\^)\s*\d+(\.\d+)?\s*(\*\*|\^)")


def _check_raw_safety(raw: str) -> None:
    for m in _NUM_PATTERN.finditer(raw):
        try:
            val = float(m.group())
        except ValueError:
            continue
        if abs(val) > MAX_NUMBER:
            raise CalcError(
                f"數字 `{m.group()}` 超過允許範圍（上限 {MAX_NUMBER:,}），"
                "請輸入較小的數值。"
            )
    for m in _POW_PATTERN.finditer(raw):
        try:
            val = float(m.group(2))
        except ValueError:
            continue
        if abs(val) > MAX_EXPONENT:
            raise CalcError(
                f"次方數 `{m.group(2)}` 過大（上限 {MAX_EXPONENT:,}），"
                "可能導致運算卡住，請縮小範圍。"
            )
    if _POW_TOWER_PATTERN.search(raw):
        raise CalcError(
            "偵測到連續冪次（例如 `9**9**9`），結果可能瞬間變成天文數字。"
            "請用括號明確拆開，例如 `9**(9**2)`。"
        )


def _guard_int_arg(value, name: str):
    v = sympy.sympify(value)
    if not v.is_number:
        raise CalcError(f"`{name}` 需要數值引數。")
    fv = float(v)
    if fv < 0:
        raise CalcError(f"`{name}` 的引數不可為負數。")
    if fv > MAX_FACT_ARG:
        raise CalcError(
            f"`{name}` 的引數過大（上限 {MAX_FACT_ARG}），避免運算時間過長。"
        )
    return v



def _safe_factorial(n):
    n = _guard_int_arg(n, "factorial(x) / x!")
    return factorial(n)


def _comb(n, k):
    n = _guard_int_arg(n, "comb(n, k)")
    k = _guard_int_arg(k, "comb(n, k)")
    return binomial(n, k)


def _perm(n, k):
    n = _guard_int_arg(n, "perm(n, k)")
    k = _guard_int_arg(k, "perm(n, k)")
    return factorial(n) / factorial(n - k)


def _h_repetition(n, k):
    """重複組合 H(n, k) = C(n + k - 1, k)，SymPy 未內建，故自訂。"""
    n = _guard_int_arg(n, "h(n, k)")
    k = _guard_int_arg(k, "h(n, k)")
    return binomial(n + k - 1, k)


def _isqrt(n):
    n = _guard_int_arg(n, "isqrt(n)")
    if not n.is_integer:
        raise CalcError("isqrt(n) 僅支援整數輸入。")
    return floor(sqrt(n))


def _cbrt(x):
    return sympy.cbrt(sympy.sympify(x))


def _gcd_variadic(*args):
    if len(args) < 2:
        raise CalcError("gcd() 至少需要兩個引數。")
    vals = [sympy.sympify(a) for a in args]
    return functools.reduce(lambda a, b: sympy.gcd(a, b), vals)


def _lcm_variadic(*args):
    if len(args) < 2:
        raise CalcError("lcm() 至少需要兩個引數。")
    vals = [sympy.sympify(a) for a in args]
    return functools.reduce(lambda a, b: sympy.lcm(a, b), vals)


def _hypot(a, b):
    a, b = sympy.sympify(a), sympy.sympify(b)
    return sqrt(a ** 2 + b ** 2)


def _as_point(p):
    if isinstance(p, (list, tuple)):
        return Matrix(list(p))
    if isinstance(p, MatrixBase):
        return p
    return Matrix([sympy.sympify(p)])


def _dist(p, q):
    pv, qv = _as_point(p), _as_point(q)
    if pv.shape != qv.shape:
        raise CalcError("dist(p, q) 的兩個點維度不一致。")
    return sqrt(sum((a - b) ** 2 for a, b in zip(pv, qv)))


def _round_custom(x, n=0):
    x = sympy.sympify(x)
    n = int(n)
    if not x.is_number:
        raise CalcError("round() 僅支援數值運算。")
    return sympy.Float(round(float(x), n)) if n else sympy.Integer(round(float(x)))


def _degrees(x):
    return sympy.sympify(x) * 180 / pi


def _radians(x):
    return sympy.sympify(x) * pi / 180


def _versin(x):
    return 1 - cos(x)


def _vercosin(x):
    return 1 + cos(x)


def _coversin(x):
    return 1 - sin(x)


def _covercosin(x):
    return 1 + sin(x)


def _haversin(x):
    return (1 - cos(x)) / 2


def _havercosin(x):
    return (1 + cos(x)) / 2


def _hacoversin(x):
    return (1 - sin(x)) / 2


def _hacovercosin(x):
    return (1 + sin(x)) / 2


def _exsec(x):
    return sec(x) - 1


def _excsc(x):
    return csc(x) - 1


def _crd(x):
    return 2 * sin(x / 2)


def _nth_root(x, n):
    x = sympy.sympify(x)
    n_sym = sympy.sympify(n)
    if not n_sym.is_number:
        raise CalcError("root(x, n) 的 n 必須是數值。")
    if not n_sym.is_integer:
        raise CalcError("root(x, n) 的 n 必須是整數。")
    n_val = int(n_sym)
    if n_val == 0:
        raise CalcError("root(x, n) 的 n 不可為零。")
    if x.is_number and x.is_real and x.is_negative:
        if n_val % 2 == 0:
            raise CalcError("負數無法計算偶數次方根（結果為虛數，暫不支援）。")
        return -((-x) ** sympy.Rational(1, n_val))
    return x ** sympy.Rational(1, n_val)



def _stat_values(args) -> List[Any]:
    if len(args) < 1:
        raise CalcError("統計函數至少需要一個數值。")
    return [sympy.sympify(a) for a in args]


def _stat_min(*args):
    return sympy.Min(*_stat_values(args))


def _stat_max(*args):
    return sympy.Max(*_stat_values(args))


def _stat_mean(*args):
    vals = _stat_values(args)
    return sympy.Add(*vals) / len(vals)


def _stat_median(*args):
    vals = sorted(_stat_values(args), key=lambda v: float(v))
    n = len(vals)
    mid = n // 2
    if n % 2 == 1:
        return vals[mid]
    return (vals[mid - 1] + vals[mid]) / 2


def _stat_mode(*args):
    vals = _stat_values(args)
    counts = Counter(vals)
    max_count = max(counts.values())
    if max_count == 1:
        raise CalcError("mode()：資料沒有明顯的眾數（每個數值都只出現一次）。")
    modes = []
    for v in vals:
        if counts[v] == max_count and v not in modes:
            modes.append(v)
    if len(modes) == 1:
        return modes[0]
    return Matrix(modes)


def _stat_variance(args, sample: bool = True):
    vals = _stat_values(args)
    n = len(vals)
    if sample and n < 2:
        raise CalcError("樣本變異數至少需要兩個數值。")
    mean = sympy.Add(*vals) / n
    ss = sympy.Add(*[(v - mean) ** 2 for v in vals])
    denom = (n - 1) if sample else n
    return ss / denom


def _stat_var(*args):
    return _stat_variance(args, sample=True)


def _stat_sd(*args):
    return sqrt(_stat_var(*args))


def _stat_ppmcc(*args):
    vals = _stat_values(args)
    n = len(vals)
    if n < 4 or n % 2 != 0:
        raise CalcError(
            "ppmcc(x1,x2,...,y1,y2,...) 需要成對的偶數個數值"
            "（前半段視為 X 資料，後半段視為 Y 資料）。"
        )
    half = n // 2
    xs, ys = vals[:half], vals[half:]
    mx = sympy.Add(*xs) / half
    my = sympy.Add(*ys) / half
    cov = sympy.Add(*[(x - mx) * (y - my) for x, y in zip(xs, ys)])
    sx = sqrt(sympy.Add(*[(x - mx) ** 2 for x in xs]))
    sy = sqrt(sympy.Add(*[(y - my) ** 2 for y in ys]))
    if sx == 0 or sy == 0:
        raise CalcError("ppmcc()：其中一組資料的變異為零，無法計算相關係數。")
    return cov / (sx * sy)


def _stat_zscore(*args):
    vals = _stat_values(args)
    mean = sympy.Add(*vals) / len(vals)
    sd = _stat_sd(*args)
    if sd == 0:
        raise CalcError("z-score()：資料標準差為零，無法計算。")
    return Matrix([(v - mean) / sd for v in vals])


def _stat_cv(*args):
    vals = _stat_values(args)
    mean = sympy.Add(*vals) / len(vals)
    if mean == 0:
        raise CalcError("cv()：平均數為零，無法計算變異係數。")
    return _stat_sd(*args) / mean


def _quantile(vals_sorted: List[Any], q):
    n = len(vals_sorted)
    if n == 1:
        return vals_sorted[0]
    pos = q * (n - 1)
    lo = int(sympy.floor(pos))
    hi = int(sympy.ceiling(pos))
    lo = max(0, min(lo, n - 1))
    hi = max(0, min(hi, n - 1))
    if lo == hi:
        return vals_sorted[lo]
    frac = pos - lo
    return vals_sorted[lo] + (vals_sorted[hi] - vals_sorted[lo]) * frac


def _stat_quartile(args, which: int):
    vals = sorted(_stat_values(args), key=lambda v: float(v))
    qmap = {1: sympy.Rational(1, 4), 2: sympy.Rational(1, 2),
            3: sympy.Rational(3, 4), 4: sympy.Integer(1)}
    return _quantile(vals, qmap[which])


def _stat_qd1(*args):
    return _stat_quartile(args, 1)


def _stat_qd2(*args):
    return _stat_quartile(args, 2)


def _stat_qd3(*args):
    return _stat_quartile(args, 3)


def _stat_qd4(*args):
    return _stat_quartile(args, 4)


def _stat_iqr(*args):
    return _stat_qd3(*args) - _stat_qd1(*args)


def _stat_asym(*args):
    """不對稱性（偏度），採母體偏度公式 m3 / m2^1.5。"""
    vals = _stat_values(args)
    n = len(vals)
    if n < 3:
        raise CalcError("asym()（偏度）至少需要三個數值。")
    mean = sympy.Add(*vals) / n
    m2 = sympy.Add(*[(v - mean) ** 2 for v in vals]) / n
    m3 = sympy.Add(*[(v - mean) ** 3 for v in vals]) / n
    if m2 == 0:
        raise CalcError("asym()：資料變異為零，無法計算偏度。")
    return m3 / (m2 ** sympy.Rational(3, 2))




def _as_matrix(v):
    if isinstance(v, MatrixBase):
        return v
    if isinstance(v, (list, tuple)):
        return Matrix(v)
    raise CalcError("此函數需要向量或矩陣作為引數（請用中括號輸入，如 [1,2,3]）。")


def _dot(v1, v2):
    v1, v2 = _as_matrix(v1), _as_matrix(v2)
    return v1.dot(v2)


def _cross(v1, v2):
    v1, v2 = _as_matrix(v1), _as_matrix(v2)
    if v1.shape != (3, 1) and v1.shape != (1, 3):
        raise CalcError("cross(v1, v2) 僅支援三維向量。")
    return v1.cross(v2)


def _norm(v):
    return _as_matrix(v).norm()


def _normalize(v):
    m = _as_matrix(v)
    n = m.norm()
    if n == 0:
        raise CalcError("零向量無法正規化。")
    return m / n


def _make_angle_between(to_unit):
    def wrapped(v1, v2):
        v1, v2 = _as_matrix(v1), _as_matrix(v2)
        denom = v1.norm() * v2.norm()
        if denom == 0:
            raise CalcError("零向量無法計算夾角。")
        return to_unit(sympy.acos(v1.dot(v2) / denom))
    return wrapped


def _projection(v1, v2):
    v1, v2 = _as_matrix(v1), _as_matrix(v2)
    denom = v2.dot(v2)
    if denom == 0:
        raise CalcError("無法投影到零向量上。")
    return (v1.dot(v2) / denom) * v2


def _distance_vec(v1, v2):
    v1, v2 = _as_matrix(v1), _as_matrix(v2)
    return (v1 - v2).norm()





_ANGLE_INPUT_FUNCS = {
    "sin": sin, "cos": cos, "tan": tan, "cot": cot, "sec": sec, "csc": csc,
    "versin": _versin, "vercosin": _vercosin,
    "coversin": _coversin, "covercosin": _covercosin,
    "haversin": _haversin, "havercosin": _havercosin,
    "hacoversin": _hacoversin, "hacovercosin": _hacovercosin,
    "exsec": _exsec, "excsc": _excsc, "crd": _crd,
}


_ANGLE_OUTPUT_FUNCS = {
    "asin": asin, "acos": acos, "atan": atan,
    "acot": acot, "asec": asec, "acsc": acsc,
}


_HYPERBOLIC_FUNCS = {
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "asinh": asinh, "acosh": acosh, "atanh": atanh,
}


def _make_angle_input_wrapper(func, to_rad):
    def wrapped(x):
        return func(to_rad(sympy.sympify(x)))
    return wrapped


def _make_angle_output_wrapper(func, to_unit):
    def wrapped(x):
        return to_unit(func(sympy.sympify(x)))
    return wrapped



_step_log: List[Tuple[Any, Any]] = []


_TRACKED_FUNC_NAMES = {
    "sqrt", "isqrt", "cbrt", "pow", "exp", "abs", "Abs",
    "log", "ln", "log10", "log2", "root",
    "degrees", "deg", "radians", "rad",
    "floor", "ceil", "ceiling", "round",
    "factorial", "gcd", "lcm", "comb", "C", "perm", "P", "h", "H",
    "hypot", "dist",
    "gamma", "erf",
    "min", "max", "avg", "mean", "med", "mode", "var", "sd",
    "ppmcc", "pccs", "zscore", "cv",
    "qd1", "qd2", "qd3", "qd4", "iqr", "asym",
} | set(_ANGLE_INPUT_FUNCS) | set(_ANGLE_OUTPUT_FUNCS) | set(_HYPERBOLIC_FUNCS)


def _record_and_wrap(display_name: str, func):
    """包一層：呼叫當下記錄「符號呼叫式 = 實際結果」，供步驟顯示用。"""
    def wrapped(*args, **kwargs):
        result = func(*args, **kwargs)
        try:
            sym_args = [a if isinstance(a, (sympy.Basic, MatrixBase)) else sympy.sympify(a)
                        for a in args]
            call_expr = sympy.Function(display_name)(*sym_args)
            if call_expr != result:
                _step_log.append((call_expr, result))
        except Exception:
            pass
        return result
    return wrapped


def build_namespace(mode: str, angle_unit: str) -> dict:
    """建立 sympify 解析時使用的白名單函式/常數表。"""

    if angle_unit == "degree":
        to_rad = lambda x: x * pi / 180
        to_unit = lambda x: x * 180 / pi
    else:
        to_rad = lambda x: x
        to_unit = lambda x: x

    ns: dict = {
        # 常數
        "pi": pi, "π": pi, "e": E, "tau": tau, "τ": tau, "inf": oo, "oo": oo,
        # 虛數
        "i": sympy.I, "I": sympy.I,
        # 基礎 / 進階運算
        "sqrt": lambda x: sqrt(sympy.sympify(x)),
        "isqrt": _isqrt,
        "cbrt": _cbrt,
        "pow": lambda x, y: sympy.sympify(x) ** sympy.sympify(y),
        "exp": lambda x: exp(sympy.sympify(x)),
        "abs": lambda x: Abs(sympy.sympify(x)),
        "Abs": lambda x: Abs(sympy.sympify(x)),
        # 對數
        "log": lambda x, b=None: (sympy_log(sympy.sympify(x)) if b is None
                                   else sympy_log(sympy.sympify(b)) / sympy_log(sympy.sympify(x))),
        "ln": lambda x: sympy_log(sympy.sympify(x)),
        "log10": lambda x: sympy_log(sympy.sympify(x), 10),
        "log2": lambda x: sympy_log(sympy.sympify(x), 2),
        # 角度轉換
        "degrees": _degrees, "deg": _degrees,
        "radians": _radians, "rad": _radians,
        # 取整
        "floor": lambda x: floor(sympy.sympify(x)),
        "ceil": lambda x: ceiling(sympy.sympify(x)),
        "ceiling": lambda x: ceiling(sympy.sympify(x)),
        "round": _round_custom,
        # 組合數學 / 數論
        "factorial": _safe_factorial,
        "gcd": _gcd_variadic,
        "lcm": _lcm_variadic,
        "comb": _comb, "C": _comb,
        "perm": _perm, "P": _perm,
        "h": _h_repetition, "H": _h_repetition,
        "hypot": _hypot,
        "dist": _dist,
        # 雙曲函數
        **_HYPERBOLIC_FUNCS,
        # 工程數學
        "gamma": lambda x: gamma(sympy.sympify(x)),
        "erf": lambda x: erf(sympy.sympify(x)),
        # 任意次方根
        "root": _nth_root,
        # 統計函數
        "min": _stat_min, "max": _stat_max,
        "avg": _stat_mean, "mean": _stat_mean,
        "med": _stat_median,
        "mode": _stat_mode,
        "var": _stat_var,
        "sd": _stat_sd,
        "ppmcc": _stat_ppmcc, "pccs": _stat_ppmcc,
        "zscore": _stat_zscore,
        "cv": _stat_cv,
        "qd1": _stat_qd1, "qd2": _stat_qd2, "qd3": _stat_qd3, "qd4": _stat_qd4,
        "iqr": _stat_iqr,
        "asym": _stat_asym,
        # 矩陣建構子（Matrix.eye / zeros / ones 等透過屬性存取使用）
        "Matrix": Matrix,
        # SymPy 標準解析轉換（auto_number 等）內部會參照到這些名稱
        "Integer": sympy.Integer,
        "Float": sympy.Float,
        "Rational": sympy.Rational,
        "Symbol": sympy.Symbol,
        "Add": sympy.Add,
        "Mul": sympy.Mul,
        "Pow": sympy.Pow,
    }

    for name, func in _ANGLE_INPUT_FUNCS.items():
        ns[name] = _make_angle_input_wrapper(func, to_rad)
    for name, func in _ANGLE_OUTPUT_FUNCS.items():
        ns[name] = _make_angle_output_wrapper(func, to_unit)

    if mode in ("vector", "matrix"):
        ns.update({
            "dot": _dot,
            "cross": _cross,
            "norm": _norm,
            "normalize": _normalize,
            "angle": _make_angle_between(to_unit),
            "projection": _projection,
            "distance": _distance_vec,
        })

    for name in list(ns.keys()):
        if name in _TRACKED_FUNC_NAMES and callable(ns[name]):
            ns[name] = _record_and_wrap(name, ns[name])

    return ns


def _convert_sqrt_symbol(s: str) -> str:
    s = re.sub(r"√\s*([A-Za-z0-9_.]+)", r"sqrt(\1)", s)
    s = s.replace("√", "sqrt")
    return s


def _convert_factorial_symbol(s: str) -> str:
    """把 x! 或 (expr)! 轉成 factorial(x) / factorial(expr)。"""
    out: List[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "!":
            if out and out[-1] == ")":
                depth = 0
                k = len(out) - 1
                while k >= 0:
                    if out[k] == ")":
                        depth += 1
                    elif out[k] == "(":
                        depth -= 1
                        if depth == 0:
                            break
                    k -= 1
                if k < 0:
                    raise CalcError("`!` 前面的括號不成對。")
                atom = "".join(out[k:])
                del out[k:]
                out.append(f"factorial({atom})")
            else:
                k = len(out) - 1
                while k >= 0 and (out[k].isalnum() or out[k] in "_."):
                    k -= 1
                atom = "".join(out[k + 1:])
                if not atom:
                    raise CalcError("`!` 前面缺少可計算的數值或變數。")
                del out[k + 1:]
                out.append(f"factorial({atom})")
            i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


_HEX_PATTERN = re.compile(r"0[xX][0-9A-Fa-f]+")


def _convert_hex_literals(s: str) -> str:
    """把 0x1A 這種十六進位字面值（含 AaCcDdFf 等字母）轉成十進位數字字串。"""
    return _HEX_PATTERN.sub(lambda m: str(int(m.group(), 16)), s)


_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_PATTERN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")


def _convert_superscript_exponents(s: str) -> str:
    """把上標數字（例如 2³）轉成 **3 這種一般冪次語法。"""
    def repl(m):
        digits = m.group(0).translate(_SUPERSCRIPT_MAP)
        return f"**({digits})"
    return _SUPERSCRIPT_PATTERN.sub(repl, s)


_THOUSANDS_PATTERN = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _strip_thousands_separators(s: str) -> str:
    """把數字中的千分位逗號拿掉，例如 12,345 -> 12345。"""
    return _THOUSANDS_PATTERN.sub(lambda m: m.group(0).replace(",", ""), s)


_REPEAT_NINE_PATTERN = re.compile(r"\d+\.\d*9{13,}\d*")


def _collapse_repeating_nines(s: str) -> str:
    """0.9999999999999（連續 13 位以上的 9，視為循環小數）直接視為 1。"""
    def repl(m):
        try:
            return str(int(round(float(m.group(0)))))
        except Exception:
            return m.group(0)
    return _REPEAT_NINE_PATTERN.sub(repl, s)


def _convert_abs_bars(s: str) -> str:
    buffers: List[List[str]] = [[]]
    for c in s:
        if c == "|":
            if len(buffers) == 1:
                buffers.append([])
            elif not buffers[-1]:
                buffers.append([])
            else:
                content = "".join(buffers.pop())
                buffers[-1].append(f"Abs({content})")
        else:
            buffers[-1].append(c)
    while len(buffers) > 1:
        content = "".join(buffers.pop())
        buffers[-1].append("|" + content)
    return "".join(buffers[0])


def _convert_mod_operator(s: str) -> str:
    return re.sub(r"\s+mod\s+", "%", s, flags=re.IGNORECASE)


def _convert_zscore_alias(s: str) -> str:
    return re.sub(r"z-score", "zscore", s, flags=re.IGNORECASE)


def substitute_placeholders(raw: str, variables: Optional[str]) -> str:
    if "?" not in raw:
        return raw
    count = raw.count("?")
    if not variables or not variables.strip():
        raise CalcError(f"算式包含 {count} 個 `?` 佔位符，請在「替換值」欄位提供對應的數值。")
    parts = [v.strip() for v in re.split(r"[,，、]", variables) if v.strip() != ""]
    if len(parts) != count:
        raise CalcError(
            f"`?` 佔位符數量（{count}）與提供的替換值數量（{len(parts)}）不一致。"
        )
    it = iter(parts)
    return re.sub(r"\?", lambda _m: next(it), raw)


_IMAG_LITERAL_PATTERN = re.compile(r"(?<![A-Za-z0-9_.])(\d+(?:\.\d+)?)[iI](?![A-Za-z0-9_])")
_IMAG_LITERAL_AFTER_PAREN_PATTERN = re.compile(r"(\))[iI](?![A-Za-z0-9_])")


def _convert_imaginary_literal(s: str) -> str:
    s = _IMAG_LITERAL_PATTERN.sub(lambda m: f"{m.group(1)}*I", s)
    s = _IMAG_LITERAL_AFTER_PAREN_PATTERN.sub(lambda m: f"{m.group(1)}*I", s)
    return s


def _insert_implicit_mult(s: str) -> str:
    s = re.sub(r"(?<![A-Za-z0-9_])(\d+(?:\.\d+)?)(\()", r"\1*\2", s)
    s = re.sub(r"(\))(\d+(?:\.\d+)?)", r"\1*\2", s)
    s = re.sub(r"(\))(\()", r"\1*\2", s)
    return s


def _wrap_bracket_literals(s: str) -> str:
    out: List[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "[":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if s[j] == "[":
                    depth += 1
                elif s[j] == "]":
                    depth -= 1
                j += 1
            if depth != 0:
                raise CalcError("中括號不成對，請檢查向量/矩陣輸入格式。")
            inner = s[i:j]
            out.append(f"Matrix({inner})")
            i = j
        else:
            out.append(c)
            i += 1
    return "".join(out)


def preprocess(raw: str, mode: str) -> str:
    s = raw.strip()
    s = _collapse_repeating_nines(s)
    s = _strip_thousands_separators(s)
    s = _convert_hex_literals(s)
    s = s.replace("×", "*").replace("÷", "/").replace("^", "**")
    s = s.replace("π", "pi").replace("𝝅", "pi").replace("τ", "tau")
    s = _convert_superscript_exponents(s)
    if mode == "scalar":
        s = s.replace("[", "(").replace("]", ")")
        s = s.replace("{", "(").replace("}", ")")
    s = _convert_abs_bars(s)
    s = _convert_sqrt_symbol(s)
    s = _convert_factorial_symbol(s)
    s = _insert_implicit_mult(s)
    s = _convert_mod_operator(s)
    s = _convert_zscore_alias(s)
    if mode in ("vector", "matrix") or ("[" in s and "]" in s):
        s = _wrap_bracket_literals(s)
    return s


def detect_mode(raw: str) -> str:
    if "[" in raw and "]" in raw:
        if re.search(r"\[\s*\[", raw):
            return "matrix"
        return "vector"
    return "scalar"




_ALLOWED_METHODS = {
    "T", "inv", "det", "rank", "trace", "eigenvals", "eigenvects",
    "nullspace", "columnspace", "LUdecomposition", "QRdecomposition",
    "diagonalize", "jordan_form", "solve", "pinv", "applyfunc", "norm",
    "distance", "dot", "cross", "normalize", "doit", "evalf", "simplify",
    "expand", "factor", "eye", "zeros", "ones",
}

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
    ast.Constant, ast.Attribute, ast.List, ast.Tuple,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
    ast.USub, ast.UAdd, ast.keyword,
)


def validate_ast(s: str) -> None:
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise CalcError(f"算式格式錯誤，請確認輸入是否正確：{e}")

    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise CalcError(
                f"算式包含不支援的語法（{type(node).__name__}），請確認輸入。"
            )
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") and node.attr.endswith("__"):
                raise CalcError("算式包含不允許的屬性存取。")
            if node.attr not in _ALLOWED_METHODS:
                raise CalcError(f"不支援的方法：`.{node.attr}()`")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise CalcError("算式包含不允許的名稱。")




def _check_matrix_dim_safety(raw: str) -> None:
    rows = re.findall(r"\[([^\[\]]*)\]", raw)
    for row in rows:
        count = len([p for p in row.split(",") if p.strip() != ""])
        if count > MAX_MATRIX_DIM:
            raise CalcError(
                f"向量/矩陣單一維度過大（上限 {MAX_MATRIX_DIM}），請縮小輸入規模。"
            )
    if len(rows) > MAX_MATRIX_DIM:
        raise CalcError(
            f"矩陣列數過多（上限 {MAX_MATRIX_DIM}），請縮小輸入規模。"
        )



@dataclass
class CalcResult:
    mode: str
    original_display: str
    result: Any

    steps: List[Tuple[str, Any, Any]] = field(default_factory=list)



def _make_display_wrapper(display_name: str):
    def wrapped(*args, **kwargs):
        sym_args = [a if isinstance(a, (sympy.Basic, MatrixBase)) else sympy.sympify(a)
                    for a in args]
        return sympy.Function(display_name)(*sym_args)
    return wrapped


def build_display_namespace(angle_unit: str) -> dict:
    ns = build_namespace("scalar", angle_unit)
    for name in list(ns.keys()):
        if name in _TRACKED_FUNC_NAMES:
            ns[name] = _make_display_wrapper(name)
    return ns


def _find_reducible_node(node: Any):
    args = getattr(node, "args", ())
    if not args:
        return None
    for child in reversed(args):
        found = _find_reducible_node(child)
        if found is not None:
            return found
    if all(isinstance(a, sympy.Number) for a in args):
        return node
    return None


def _replace_node_keep_unevaluated(state: Any, target: Any, value: Any):
    if state == target:
        return value
    args = getattr(state, "args", ())
    if not args:
        return state
    new_args = tuple(_replace_node_keep_unevaluated(a, target, value) for a in args)
    if new_args == args:
        return state
    try:
        return state.func(*new_args, evaluate=False)
    except TypeError:
        return state.func(*new_args)


def _reduce_one_step(state: Any, step_log_map: dict):
    target = _find_reducible_node(state)
    if target is None:
        return None, None

    if isinstance(target, sympy.core.function.AppliedUndef):
        if target not in step_log_map:
            return None, None
        value = step_log_map[target]
    else:
        try:
            value = target.func(*target.args)
        except Exception:
            return None, None
        if not isinstance(value, sympy.Number):
            return None, None

    new_state = _replace_node_keep_unevaluated(state, target, value)
    return target, new_state


from sympy.parsing.sympy_parser import (
    EvaluateFalseTransformer,
    eval_expr as _sympy_eval_expr,
    stringify_expr as _sympy_stringify_expr,
)


class _ModAwareEvaluateFalseTransformer(EvaluateFalseTransformer):
    def visit_BinOp(self, node):
        if isinstance(node.op, ast.Mod):
            left = self.visit(node.left)
            right = self.visit(node.right)
            return ast.Call(
                func=ast.Name(id="Mod", ctx=ast.Load()),
                args=[left, right],
                keywords=[ast.keyword(arg="evaluate", value=ast.Constant(value=False))],
            )
        return super().visit_BinOp(node)


def _evaluate_false_mod_aware(code: str):
    node = ast.parse(code)
    transformed_node = _ModAwareEvaluateFalseTransformer().visit(node)
    transformed_node = ast.Expression(transformed_node.body[0].value)
    return ast.fix_missing_locations(transformed_node)


def _build_display_expr(processed: str, angle_unit: str):
    try:
        display_namespace = build_display_namespace(angle_unit)
        display_namespace.setdefault("Mod", sympy.Mod)
        display_global = {"__builtins__": {}}
        display_global.update(display_namespace)
        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        code = _sympy_stringify_expr(processed, display_namespace, display_global, transformations)
        compiled = compile(_evaluate_false_mod_aware(code), "<string>", "eval")
        display_expr = _sympy_eval_expr(compiled, display_namespace, display_global)
        if isinstance(display_expr, list) or isinstance(display_expr, MatrixBase):
            return None
        return display_expr
    except Exception:
        return None


def _exprs_equal(a: Any, b: Any) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return False
    if isinstance(a, MatrixBase) or isinstance(b, MatrixBase):
        return False
    try:
        if a == b:
            return True
        diff = sympy.simplify(sympy.sympify(a) - sympy.sympify(b))
        return diff == 0
    except Exception:
        return False


def _exprs_equal_after_subs(a: Any, b: Any, subs_map: dict) -> bool:
    if isinstance(a, str) or isinstance(b, str):
        return False
    if isinstance(a, MatrixBase) or isinstance(b, MatrixBase):
        return False
    try:
        aa = sympy.sympify(a).xreplace(subs_map) if subs_map else sympy.sympify(a)
        bb = sympy.sympify(b).xreplace(subs_map) if subs_map else sympy.sympify(b)
        if aa == bb:
            return True
        diff = sympy.simplify(aa - bb)
        return diff == 0
    except Exception:
        return False


def _guard_degenerate_lhs(lhs: Any, rhs: Any, raw: str) -> Any:
    if lhs is None or isinstance(lhs, str):
        return lhs
    if isinstance(lhs, MatrixBase) or isinstance(rhs, MatrixBase):
        return lhs
    try:
        if lhs == rhs:
            return raw
    except Exception:
        pass
    return lhs


def _format_float_str(value: Any, digits: int = DECIMAL_DIGITS) -> Any:
    fv = float(value)
    s = f"{fv:.{digits}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-", "-0"):
        s = "0"
    if "." in s:
        sig = len(s.replace(".", "").replace("-", "").lstrip("0")) or 1
        return sympy.Float(s, sig)
    return sympy.Integer(s)


def _cap_float_precision(expr: Any, digits: int = DECIMAL_DIGITS) -> Any:
    if isinstance(expr, MatrixBase):
        return expr.applyfunc(lambda v: _cap_float_precision(v, digits))
    try:
        if isinstance(expr, sympy.Float):
            return _format_float_str(expr, digits)
        if hasattr(expr, "atoms"):
            floats = expr.atoms(sympy.Float)
            if floats:
                mapping = {f: _format_float_str(f, digits) for f in floats}
                return expr.xreplace(mapping)
    except Exception:
        pass
    return expr

_SCI_NOTATION_SMALL_THRESHOLD = Decimal(1).scaleb(-DECIMAL_DIGITS)


def format_large_result(value: Any) -> Any:
    if isinstance(value, MatrixBase):
        return value.applyfunc(format_large_result)
    try:
        v = sympy.sympify(value)
        if not (v.is_number and getattr(v, "is_real", False) and v.is_finite):
            return value
        if v == 0:
            return value
        ev = v.evalf(30)
        dec = Decimal(str(ev))
        abs_dec = abs(dec)
        if not (abs_dec > LARGE_INT_THRESHOLD or abs_dec < _SCI_NOTATION_SMALL_THRESHOLD):
            return value
        sci_str = format(dec, ".6e")
        mant, exp_str = sci_str.split("e")
        exp = int(exp_str)
        if "." in mant:
            mant = mant.rstrip("0").rstrip(".")
        if mant in ("", "-"):
            mant += "0"
        mant_val = sympy.Float(mant)
        power = sympy.Pow(sympy.Integer(10), exp, evaluate=False)
        return sympy.Mul(mant_val, power, evaluate=False)
    except Exception:
        pass
    return value


def has_decimal_variant(value: Any) -> bool:
    if isinstance(value, MatrixBase):
        return any(has_decimal_variant(v) for v in value)
    try:
        v = sympy.sympify(value)
        if not v.is_number:
            return False
        re_part, im_part = sympy.re(v), sympy.im(v)
        if im_part != 0:
            return has_decimal_variant(re_part) or has_decimal_variant(im_part)
        if re_part.is_Integer:
            return False
        return True
    except Exception:
        return False


def to_decimal_display(value: Any, digits: int = DECIMAL_DIGITS) -> Any:
    """把結果（純量或矩陣）轉換成小數表示法，最多到小數點後 `digits` 位。"""
    if isinstance(value, MatrixBase):
        return value.applyfunc(lambda v: to_decimal_display(v, digits))
    try:
        v = sympy.sympify(value)
        if not v.is_number:
            return value
        re_part, im_part = sympy.re(v), sympy.im(v)
        if im_part != 0:
            re_val = to_decimal_display(re_part, digits) if re_part != 0 else sympy.Integer(0)
            im_val = to_decimal_display(im_part, digits)
            return sympy.Add(re_val, sympy.Mul(im_val, sympy.I, evaluate=False), evaluate=False)
        fv = v.evalf(30)
        return _format_float_str(fv, digits)
    except Exception:
        return value


def format_decimal_string(value: Any, digits: int = DECIMAL_DIGITS) -> str:
    try:
        v = sympy.sympify(value)
    except Exception:
        return str(value)

    if not v.is_number:
        return str(value)

    re_part, im_part = sympy.re(v), sympy.im(v)
    if im_part != 0:
        im_is_neg = bool(im_part.is_negative)
        im_abs_str = format_decimal_string(Abs(im_part), digits)
        if re_part == 0:
            return f"-{im_abs_str}i" if im_is_neg else f"{im_abs_str}i"
        re_str = format_decimal_string(re_part, digits)
        sign = "-" if im_is_neg else "+"
        return f"{re_str}{sign}{im_abs_str}i"

    v = re_part

    if v.is_Integer:
        return str(int(v))

    old_prec = getcontext().prec
    try:
        getcontext().prec = digits + 40

        if v.is_Rational:
            p, q = int(v.p), int(v.q)
            neg = p < 0
            exact = Decimal(abs(p)) / Decimal(q)
            has_more = None  # 稍後用精確值判斷
        else:
            neg = bool(v.is_negative)
            try:
                exact = abs(Decimal(str(v.evalf(digits + 30))))
            except Exception:
                exact = abs(Decimal(str(float(v.evalf(digits + 30)))))
            has_more = True  

        quant = Decimal(1).scaleb(-digits)
        truncated = exact.quantize(quant, rounding=ROUND_DOWN)
        if has_more is None:
            has_more = (exact - truncated) != 0
    except Exception:
        return str(value)
    finally:
        getcontext().prec = old_prec

    s = f"{truncated:.{digits}f}"
    if not has_more:
        if "." in s:
            s = s.rstrip("0").rstrip(".")
    if s in ("", "-0"):
        s = "0"
    if neg and s != "0" and not s.startswith("-"):
        s = "-" + s
    if has_more:
        s += "..."
    return s


def to_decimal_display_text(value: Any, digits: int = DECIMAL_DIGITS):
    if isinstance(value, MatrixBase):
        rows, cols = value.shape
        return [
            [format_decimal_string(value[i, j], digits) for j in range(cols)]
            for i in range(rows)
        ]
    return format_decimal_string(value, digits)


def _do_calculate(raw: str, mode: str, angle_unit: str) -> CalcResult:
    processed = preprocess(raw, mode)
    validate_ast(processed)

    namespace = build_namespace(mode, angle_unit)
    transformations = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    global_dict = {"__builtins__": {}}
    global_dict.update(namespace)

    _step_log.clear()
    try:
        expr = parse_expr(
            processed,
            local_dict=namespace,
            global_dict=global_dict,
            transformations=transformations,
            evaluate=True,
        )
    except CalcError:
        raise
    except Exception as e:
        raise CalcError(f"算式格式錯誤，請確認輸入是否正確：{e}")

    if isinstance(expr, list):
        expr = Matrix(expr)

    sub_steps = list(_step_log) 
    step_log_map = dict(sub_steps)  

    display_expr = _build_display_expr(processed, angle_unit) if mode == "scalar" else None

    steps: List[Tuple[str, Any, Any]] = [
        ("原始算式", None, display_expr if display_expr is not None else raw)
    ]

    if mode == "scalar" and display_expr is not None:
        state = display_expr
        guard = 0
        while guard < 60:
            guard += 1
            prev_state = state
            target, new_state = _reduce_one_step(state, step_log_map)
            if target is None:
                break
            if isinstance(target, sympy.core.function.AppliedUndef):
                steps.append((f"計算 {target}", prev_state, new_state))
            state = new_state
            if isinstance(state, sympy.Number):
                break
    else:
        for call_expr, value in sub_steps:
            steps.append((f"計算 {call_expr}", call_expr, value))

    if mode == "scalar":
        try:
            expanded = sympy.expand(expr)
        except Exception:
            expanded = expr
        if expanded != expr:
            steps.append(("展開 / 化簡", expr, expanded))
        try:
            simplified = simplify(expr)
        except Exception:
            simplified = expanded
        if simplified != expanded:
            steps.append(("進一步化簡", expanded, simplified))
        result = simplified
    else:
        result = expr

    result = _cap_float_precision(result)
    result = format_large_result(result)

    final_lhs = display_expr if display_expr is not None else raw

    skip_final_step = False
    if steps:
        _, last_lhs, last_rhs = steps[-1]
        if (
            last_lhs is not None
            and not isinstance(last_lhs, str)
            and not isinstance(final_lhs, str)
            and not isinstance(last_lhs, MatrixBase)
            and not isinstance(final_lhs, MatrixBase)
            and not isinstance(last_rhs, MatrixBase)
            and not isinstance(result, MatrixBase)
        ):
            try:
                if last_lhs == final_lhs and last_rhs == result:
                    skip_final_step = True
            except Exception:
                pass

    if skip_final_step:
        _, last_lhs, last_rhs = steps[-1]
        steps[-1] = ("最終結果", last_lhs, last_rhs)
    else:
        steps.append(("最終結果", final_lhs, result))

    guarded_steps: List[Tuple[str, Any, Any]] = []
    for title, lhs, rhs in steps:
        guarded_steps.append((title, _guard_degenerate_lhs(lhs, rhs, raw), rhs))

    return CalcResult(mode=mode, original_display=raw, result=result, steps=guarded_steps)


from sympy.core.parameters import evaluate as _sympy_evaluate_ctx

_FROZEN_TAG = "__frozen_sympy_expr__"


def _freeze_step_value(value: Any) -> Any:
    if value is None or isinstance(value, str) or isinstance(value, MatrixBase):
        return value
    try:
        return (_FROZEN_TAG, sympy.srepr(value))
    except Exception:
        return value


def _thaw_step_value(value: Any) -> Any:
    if (
        isinstance(value, tuple)
        and len(value) == 2
        and value[0] == _FROZEN_TAG
    ):
        with _sympy_evaluate_ctx(False):
            return sympy.sympify(value[1], evaluate=False)
    return value


def _freeze_calc_result(result: "CalcResult") -> "CalcResult":
    result.steps = [
        (title, _freeze_step_value(lhs), _freeze_step_value(rhs))
        for title, lhs, rhs in result.steps
    ]
    return result


def _thaw_calc_result(result: "CalcResult") -> "CalcResult":
    result.steps = [
        (title, _thaw_step_value(lhs), _thaw_step_value(rhs))
        for title, lhs, rhs in result.steps
    ]
    return result


def _worker(raw: str, mode: str, angle_unit: str, queue: "mp.Queue") -> None:
    try:
        result = _do_calculate(raw, mode, angle_unit)
        queue.put(("ok", _freeze_calc_result(result)))
    except CalcError as e:
        queue.put(("error", str(e)))
    except Exception as e:  
        queue.put(("error", f"計算過程發生未預期的錯誤：{e}"))


async def safe_calculate(
    raw: str,
    mode: Optional[str],
    angle_unit: str,
    loop=None,
    timeout: float = COMPUTE_TIMEOUT,
    variables: Optional[str] = None,
) -> CalcResult:
    import asyncio

    raw = substitute_placeholders(raw, variables)

    _check_raw_safety(raw)

    resolved_mode = mode or detect_mode(raw)
    if resolved_mode in ("vector", "matrix"):
        _check_matrix_dim_safety(raw)

    if loop is None:
        loop = asyncio.get_event_loop()

    def _run() -> CalcResult:
        ctx = mp.get_context("spawn")
        q: "mp.Queue" = ctx.Queue()
        p = ctx.Process(target=_worker, args=(raw, resolved_mode, angle_unit, q))
        p.start()
        p.join(timeout)
        if p.is_alive():
            p.terminate()
            p.join()
            raise CalcError(
                "計算時間過長（可能是數字過大或運算過於複雜），請簡化算式後再試一次。"
            )
        if q.empty():
            raise CalcError("計算過程發生未知錯誤（子行程未回傳結果）。")
        status, payload = q.get()
        if status == "error":
            raise CalcError(payload)
        return _thaw_calc_result(payload)

    return await loop.run_in_executor(None, _run)