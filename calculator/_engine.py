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
    pass


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


def _guard_nonneg_integer(value, name: str):
    v = _guard_int_arg(value, name)
    fv = float(v)
    if fv != int(fv):
        raise CalcError(f"`{name}` 僅支援非負整數。")
    return sympy.Integer(int(fv))


def _safe_double_factorial(n):
    n = _guard_nonneg_integer(n, "double_factorial(x) / x!!")
    return sympy.factorial2(n)


def _safe_subfactorial(n):
    n = _guard_nonneg_integer(n, "subfactorial(x) / !x")
    return sympy.subfactorial(n)


def _comb(n, k):
    n = _guard_int_arg(n, "comb(n, k)")
    k = _guard_int_arg(k, "comb(n, k)")
    return binomial(n, k)


def _perm(n, k):
    n = _guard_int_arg(n, "perm(n, k)")
    k = _guard_int_arg(k, "perm(n, k)")
    return factorial(n) / factorial(n - k)


def _h_repetition(n, k):
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
    "factorial", "double_factorial", "factorial2", "subfactorial",
    "gcd", "lcm", "comb", "C", "perm", "P", "h", "H",
    "hypot", "dist",
    "gamma", "erf",
    "min", "max", "avg", "mean", "med", "mode", "var", "sd",
    "ppmcc", "pccs", "zscore", "cv",
    "qd1", "qd2", "qd3", "qd4", "iqr", "asym",
} | set(_ANGLE_INPUT_FUNCS) | set(_ANGLE_OUTPUT_FUNCS) | set(_HYPERBOLIC_FUNCS)


def _record_and_wrap(display_name: str, func):
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

    if angle_unit == "degree":
        to_rad = lambda x: x * pi / 180
        to_unit = lambda x: x * 180 / pi
    else:
        to_rad = lambda x: x
        to_unit = lambda x: x

    ns: dict = {
        "pi": pi, "π": pi, "e": E, "tau": tau, "τ": tau, "inf": oo, "oo": oo,
        "i": sympy.I, "I": sympy.I,
        "sqrt": lambda x: sqrt(sympy.sympify(x)),
        "isqrt": _isqrt,
        "cbrt": _cbrt,
        "pow": lambda x, y: sympy.sympify(x) ** sympy.sympify(y),
        "exp": lambda x: exp(sympy.sympify(x)),
        "abs": lambda x: Abs(sympy.sympify(x)),
        "Abs": lambda x: Abs(sympy.sympify(x)),
        "log": lambda x, b=None: (sympy_log(sympy.sympify(x), 10) if b is None
                                   else sympy_log(sympy.sympify(b)) / sympy_log(sympy.sympify(x))),
        "ln": lambda x: sympy_log(sympy.sympify(x)),
        "log10": lambda x: sympy_log(sympy.sympify(x), 10),
        "log2": lambda x: sympy_log(sympy.sympify(x), 2),
        "degrees": _degrees, "deg": _degrees,
        "radians": _radians, "rad": _radians,
        "floor": lambda x: floor(sympy.sympify(x)),
        "ceil": lambda x: ceiling(sympy.sympify(x)),
        "ceiling": lambda x: ceiling(sympy.sympify(x)),
        "round": _round_custom,
        "factorial": _safe_factorial,
        "double_factorial": _safe_double_factorial,
        "factorial2": _safe_double_factorial,
        "subfactorial": _safe_subfactorial,
        "gcd": _gcd_variadic,
        "lcm": _lcm_variadic,
        "comb": _comb, "C": _comb,
        "perm": _perm, "P": _perm,
        "h": _h_repetition, "H": _h_repetition,
        "hypot": _hypot,
        "dist": _dist,
        **_HYPERBOLIC_FUNCS,
        "gamma": lambda x: gamma(sympy.sympify(x)),
        "erf": lambda x: erf(sympy.sympify(x)),
        "root": _nth_root,
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
        "Matrix": Matrix,
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

    if mode in ("vector", "matrix") or mode in _MATRIX_OP_MODES:
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


_IDENT_TAIL = re.compile(r"[A-Za-z_][A-Za-z0-9_]*$")


def _ends_operand(out: List[str]) -> bool:
    """out 目前的結尾（略過空白）是否是一個運算元的結尾：數字／名稱／右括號。"""
    k = len(out) - 1
    while k >= 0 and out[k].isspace():
        k -= 1
    return k >= 0 and (out[k].isalnum() or out[k] in "_.)")


def _match_paren(s: str, i: int) -> int:
    """s[i] 是 '('，回傳對應 ')' 的索引。"""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    raise CalcError("`!` 後面的括號不成對。")


def _pop_postfix_atom(out: List[str]) -> str:
    """取出後綴 `!`／`!!` 作用的運算元（含前面的函式名稱），並從 out 移除。"""
    while out and out[-1].isspace():
        out.pop()
    if not out:
        raise CalcError("`!` 前面缺少可計算的數值或變數。")
    if out[-1] == ")":
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
        m = _IDENT_TAIL.search("".join(out[:k]))
        if m:  # sin(30)! 這類「函式名稱＋括號」要整個當運算元
            k -= len(m.group())
        atom = "".join(out[k:])
        del out[k:]
        return atom
    k = len(out) - 1
    while k >= 0 and (out[k].isalnum() or out[k] in "_."):
        k -= 1
    atom = "".join(out[k + 1:])
    if not atom:
        raise CalcError("`!` 前面缺少可計算的數值或變數。")
    del out[k + 1:]
    return atom


def _take_prefix_operand(s: str, i: int) -> Tuple[str, int]:
    """前綴 `!x`（子階乘）：從 s[i] 取出運算元，回傳 (運算元, 下一個索引)。"""
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    if i >= n:
        raise CalcError("`!` 後面缺少可計算的數值或變數。")
    if s[i] == "(":
        j = _match_paren(s, i)
        return _convert_factorial_symbol(s[i:j + 1]), j + 1
    j = i
    while j < n and (s[j].isalnum() or s[j] in "_."):
        j += 1
    token = s[i:j]
    if not token:
        raise CalcError("`!` 後面缺少可計算的數值或變數。")
    if j < n and s[j] == "(" and (token[0].isalpha() or token[0] == "_"):
        k = _match_paren(s, j)  # !sqrt(16) 這類函式呼叫
        return _convert_factorial_symbol(s[i:k + 1]), k + 1
    return token, j


def _convert_factorial_symbol(s: str) -> str:
    """階乘符號轉換：
        x!   → factorial(x)
        x!!  → factorial2(x)       雙階乘
        !x   → subfactorial(x)     子階乘（`!` 在運算元前面時）
    """
    out: List[str] = []
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c != "!":
            out.append(c)
            i += 1
            continue
        if _ends_operand(out):
            double = i + 1 < n and s[i + 1] == "!"
            if double and i + 2 < n and s[i + 2] == "!":
                raise CalcError("不支援連續三個以上的 `!`，請用括號拆開，例如 `(5!!)!`。")
            atom = _pop_postfix_atom(out)
            out.extend(f"{'factorial2' if double else 'factorial'}({atom})")
            i += 2 if double else 1
        else:
            if i + 1 < n and s[i + 1] == "!":
                raise CalcError("`!!` 前面缺少可計算的數值或變數。")
            atom, i = _take_prefix_operand(s, i + 1)
            out.extend(f"subfactorial({atom})")
    return "".join(out)


_HEX_PATTERN = re.compile(r"0[xX][0-9A-Fa-f]+")


def _convert_hex_literals(s: str) -> str:
    return _HEX_PATTERN.sub(lambda m: str(int(m.group(), 16)), s)


_SUPERSCRIPT_MAP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
_SUPERSCRIPT_PATTERN = re.compile(r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+")


def _convert_superscript_exponents(s: str) -> str:
    def repl(m):
        digits = m.group(0).translate(_SUPERSCRIPT_MAP)
        return f"**({digits})"
    return _SUPERSCRIPT_PATTERN.sub(repl, s)


_THOUSANDS_PATTERN = re.compile(r"(?<![\d(,])\d{1,3}(?:,\d{3})+(?:\.\d+)?")


def _strip_thousands