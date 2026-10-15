"""prolog/arith_numba.py — Numba-accelerated arithmetic evaluation.

Designed for low-power hardware (ARM, Raspberry Pi, ROS nodes).

How it works:
  * A Prolog term is compiled into flat numpy arrays ONCE
    (cached by id(term) with a strong reference to the Term itself).
  * The JIT kernel is a non-recursive stack machine. It runs on any
    version of Numba and any ARM variant (including 32-bit ARMv7, if
    Numba still supports the platform).
  * Computations are done in float64; the result is coerced to int if
    all leaves were integers and the value is integral.
  * Errors (unbound variable, division by zero, domain) are raised at
    the Python boundary as PrologError.
  * Any Numba failure (import, JIT, unknown operation) → silent
    fallback to pure Python. The module never breaks the engine.

Hot-path API:
  try_eval_fast(expr, subst) — the single entry point.
    Returns:
      * a number (int/float) — successful computation through JIT;
      * NOT_ATTEMPTED        — Numba not applicable (small tree, disabled,
                               unknown operation, JIT already failed);
      * raises PrologError   — semantic error (unbound variable,
                               division by zero, etc.).

  No RLock/OrderedDict/move_to_end: under the GIL, dict.get and
  dict.__setitem__ are atomic, and on overflow the cache is cleared
  entirely. This eliminates ~400 ns of overhead on every cache hit,
  which on arith_batch_5000 (50,000 calls) makes a noticeable difference.

ARM specifics:
  * The Numba enable threshold is higher on ARM/32-bit (JIT overhead
    is larger).
  * `cache=True` — a subsequent process start does not waste time
    recompiling.
  * `warmup_arith_numba()` — pre-compilation at startup.
  * `fastmath=True` — only via env variable to avoid changing FP.
  * No `prange` — 32-bit ARM does not support it, and for a single tree
    it is not needed anyway.

Environment variables:
  PROLOG_NUMBA_THRESHOLD=N   node-count threshold (default 20/30/40)
  PROLOG_NUMBA_FASTMATH=1    enable fastmath (may change FP results)
  PROLOG_NUMBA_CACHE=N       verdict cache size (default 10000)
  PROLOG_NUMBA_DISABLE=1     disable completely (for debugging)

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""

from __future__ import annotations

import os
import platform
import sys
import threading
from typing import Any, Dict, List, Optional

import numpy as np

from prolog.terms import Term, Variable, Number


# =====================================================================
# Numba availability
# =====================================================================
_NUMBA_AVAILABLE = False
_NUMBA_VERSION: tuple = (0, 0, 0)
try:
    import numba
    from numba import njit
    _NUMBA_VERSION = tuple(int(x) for x in numba.__version__.split('.')[:3])
    _NUMBA_AVAILABLE = True
except ImportError:
    njit = None  # type: ignore


# =====================================================================
# Platform detection
# =====================================================================
_MACHINE = platform.machine().lower()
_IS_ARM = 'arm' in _MACHINE or 'aarch' in _MACHINE
_IS_64BIT = sys.maxsize > 2 ** 32
_IS_CONSTRAINED = _IS_ARM or not _IS_64BIT


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if v is None:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ('1', 'true', 'yes', 'on')


# Node-count threshold: below — Python is faster, above — Numba is faster.
if _IS_CONSTRAINED and not _IS_64BIT:
    _DEFAULT_THRESHOLD = 40         # 32-bit ARM: JIT-call is expensive
elif _IS_CONSTRAINED:
    _DEFAULT_THRESHOLD = 30         # 64-bit ARM
else:
    _DEFAULT_THRESHOLD = 20         # x86_64

_NUMBA_THRESHOLD = _env_int('PROLOG_NUMBA_THRESHOLD', _DEFAULT_THRESHOLD)
_USE_FASTMATH = _env_bool('PROLOG_NUMBA_FASTMATH', False)
_CACHE_SIZE = _env_int('PROLOG_NUMBA_CACHE', 10000)
_DISABLED = _env_bool('PROLOG_NUMBA_DISABLE', False)


# =====================================================================
# Public sentinel and activity flag
# =====================================================================
# A single sentinel instead of None: lets the caller distinguish
# "Numba not applicable" from "result = None" (the latter never happens,
# but an explicit sentinel makes the check `is not NOT_ATTEMPTED` unambiguous).
NOT_ATTEMPTED = object()

# Activity of Numba in a mutable container: allows disabling from the hot path
# without the global keyword and without races (under the GIL, list.__setitem__ is atomic).
_numba_active: List[bool] = [True]


def _deactivate_numba() -> None:
    """Disables Numba for the entire process. Called on the first JIT failure."""
    _numba_active[0] = False


# =====================================================================
# Operation codes
# =====================================================================
OP_NUM = 0
OP_VAR = 1
# binary
OP_ADD = 2
OP_SUB = 3
OP_MUL = 4
OP_DIV = 5
OP_MOD = 6
OP_DIV_INT = 7
OP_MAX = 30
OP_MIN = 31
OP_POW = 32
OP_REM = 33
OP_SHR = 40
OP_SHL = 41
OP_XOR = 42
# unary
OP_NEG = 8
OP_ABS = 10
OP_SIGN = 11
OP_ROUND = 12
OP_FLOOR = 13
OP_CEILING = 14
OP_TRUNCATE = 15
OP_FLOAT = 16
OP_INTEGER = 17
OP_SIN = 20
OP_COS = 21
OP_ATAN = 22
OP_EXP = 23
OP_LOG = 24
OP_SQRT = 25

_FUNCTOR_TO_OP = {
    '+': OP_ADD, '-': OP_SUB, '*': OP_MUL, '/': OP_DIV,
    'mod': OP_MOD, 'div': OP_DIV_INT,
    'abs': OP_ABS, 'sign': OP_SIGN, 'round': OP_ROUND,
    'floor': OP_FLOOR, 'ceiling': OP_CEILING, 'truncate': OP_TRUNCATE,
    'float': OP_FLOAT, 'integer': OP_INTEGER,
    'sin': OP_SIN, 'cos': OP_COS, 'atan': OP_ATAN,
    'exp': OP_EXP, 'log': OP_LOG, 'sqrt': OP_SQRT,
    'max': OP_MAX, 'min': OP_MIN,
    '**': OP_POW, 'pow': OP_POW,
    'rem': OP_REM,
    '>>': OP_SHR, '<<': OP_SHL, 'xor': OP_XOR,
}

_UNARY_OPS = {
    OP_NEG, OP_ABS, OP_SIGN, OP_ROUND, OP_FLOOR, OP_CEILING,
    OP_TRUNCATE, OP_FLOAT, OP_INTEGER,
    OP_SIN, OP_COS, OP_ATAN, OP_EXP, OP_LOG, OP_SQRT,
}
_BINARY_OPS = {
    OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_MOD, OP_DIV_INT,
    OP_MAX, OP_MIN, OP_POW, OP_REM, OP_SHR, OP_SHL, OP_XOR,
}


# =====================================================================
# JIT kernel — non-recursive stack machine
# =====================================================================
if _NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=_USE_FASTMATH)
    def _eval_kernel(ops, a1, a2, nv, vv, root, stack_size):
        """Non-recursive stack machine. See the module docstring."""
        node_stk = np.empty(stack_size, dtype=np.int64)
        val_stk = np.empty(stack_size, dtype=np.float64)
        nsp = 0
        vsp = 0
        node_stk[0] = root << 1
        nsp = 1

        while nsp > 0:
            nsp -= 1
            packed = node_stk[nsp]
            idx = packed >> 1
            action = packed & 1

            if action == 0:
                op = ops[idx]
                if op == 0:
                    val_stk[vsp] = nv[idx]
                    vsp += 1
                    continue
                if op == 1:
                    val_stk[vsp] = vv[a1[idx]]
                    vsp += 1
                    continue
                node_stk[nsp] = (idx << 1) | 1
                nsp += 1
                a2i = a2[idx]
                if a2i >= 0:
                    node_stk[nsp] = (a2i << 1) | 0
                    nsp += 1
                a1i = a1[idx]
                node_stk[nsp] = (a1i << 1) | 0
                nsp += 1
            else:
                op = ops[idx]
                a2i = a2[idx]
                if a2i >= 0:
                    rhs = val_stk[vsp - 1]; vsp -= 1
                    lhs = val_stk[vsp - 1]; vsp -= 1
                    r = np.nan
                    if op == 2:
                        r = lhs + rhs
                    elif op == 3:
                        r = lhs - rhs
                    elif op == 4:
                        r = lhs * rhs
                    elif op == 5:
                        if rhs != 0.0:
                            r = lhs / rhs
                    elif op == 6:
                        if rhs != 0.0:
                            r = lhs - rhs * np.floor(lhs / rhs)
                    elif op == 7:
                        if rhs != 0.0:
                            r = np.floor(lhs / rhs)
                    elif op == 30:
                        r = max(lhs, rhs)
                    elif op == 31:
                        r = min(lhs, rhs)
                    elif op == 32:
                        r = lhs ** rhs
                    elif op == 33:
                        if rhs != 0.0:
                            q = np.trunc(lhs / rhs)
                            r = lhs - rhs * q
                    elif op == 40:
                        r = float(np.int64(lhs) >> np.int64(rhs))
                    elif op == 41:
                        r = float(np.int64(lhs) << np.int64(rhs))
                    elif op == 42:
                        r = float(np.int64(lhs) ^ np.int64(rhs))
                    val_stk[vsp] = r
                    vsp += 1
                else:
                    x = val_stk[vsp - 1]
                    r = np.nan
                    if op == 8:
                        r = -x
                    elif op == 10:
                        r = abs(x)
                    elif op == 11:
                        if x > 0.0:
                            r = 1.0
                        elif x < 0.0:
                            r = -1.0
                        else:
                            r = 0.0
                    elif op == 12:
                        r = np.rint(x)
                    elif op == 13:
                        r = np.floor(x)
                    elif op == 14:
                        r = np.ceil(x)
                    elif op == 15:
                        r = np.trunc(x)
                    elif op == 16:
                        r = x
                    elif op == 17:
                        r = np.trunc(x)
                    elif op == 20:
                        r = np.sin(x)
                    elif op == 21:
                        r = np.cos(x)
                    elif op == 22:
                        r = np.arctan(x)
                    elif op == 23:
                        r = np.exp(x)
                    elif op == 24:
                        if x > 0.0:
                            r = np.log(x)
                    elif op == 25:
                        if x >= 0.0:
                            r = np.sqrt(x)
                    val_stk[vsp - 1] = r

        return val_stk[0]
else:
    def _eval_kernel(*args, **kwargs):  # pragma: no cover
        raise RuntimeError("Numba not available")


# =====================================================================
# Bounded iterative node counting
# =====================================================================
def _count_nodes_bounded(term, limit: int) -> int:
    """Bounded node counting. Stops when count > limit.

    Returns either the exact size (if the tree < limit) or any
    number > limit. Called ONLY on cache miss — once per term.
    """
    count = 0
    stack = [term]
    while stack:
        t = stack.pop()
        count += 1
        if count > limit:
            return count
        if isinstance(t, Term) and t.arguments:
            stack.extend(t.arguments)
    return count


# =====================================================================
# Compilation: Term → flat arrays
# =====================================================================
class _CompiledArith:
    __slots__ = ('ops', 'a1', 'a2', 'nv', 'var_objs', 'root',
                 'n_nodes', 'has_float_leaves')

    def __init__(self, ops, a1, a2, nv, var_objs, root, n_nodes, has_float_leaves):
        self.ops = ops
        self.a1 = a1
        self.a2 = a2
        self.nv = nv
        self.var_objs = var_objs
        self.root = root
        self.n_nodes = n_nodes
        self.has_float_leaves = has_float_leaves


def _compile_term(term: Term) -> _CompiledArith:
    """Decomposes an arithmetic term into flat arrays.

    Raises ValueError/TypeError on unsupported constructs —
    the caller catches and caches None so it is not tried again.
    """
    ops: List[int] = []
    a1: List[int] = []
    a2: List[int] = []
    nv: List[float] = []
    var_objs: List[Variable] = []
    var_slots: Dict[Variable, int] = {}
    has_float_leaves = False

    def add(t):
        nonlocal has_float_leaves
        idx = len(ops)
        ops.append(OP_NUM)
        a1.append(-1)
        a2.append(-1)
        nv.append(0.0)

        if isinstance(t, Number):
            ops[idx] = OP_NUM
            nv[idx] = float(t.value)
            if isinstance(t.value, float):
                has_float_leaves = True
            return idx

        if isinstance(t, Variable):
            if t.name == '_':
                raise ValueError("anonymous variable in arithmetic")
            ops[idx] = OP_VAR
            slot = var_slots.get(t)
            if slot is None:
                slot = len(var_objs)
                var_slots[t] = slot
                var_objs.append(t)
            a1[idx] = slot
            return idx

        if not isinstance(t, Term):
            raise ValueError("unsupported term type: %s" % type(t).__name__)

        f = t.functor
        n_args = len(t.arguments)
        if n_args == 0:
            raise ValueError("bare atom in arithmetic: %r" % (f,))

        if f == '-' and n_args == 1:
            op = OP_NEG
        else:
            op = _FUNCTOR_TO_OP.get(f)
            if op is None:
                raise ValueError("unsupported functor: %r" % (f,))

        if op in _UNARY_OPS:
            if n_args != 1:
                raise ValueError("%r takes 1 arg, got %d" % (f, n_args))
            ops[idx] = op
            a1[idx] = add(t.arguments[0])
            return idx

        if op in _BINARY_OPS:
            if n_args != 2:
                raise ValueError("%r takes 2 args, got %d" % (f, n_args))
            ops[idx] = op
            a1[idx] = add(t.arguments[0])
            a2[idx] = add(t.arguments[1])
            return idx

        raise ValueError("cannot classify functor: %r" % (f,))

    root = add(term)

    return _CompiledArith(
        ops=np.array(ops, dtype=np.int64),
        a1=np.array(a1, dtype=np.int64),
        a2=np.array(a2, dtype=np.int64),
        nv=np.array(nv, dtype=np.float64),
        var_objs=var_objs,
        root=root,
        n_nodes=len(ops),
        has_float_leaves=has_float_leaves,
    )


# =====================================================================
# Single verdict cache (small tree OR compiled object)
# =====================================================================
# id(term) -> (term, _CompiledArith | None)
#
# No RLock, no OrderedDict, no move_to_end. Under the GIL, dict.get and
# dict.__setitem__ are atomic. On overflow the cache is cleared entirely —
# a simple eviction, acceptable for an LRU-like scenario.
#
# A strong reference to term in the value keeps the object alive, so the
# id cannot be reused by another object while the entry is in the cache.
_fast_cache: Dict[int, tuple] = {}


def _get_or_compile(term) -> Optional[_CompiledArith]:
    """Returns the compiled tree, or None (small / non-compilable).

    On the first call for a given id, it performs one traversal (pre-check)
    and, if needed, compilation. The result is cached — including the None
    verdict for small trees, so they are not traversed again.
    """
    tid = id(term)
    entry = _fast_cache.get(tid)
    if entry is not None:
        return entry[1]

    n = _count_nodes_bounded(term, _NUMBA_THRESHOLD)
    if n < _NUMBA_THRESHOLD:
        compiled: Optional[_CompiledArith] = None
    else:
        try:
            compiled = _compile_term(term)
        except (ValueError, TypeError):
            compiled = None

    if len(_fast_cache) >= _CACHE_SIZE:
        _fast_cache.clear()
    _fast_cache[tid] = (term, compiled)
    return compiled


# =====================================================================
# Runtime variable resolution
# =====================================================================
def _resolve_vars(var_objs: List[Variable], subst: Dict) -> np.ndarray:
    """Builds the float64 array of variable values.

    An unbound variable or a variable bound to a non-Number raises
    PrologError.
    """
    from prolog.predicates import PrologError, evaluate_expression

    n = len(var_objs)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    out = np.empty(n, dtype=np.float64)
    for i, var in enumerate(var_objs):
        v = subst.get(var)
        if v is None:
            raise PrologError('instantiation_error', 'variable not bound', var)
        hops = 0
        while isinstance(v, Variable):
            hops += 1
            if hops > 10_000:
                raise PrologError('type_error', 'evaluable', var)
            nxt = subst.get(v)
            if nxt is None:
                raise PrologError('instantiation_error', 'variable not bound', v)
            v = nxt
        if isinstance(v, Number):
            out[i] = float(v.value)
            continue
        result = evaluate_expression(v, subst)
        if not isinstance(result, (int, float)):
            raise PrologError('type_error', 'evaluable', v)
        out[i] = float(result)
    return out


# =====================================================================
# Low-level evaluation of a compiled tree
# =====================================================================
def _eval_compiled(compiled: _CompiledArith, subst: Dict, expr):
    """Evaluates a precompiled tree.

    Assumes the caller already checked:
      * compiled is not None,
      * Numba is active.

    Returns a number (int/float).
    Raises PrologError on semantic error.
    On JIT failure — deactivates Numba and raises RuntimeError (the caller
    swallows it and falls back to the Python path).
    """
    # Variable resolution — may raise PrologError, that is fine
    vv = _resolve_vars(compiled.var_objs, subst)

    stack_size = compiled.n_nodes + 10
    try:
        result = _eval_kernel(
            compiled.ops, compiled.a1, compiled.a2,
            compiled.nv, vv, compiled.root, stack_size,
        )
    except Exception as e:
        _deactivate_numba()
        raise RuntimeError("JIT kernel failed: %s" % (e,))

    # NaN — error marker (division by 0, log<0, sqrt<0, etc.)
    if result != result:
        from prolog.predicates import PrologError
        raise PrologError('evaluation_error', 'undefined_result', expr)

    if compiled.has_float_leaves:
        return float(result)
    if float(result).is_integer():
        return int(result)
    return float(result)


# =====================================================================
# Hot-path API
# =====================================================================
def try_eval_fast(expr, subst):
    """The single entry point for the fast path.

    Returns:
      * a number (int/float) — successful computation through JIT;
      * NOT_ATTEMPTED        — Numba not applicable (small tree,
                               disabled, JIT already failed, unknown op);
      * raises PrologError   — semantic error (unbound variable,
                               division by zero, etc.).

    Logic:
      1. Fast activity-flag check — a single list-index.
      2. dict.get(id(expr)) — O(1).
      3. If an entry exists and compiled is not None → JIT directly.
         If it exists and compiled is None → NOT_ATTEMPTED (small tree).
         If no entry → _get_or_compile (one traversal + optional
         compilation) → JIT or NOT_ATTEMPTED.
    """
    if not _numba_active[0] or _DISABLED:
        return NOT_ATTEMPTED

    tid = id(expr)
    entry = _fast_cache.get(tid)

    if entry is not None:
        compiled = entry[1]
        if compiled is None:
            return NOT_ATTEMPTED
        return _eval_compiled(compiled, subst, expr)

    # Cache miss — full path
    compiled = _get_or_compile(expr)
    if compiled is None:
        return NOT_ATTEMPTED
    return _eval_compiled(compiled, subst, expr)


# Compatibility with the old name (in case anyone still imports it)
def evaluate_expression_numba(expr, subst):
    """Deprecated wrapper. Use try_eval_fast."""
    r = try_eval_fast(expr, subst)
    if r is NOT_ATTEMPTED:
        return None
    return r


# =====================================================================
# Warmup and diagnostics
# =====================================================================
def warmup_arith_numba() -> bool:
    """Warms up the JIT kernel so that the first real query does not "hang".

    It is recommended to call this once at application startup — especially
    on ARM, where the first JIT compilation may take several seconds.
    """
    if _DISABLED or not _NUMBA_AVAILABLE or not _numba_active[0]:
        return False

    # Binary branch: 1 + 1
    ops = np.array([OP_NUM, OP_NUM, OP_ADD], dtype=np.int64)
    a1 = np.array([-1, -1, 0], dtype=np.int64)
    a2 = np.array([-1, -1, 1], dtype=np.int64)
    nv = np.array([1.0, 1.0, 0.0], dtype=np.float64)
    vv = np.empty(0, dtype=np.float64)
    try:
        _eval_kernel(ops, a1, a2, nv, vv, 2, 8)
    except Exception:
        _deactivate_numba()
        return False

    # Unary branch
    ops = np.array([OP_NUM, OP_NEG], dtype=np.int64)
    a1 = np.array([-1, 0], dtype=np.int64)
    a2 = np.array([-1, -1], dtype=np.int64)
    nv = np.array([1.0, 0.0], dtype=np.float64)
    try:
        _eval_kernel(ops, a1, a2, nv, vv, 1, 4)
    except Exception:
        pass

    return True


def arith_numba_status() -> Dict[str, Any]:
    """Snapshot of the current state of the Numba path (for diagnostics)."""
    return {
        'available': bool(_NUMBA_AVAILABLE),
        'active': bool(_numba_active[0]),
        'numba_version': _NUMBA_VERSION,
        'platform': _MACHINE,
        'is_arm': _IS_ARM,
        'is_64bit': _IS_64BIT,
        'threshold': _NUMBA_THRESHOLD,
        'fastmath': _USE_FASTMATH,
        'cache_size': len(_fast_cache),
        'cache_capacity': _CACHE_SIZE,
        'disabled': _DISABLED,
    }


def clear_arith_cache() -> None:
    _fast_cache.clear()


__all__ = [
    'try_eval_fast',
    'evaluate_expression_numba',   # backward-compat
    'warmup_arith_numba',
    'arith_numba_status',
    'clear_arith_cache',
    'NOT_ATTEMPTED',
]