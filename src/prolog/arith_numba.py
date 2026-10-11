"""prolog/arith_numba.py — Numba-accelerated arithmetic evaluation.

Рассчитано на слабое железо (ARM, Raspberry Pi, ROS-ноды).

Принцип работы:
  * Терм Prolog компилируется в плоские numpy-массивы ОДИН раз
    (кэш по id(term) с сильной ссылкой на сам Term).
  * JIT-ядро — нерекурсивная стек-машина. Работает на любых версиях
    Numba и любых ARM-вариантах (включая 32-битный ARMv7, если Numba
    ещё поддерживает платформу).
  * Вычисления идут во float64; результат приводится к int, если все
    листья были целочисленными и значение целое.
  * Ошибки (незаполненная переменная, деление на ноль, домен) поднимаются
    на Python-границе как PrologError.
  * Любой сбой Numba (импорт, JIT, неизвестная операция) → тихий
    fallback на чистый Python. Модуль никогда не ломает движок.

Hot-path API:
  try_eval_fast(expr, subst) — единая точка входа.
    Возвращает:
      * число (int/float)     — успешное вычисление через JIT;
      * NOT_ATTEMPTED         — Numba не применима (мелкое дерево, отключена,
                                неизвестная операция, JIT уже падал);
      * бросает PrologError   — семантическая ошибка (незаполненная
                                переменная, деление на ноль и т.п.).

  Никаких RLock/OrderedDict/move_to_end: под GIL dict.get и dict.__setitem__
  атомарны, а при переполнении кэш очищается целиком. Это устраняет
  ~400 нс overhead на каждый cache hit, что на arith_batch_5000 (50 000
  вызовов) даёт заметную разницу.

ARM-специфика:
  * Порог включения Numba выше на ARM/32-бит (JIT-overhead больше).
  * `cache=True` — повторный старт процесса не тратит время на компиляцию.
  * `warmup_arith_numba()` — пред-компиляция на старте.
  * `fastmath=True` — только по env-переменной, чтобы не менять FP.
  * Никакого `prange` — 32-битный ARM его не поддерживает, а для одного
    дерева он и не нужен.

Переменные окружения:
  PROLOG_NUMBA_THRESHOLD=N   порог по числу узлов (по умолчанию 20/30/40)
  PROLOG_NUMBA_FASTMATH=1    включить fastmath (может менять FP-результаты)
  PROLOG_NUMBA_CACHE=N       размер кэша вердиктов (по умолчанию 10000)
  PROLOG_NUMBA_DISABLE=1     полностью отключить (для отладки)

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
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
# Доступность Numba
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
# Определение платформы
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


# Порог по числу узлов: ниже — Python быстрее, выше — Numba быстрее.
if _IS_CONSTRAINED and not _IS_64BIT:
    _DEFAULT_THRESHOLD = 40         # 32-bit ARM: JIT-call дорогой
elif _IS_CONSTRAINED:
    _DEFAULT_THRESHOLD = 30         # 64-bit ARM
else:
    _DEFAULT_THRESHOLD = 20         # x86_64

_NUMBA_THRESHOLD = _env_int('PROLOG_NUMBA_THRESHOLD', _DEFAULT_THRESHOLD)
_USE_FASTMATH = _env_bool('PROLOG_NUMBA_FASTMATH', False)
_CACHE_SIZE = _env_int('PROLOG_NUMBA_CACHE', 10000)
_DISABLED = _env_bool('PROLOG_NUMBA_DISABLE', False)


# =====================================================================
# Публичные sentinel и флаг активности
# =====================================================================
# Единый sentinel вместо None: позволяет вызывающей стороне отличить
# «Numba не применима» от «результат = None» (последнего не бывает, но
# явный sentinel делает проверку `is not NOT_ATTEMPTED` однозначной).
NOT_ATTEMPTED = object()

# Активность Numba в изменяемом контейнере: позволяет отключать из hot path
# без ключевого слова global и без гонок (под GIL list.__setitem__ атомарен).
_numba_active: List[bool] = [True]


def _deactivate_numba() -> None:
    """Отключает Numba на весь процесс. Вызывается при первом сбое JIT."""
    _numba_active[0] = False


# =====================================================================
# Коды операций
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
# JIT-ядро — нерекурсивная стек-машина
# =====================================================================
if _NUMBA_AVAILABLE:
    @njit(cache=True, nogil=True, fastmath=_USE_FASTMATH)
    def _eval_kernel(ops, a1, a2, nv, vv, root, stack_size):
        """Нерекурсивная стек-машина. См. описание в докстринге модуля."""
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
# Ограниченный итеративный подсчёт узлов
# =====================================================================
def _count_nodes_bounded(term, limit: int) -> int:
    """Ограниченный подсчёт узлов. Останавливается при count > limit.

    Возвращает либо точный размер (если дерево < limit), либо любое
    число > limit. Вызывается ТОЛЬКО на cache miss — один раз на терм.
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
# Компиляция Term → плоские массивы
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
    """Разбирает арифметический терм в плоские массивы.

    Бросает ValueError/TypeError на неподдерживаемых конструкциях —
    вызывающая сторона ловит и кэширует None, чтобы больше не пытаться.
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
# Единый кэш вердиктов (маленькое дерево ИЛИ скомпилированный объект)
# =====================================================================
# id(term) -> (term, _CompiledArith | None)
#
# Без RLock, без OrderedDict, без move_to_end. Под GIL dict.get и
# dict.__setitem__ атомарны. При переполнении кэш очищается целиком —
# простое вытеснение, приемлемое для LRU-подобного сценария.
#
# Сильная ссылка на term в значении удерживает объект живым, поэтому
# id не может быть переиспользован другим объектом, пока запись в кэше.
_fast_cache: Dict[int, tuple] = {}


def _get_or_compile(term) -> Optional[_CompiledArith]:
    """Возвращает скомпилированное дерево или None (мелкое / некомпилируемое).

    На первом вызове для данного id делает один обход (pre-check) и,
    если нужно, компиляцию. Результат кэшируется — включая вердикт None
    для мелких деревьев, чтобы не обходить их повторно.
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
# Резолв переменных в рантайме
# =====================================================================
def _resolve_vars(var_objs: List[Variable], subst: Dict) -> np.ndarray:
    """Строит float64-массив значений переменных.

    Незаполненная переменная или переменная, связанная с не-Number,
    приводит к PrologError.
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
# Низкоуровневое вычисление скомпилированного дерева
# =====================================================================
def _eval_compiled(compiled: _CompiledArith, subst: Dict, expr):
    """Вычисляет заранее скомпилированное дерево.

    Предполагается, что вызывающая сторона уже проверила:
      * compiled is not None,
      * Numba активна.

    Возвращает число (int/float).
    Бросает PrologError при семантической ошибке.
    При сбое JIT — деактивирует Numba и бросает RuntimeError (его
    вызывающая сторона проглотит и уйдёт в Python-путь).
    """
    # Резолв переменных — может бросить PrologError, это нормально
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

    # NaN — маркер ошибки (деление на 0, log<0, sqrt<0 и т.п.)
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
    """Единая точка входа для быстрого пути.

    Возвращает:
      * число (int/float)   — успешное вычисление через JIT;
      * NOT_ATTEMPTED       — Numba не применима (мелкое дерево,
                              отключена, JIT уже падал, неизвестная операция);
      * бросает PrologError — семантическая ошибка (незаполненная
                              переменная, деление на ноль и т.п.).

    Логика:
      1. Быстрая проверка флага активности — один list-index.
      2. dict.get(id(expr)) — O(1).
      3. Если запись есть и compiled is not None → сразу JIT.
         Если есть и compiled is None → NOT_ATTEMPTED (мелкое дерево).
         Если записи нет → _get_or_compile (один обход + опциональная
         компиляция) → JIT или NOT_ATTEMPTED.
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

    # Cache miss — полный путь
    compiled = _get_or_compile(expr)
    if compiled is None:
        return NOT_ATTEMPTED
    return _eval_compiled(compiled, subst, expr)


# Совместимость со старым именем (на случай, если кто-то ещё его импортирует)
def evaluate_expression_numba(expr, subst):
    """Устаревшая обёртка. Используйте try_eval_fast."""
    r = try_eval_fast(expr, subst)
    if r is NOT_ATTEMPTED:
        return None
    return r


# =====================================================================
# Прогрев и диагностика
# =====================================================================
def warmup_arith_numba() -> bool:
    """Прогревает JIT-ядро, чтобы первый реальный запрос не «залипал».

    Рекомендуется вызвать один раз на старте приложения — особенно на ARM,
    где первая JIT-компиляция может занять несколько секунд.
    """
    if _DISABLED or not _NUMBA_AVAILABLE or not _numba_active[0]:
        return False

    # Бинарная ветка: 1 + 1
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

    # Унарная ветка
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
    """Снимок текущего состояния Numba-пути (для диагностики)."""
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