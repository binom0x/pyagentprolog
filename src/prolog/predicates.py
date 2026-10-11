""" PyAgentProlog

Модуль для встроенных предикатов (оптимизирован: все предикаты возвращают генераторы,
убраны лишние копирования подстановок).

Добавлены предикаты управления семантической унификацией:
    semantic_on/0, semantic_on/1, semantic_off/0,
    semantic_threshold/1, semantic_status/1.

Арифметика evaluate_expression теперь имеет быстрый путь через Numba
(см. prolog/arith_numba.py). Hot-path вызов идёт через try_eval_fast,
который на cache hit возвращает результат или sentinel NOT_ATTEMPTED
за один dict lookup. Если Numba недоступна или выражение слишком
маленькое — используется чистый Python-обход.

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1 (findall/bagof/setof больше не глотают исключения)
"""

import math
from typing import List, Dict, Any, Optional, Callable, Set, Tuple, Iterator, Generator
from collections.abc import Iterable
from prolog.terms import Term, Variable, Number

# Для аннотаций
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from prolog.interpreter import Database


# =====================================================================
# Опциональное Numba-ускорение арифметики.
# Модуль arith_numba.py сам решает, доступна ли Numba и подходит ли
# платформа. Если что-то не так — try_eval_fast всегда вернёт sentinel
# NOT_ATTEMPTED, и мы спокойно уйдём в Python-путь.
# =====================================================================
try:
    from prolog.arith_numba import (
        try_eval_fast as _try_eval_fast,
        NOT_ATTEMPTED as _NUMBA_NOT_ATTEMPTED,
        warmup_arith_numba as _warmup_numba,
        arith_numba_status as _numba_status,
    )
    _ARITH_NUMBA_AVAILABLE = True
except Exception:
    _ARITH_NUMBA_AVAILABLE = False
    _NUMBA_NOT_ATTEMPTED = object()

    def _try_eval_fast(expr, subst):
        return _NUMBA_NOT_ATTEMPTED

    def _warmup_numba():
        return False

    def _numba_status():
        return {'available': False, 'active': False}


# ----------------------------------------------------------------------
# Вспомогательная функция для печати непойманных исключений из
# мета-предикатов (findall/bagof/setof/call/once/forall).
# ----------------------------------------------------------------------
def _reraise_unexpected(where: str, goal_repr, exc: BaseException) -> None:
    """Печатает traceback и пробрасывает исключение дальше.

    Используется в мета-предикатах, чтобы не превращать настоящую ошибку
    (TypeError из Python-действия, AttributeError и т.п.) в тихий fail.
    """
    import traceback
    print(f"[{where}] unexpected {type(exc).__name__} in Goal={goal_repr}: {exc}")
    traceback.print_exc()
    raise


class Predicate:
    """Базовый класс для предикатов"""
    __slots__ = ('name', 'arity', 'func')
    
    def __init__(self, name: str, arity: int, func: Optional[Callable] = None):
        self.name = name
        self.arity = arity
        self.func = func
    
    def execute(self, goal: Term, database, subst: Dict, depth: int) -> Iterator[Dict]:
        """Выполнение предиката (возвращает генератор)"""
        if self.func:
            result = self.func(goal, database, subst, depth)
            if result is None:
                return
            # Если result - итерируемый объект (кроме строк и термов), то выдаём элементы
            if isinstance(result, (list, tuple)):
                yield from result
            elif isinstance(result, Iterable) and not isinstance(result, (str, Term)):
                yield from result
            else:
                yield result
        return

    def __str__(self):
        return f"{self.name}/{self.arity}"


# ----------------------------------------------------------------------
# Декоратор для регистрации пользовательских предикатов
# ----------------------------------------------------------------------
def prolog_predicate(name: str, arity: int):
    def decorator(func):
        func._is_prolog_predicate = True
        func._predicate_name = name
        func._predicate_arity = arity
        return func
    return decorator


# ----------------------------------------------------------------------
# Исключения
# ----------------------------------------------------------------------
class RepeatException(Exception):
    """Исключение для реализации repeat/0"""
    def __init__(self, subst):
        self.subst = subst
        super().__init__(subst)


class CutException(Exception):
    """Исключение, возникающее при выполнении отсечения (!). Содержит подстановку."""
    def __init__(self, subst):
        self.subst = subst
        super().__init__(subst)


class PrologError(Exception):
    """Исключение для ошибок выполнения Prolog (ISO style)."""
    def __init__(self, error_type, detail, term=None):
        self.error_type = error_type
        self.detail = detail
        self.term = term
        super().__init__(f"{error_type}: {detail}" + (f" in term {term}" if term else ""))


# ----------------------------------------------------------------------
# Вспомогательные функции
# ----------------------------------------------------------------------
def _safe_math(func, *args, term=None):
    """Обёртка для безопасного вычисления математических функций с генерацией PrologError."""
    try:
        result = func(*args)
        if isinstance(result, complex):
            raise PrologError('evaluation_error', 'complex result not allowed', term)
        return result
    except (ValueError, ZeroDivisionError, ArithmeticError) as e:
        if isinstance(e, ZeroDivisionError):
            raise PrologError('evaluation_error', 'zero_divisor', term)
        else:
            raise PrologError('evaluation_error', str(e), term)


def evaluate_expression(expr: Term, subst: Dict) -> Any:
    """Вычисление арифметического выражения.

    Быстрый путь: если размер дерева достаточен и Numba активна — дерево
    компилируется один раз в плоские numpy-массивы, дальше вычисление идёт
    в JIT-ядре (нерекурсивная стек-машина, работает и на ARM).

    Hot-path вызов идёт через try_eval_fast:
      * на cache hit — один dict lookup;
      * на cache miss — pre-check + опциональная компиляция;
      * sentinel NOT_ATTEMPTED означает «иди в Python-путь».

    Малые деревья и все случаи сбоя Numba идут через чистый Python-обход,
    который сохраняет точное поведение (включая тип ошибок).
    """
    # --- Быстрый путь через Numba (только для больших деревьев) ---
    if _ARITH_NUMBA_AVAILABLE:
        try:
            fast = _try_eval_fast(expr, subst)
            if fast is not _NUMBA_NOT_ATTEMPTED:
                return fast
        except PrologError:
            raise
        except Exception:
            # Любой непредвиденный сбой Numba — тихо уходим в Python.
            pass

    # --- Чистый Python-путь ---
    if isinstance(expr, Number):
        return expr.value

    if isinstance(expr, Variable):
        if expr in subst:
            value = subst[expr]
            return evaluate_expression(value, {})
        else:
            raise PrologError('instantiation_error', 'variable not bound', expr)

    if isinstance(expr, Term) and expr.functor in ['+', '-', '*', '/', 'mod', 'div']:
        if len(expr.arguments) != 2:
            raise PrologError('type_error', 'evaluable', expr)

        left = evaluate_expression(expr.arguments[0], subst)
        right = evaluate_expression(expr.arguments[1], subst)

        if expr.functor == '+':
            return left + right
        elif expr.functor == '-':
            return left - right
        elif expr.functor == '*':
            return left * right
        elif expr.functor == '/':
            if right == 0:
                raise PrologError('evaluation_error', 'zero_divisor', expr)
            return left / right
        elif expr.functor == 'mod':
            if right == 0:
                raise PrologError('evaluation_error', 'zero_divisor', expr)
            return (left % right + right) % right
        elif expr.functor == 'div':
            if right == 0:
                raise PrologError('evaluation_error', 'zero_divisor', expr)
            return left // right

    if isinstance(expr, Term) and expr.functor == '-' and len(expr.arguments) == 1:
        arg = evaluate_expression(expr.arguments[0], subst)
        return -arg

    # Унарные арифметические функции
    UNARY = {
        'abs': lambda x: abs(x),
        'sign': lambda x: -1 if x < 0 else (1 if x > 0 else 0),
        'round': lambda x: round(x),
        'floor': lambda x: math.floor(x),
        'ceiling': lambda x: math.ceil(x),
        'truncate': lambda x: int(x) if x >= 0 else -int(-x),
        'float': lambda x: float(x),
        'integer': lambda x: int(x),
        'sin': lambda x: math.sin(x),
        'cos': lambda x: math.cos(x),
        'atan': lambda x: math.atan(x),
        'exp': lambda x: math.exp(x),
        'log': lambda x: math.log(x),
        'sqrt': lambda x: math.sqrt(x),
        '-': lambda x: -x,
    }
    if expr.functor in UNARY:
        if len(expr.arguments) == 1:
            arg = evaluate_expression(expr.arguments[0], subst)
            return _safe_math(UNARY[expr.functor], arg, term=expr)

    # Бинарные арифметические функции
    BINARY = {
        'max': lambda x, y: max(x, y),
        'min': lambda x, y: min(x, y),
        '**': lambda x, y: x ** y,
        'pow': lambda x, y: x ** y,
        'rem': lambda x, y: x - y * int(x / y) if y != 0 else 0,
        'mod': lambda x, y: x - y * math.floor(x / y) if y != 0 else 0,
        '//': lambda x, y: int(x / y),
        '>>': lambda x, y: x >> y,
        '<<': lambda x, y: x << y,
        'xor': lambda x, y: x ^ y,
    }
    if expr.functor in BINARY:
        if len(expr.arguments) == 2:
            left = evaluate_expression(expr.arguments[0], subst)
            right = evaluate_expression(expr.arguments[1], subst)
            return _safe_math(BINARY[expr.functor], left, right, term=expr)

    if isinstance(expr, Term) and not expr.arguments:
        try:
            functor_str = str(expr.functor)
            if (functor_str.startswith("'") and functor_str.endswith("'")) or \
               (functor_str.startswith('"') and functor_str.endswith('"')):
                functor_str = functor_str[1:-1]
            if '.' in functor_str or 'e' in functor_str.lower():
                return float(functor_str)
            else:
                return int(functor_str)
        except Exception:
            raise PrologError('type_error', 'evaluable', expr)

    raise PrologError('type_error', 'evaluable', expr)


def get_number_value(term: Term, subst: Dict) -> Optional[float]:
    return evaluate_expression(term, subst)


def _term_to_rule(term: Term):
    from prolog.terms import Rule
    if isinstance(term, Term) and term.functor == ':-' and len(term.arguments) == 2:
        return Rule(term.arguments[0], term.arguments[1])
    else:
        return Rule(term, Term('true', []))


def _list_to_python(lst_term: Term):
    """Преобразует Prolog-список в Python-список термов (включая переменные)."""
    elements = []
    current = lst_term
    while isinstance(current, Term) and current.functor == '.' and len(current.arguments) == 2:
        elements.append(current.arguments[0])
        current = current.arguments[1]
    if isinstance(current, Term) and current.functor == '[]' and len(current.arguments) == 0:
        return elements
    else:
        return None


def _build_prolog_list(items: List[Term]) -> Term:
    """Строит Prolog-список из списка термов."""
    result = Term('[]')
    for item in reversed(items):
        result = Term('.', [item, result])
    return result


def _term_to_string(term: Term) -> str:
    """Извлекает строковое содержимое из Prolog-терма (атома или строки в кавычках)."""
    if isinstance(term, Number):
        return str(term.value)
    if isinstance(term, Term) and not term.arguments:
        functor = term.functor
        if (functor.startswith('"') and functor.endswith('"')) or \
           (functor.startswith("'") and functor.endswith("'")):
            return functor[1:-1]
        return functor
    return str(term)


def _string_to_term(s: str, quoted: bool = True) -> Term:
    """Преобразует Python-строку в Prolog-терм (строку в двойных кавычках или атом)."""
    if quoted:
        return Term(f'"{s}"')
    else:
        return Term(s)


def _collect_variables(term: Term) -> Set[Variable]:
    """Собирает все переменные в терме (кроме анонимных)."""
    vars_set = set()
    def collect(t):
        if isinstance(t, Variable):
            if t.name != "_":
                vars_set.add(t)
        elif isinstance(t, Term):
            for arg in t.arguments:
                collect(arg)
    collect(term)
    return vars_set


def _extract_existential(goal: Term, subst: Dict) -> Tuple[Set[Variable], Term]:
    """Разбирает терм, извлекая экзистенциальные переменные '^'."""
    existential = set()
    current = goal
    while isinstance(current, Term) and current.functor == '^' and len(current.arguments) == 2:
        left = current.arguments[0]
        right = current.arguments[1]
        left_var = left.substitute(subst)
        if isinstance(left_var, Variable):
            existential.add(left_var)
        elif isinstance(left_var, Term):
            existential.update(_collect_variables(left_var))
        current = right
    return existential, current


def _is_ground(term: Term, subst: Dict) -> bool:
    """Проверяет, не содержит ли терм переменных (полностью связан)."""
    term = term.substitute(subst)
    if isinstance(term, Variable):
        return False
    if isinstance(term, Term):
        return all(_is_ground(arg, subst) for arg in term.arguments)
    return True


def is_identical(term1, term2):
    """Проверяет, являются ли два терма идентичными (без подстановки)."""
    if term1 is term2:
        return True
    if isinstance(term1, Variable) and isinstance(term2, Variable):
        return term1.name == term2.name
    if isinstance(term1, Number) and isinstance(term2, Number):
        return term1.value == term2.value
    if isinstance(term1, Term) and isinstance(term2, Term):
        if term1.functor != term2.functor:
            return False
        if len(term1.arguments) != len(term2.arguments):
            return False
        return all(is_identical(arg1, arg2) for arg1, arg2 in zip(term1.arguments, term2.arguments))
    return False


def standard_order(term1, term2):
    """Стандартный порядок термов."""
    if isinstance(term1, Variable) and isinstance(term2, Variable):
        if term1.name < term2.name:
            return -1
        elif term1.name > term2.name:
            return 1
        else:
            return 0
    elif isinstance(term1, Variable):
        return -1
    elif isinstance(term2, Variable):
        return 1

    if isinstance(term1, Number) and isinstance(term2, Number):
        if term1.value < term2.value:
            return -1
        elif term1.value > term2.value:
            return 1
        else:
            return 0
    if isinstance(term1, Number):
        return -1
    if isinstance(term2, Number):
        return 1

    if isinstance(term1, Term) and isinstance(term2, Term):
        arity1 = len(term1.arguments)
        arity2 = len(term2.arguments)
        if arity1 < arity2:
            return -1
        elif arity1 > arity2:
            return 1
        if term1.functor < term2.functor:
            return -1
        elif term1.functor > term2.functor:
            return 1
        for arg1, arg2 in zip(term1.arguments, term2.arguments):
            cmp = standard_order(arg1, arg2)
            if cmp != 0:
                return cmp
        return 0
    return 0


# ======================================================================
# ВСТРОЕННЫЕ ПРЕДИКАТЫ
# ======================================================================

# ----------------------------------------------------------------------
# Арифметические и сравнивающие предикаты
# ----------------------------------------------------------------------
@prolog_predicate('is', 2)
def builtin_is(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """X is Expression"""
    if len(goal.arguments) != 2:
        return
    left = goal.arguments[0]
    right = goal.arguments[1]
    try:
        value = evaluate_expression(right, subst)
    except PrologError:
        return
    if value is None:
        return
    result = Number(value)
    new_subst = left.unify(result, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('>', 2)
def builtin_gt(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x > y:
        yield subst.copy()


@prolog_predicate('<', 2)
def builtin_lt(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x < y:
        yield subst.copy()


@prolog_predicate('>=', 2)
def builtin_ge(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x >= y:
        yield subst.copy()


@prolog_predicate('=<', 2)
def builtin_le(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x <= y:
        yield subst.copy()


@prolog_predicate('=:=', 2)
def builtin_eq(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x == y:
        yield subst.copy()


@prolog_predicate(r'=\=', 2)
def builtin_ne(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    try:
        x = get_number_value(goal.arguments[0], subst)
        y = get_number_value(goal.arguments[1], subst)
    except PrologError:
        return
    if x is not None and y is not None and x != y:
        yield subst.copy()


@prolog_predicate('=', 2)
def builtin_unify(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    new_subst = goal.arguments[0].unify(goal.arguments[1], subst.copy(), depth=depth)
    if new_subst is not None:
        yield new_subst


# ----------------------------------------------------------------------
# Предикаты работы со списками
# ----------------------------------------------------------------------
@prolog_predicate('member', 2)
def builtin_member(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """member(X, List) — линейный итеративный обход без substitute на каждом шаге."""
    if len(goal.arguments) != 2:
        return
    elem = goal.arguments[0]
    current = goal.arguments[1]
    open_counter = 0

    while True:
        # разыменовываем цепочку переменных в текущей ячейке
        while isinstance(current, Variable) and current in subst:
            current = subst[current]

        if isinstance(current, Variable):
            # открытый хвост: одно решение — продлить списком [elem | _]
            open_counter += 1
            tail = Variable(f'_member_open_{open_counter}')
            new_cons = Term('.', [elem, tail])
            new_subst = current.unify(new_cons, subst.copy(), depth=depth)
            if new_subst is not None:
                yield new_subst
            return

        if not isinstance(current, Term):
            return
        if current.functor == '[]' and not current.arguments:
            return
        if current.functor == '.' and len(current.arguments) == 2:
            head, tail = current.arguments
            # unify сам разыменует elem и head через subst
            new_subst = elem.unify(head, subst.copy(), depth=depth)
            if new_subst is not None:
                yield new_subst
            current = tail
            continue
        return


@prolog_predicate('append', 3)
def builtin_append(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """append(A, B, C).

    Реализация повторяет реляционную семантику стандартного append/3
    (clause-1: append([], L, L); clause-2: append([H|T], L, [H|R]) :- append(T, L, R)),
    но без substitute() на каждом шаге и без рекурсии.

    Режимы:
      * A ground           → единственное решение C = A ++ B.
      * C ground, A свобод. → |C|+1 решений (перебор расщеплений C = A ++ B).
      * C открыт, A свобод. → одно решение: A := [], B := C' (где C' — остаток C).
    """
    if len(goal.arguments) != 3:
        return
    a, b, c = goal.arguments

    # --- Обход A: собрать heads и определить, замкнут ли список ---
    a_elems: List[Term] = []
    a_open = None
    current = a
    while True:
        while isinstance(current, Variable) and current in subst:
            current = subst[current]
        if isinstance(current, Variable):
            a_open = current
            break
        if not isinstance(current, Term):
            return
        if current.functor == '[]' and not current.arguments:
            break
        if current.functor == '.' and len(current.arguments) == 2:
            a_elems.append(current.arguments[0])
            current = current.arguments[1]
            continue
        return

    # --- Режим 1: A полностью замкнут ---
    if a_open is None:
        # C = A ++ B, единственное решение.
        chain = b
        for elem in reversed(a_elems):
            chain = Term('.', [elem, chain])
        new_subst = c.unify(chain, subst.copy(), depth=depth)
        if new_subst is not None:
            yield new_subst
        return

    # --- Обход C: собрать heads и определить, замкнут ли список ---
    c_elems: List[Term] = []
    c_open = None
    current = c
    while True:
        while isinstance(current, Variable) and current in subst:
            current = subst[current]
        if isinstance(current, Variable):
            c_open = current
            break
        if not isinstance(current, Term):
            return
        if current.functor == '[]' and not current.arguments:
            break
        if current.functor == '.' and len(current.arguments) == 2:
            c_elems.append(current.arguments[0])
            current = current.arguments[1]
            continue
        return

    # --- Режим 2: C полностью замкнут → перебираем расщепления ---
    if c_open is None:
        if len(c_elems) < len(a_elems):
            return  # C короче известного префикса A — расщеплений нет
        # Известный префикс A должен совпасть с началом C
        prefix_subst = subst.copy()
        for ae, ce in zip(a_elems, c_elems[:len(a_elems)]):
            prefix_subst = ae.unify(ce, prefix_subst, depth=depth)
            if prefix_subst is None:
                return
        # k — сколько элементов C отдано под A (k >= len(a_elems))
        for k in range(len(a_elems), len(c_elems) + 1):
            # a_open := C[len(a_elems):k]
            a_tail = Term('[]')
            for elem in reversed(c_elems[len(a_elems):k]):
                a_tail = Term('.', [elem, a_tail])
            # B := C[k:]
            b_val = Term('[]')
            for elem in reversed(c_elems[k:]):
                b_val = Term('.', [elem, b_val])

            cur = prefix_subst
            cur = a_open.unify(a_tail, cur, depth=depth)
            if cur is None:
                continue
            cur = b.unify(b_val, cur, depth=depth)
            if cur is not None:
                yield cur
        return

    # --- Режим 3: C открыт → A должен быть замкнут (a_open := []) ---
    # B получается «достройкой» C до корректного списка.
    if len(a_elems) <= len(c_elems):
        # Известный префикс A укладывается в известный префикс C.
        cur = subst.copy()
        for ae, ce in zip(a_elems, c_elems[:len(a_elems)]):
            cur = ae.unify(ce, cur, depth=depth)
            if cur is None:
                return
        cur = a_open.unify(Term('[]'), cur, depth=depth)
        if cur is None:
            return
        # B := C[len(a_elems):] ++ c_open
        b_val = c_open
        for elem in reversed(c_elems[len(a_elems):]):
            b_val = Term('.', [elem, b_val])
        cur = b.unify(b_val, cur, depth=depth)
        if cur is not None:
            yield cur
    else:
        # C короче A — оставшаяся часть A должна уйти в c_open.
        cur = subst.copy()
        for ce, ae in zip(c_elems, a_elems[:len(c_elems)]):
            cur = ce.unify(ae, cur, depth=depth)
            if cur is None:
                return
        cur = a_open.unify(Term('[]'), cur, depth=depth)
        if cur is None:
            return
        # c_open := A[len(c_elems):] ++ B
        chain = b
        for elem in reversed(a_elems[len(c_elems):]):
            chain = Term('.', [elem, chain])
        cur = c_open.unify(chain, cur, depth=depth)
        if cur is not None:
            yield cur


@prolog_predicate('length', 2)
def builtin_length(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """length(List, N) — итеративный подсчёт, O(len(List))."""
    if len(goal.arguments) != 2:
        return
    lst = goal.arguments[0]
    n = goal.arguments[1]

    current = lst
    count = 0
    open_tail = None

    while True:
        while isinstance(current, Variable) and current in subst:
            current = subst[current]
        if isinstance(current, Variable):
            open_tail = current
            break
        if not isinstance(current, Term):
            return
        if current.functor == '[]' and not current.arguments:
            break
        if current.functor == '.' and len(current.arguments) == 2:
            count += 1
            current = current.arguments[1]
            continue
        return  # improper list

    if open_tail is None:
        # список полностью замкнут — одно решение
        new_subst = n.unify(Number(count), subst.copy(), depth=depth)
        if new_subst is not None:
            yield new_subst
        return

    # список открыт: разрешаем n
    n_resolved = n
    while isinstance(n_resolved, Variable) and n_resolved in subst:
        n_resolved = subst[n_resolved]

    if isinstance(n_resolved, Number):
        target = int(n_resolved.value)
        if target < count:
            return
        # дозамкнуть хвост target-count анонимными переменными
        tail = Term('[]')
        for _ in range(target - count):
            tail = Term('.', [Variable('_'), tail])
        new_subst = open_tail.unify(tail, subst.copy(), depth=depth)
        if new_subst is not None:
            yield new_subst
        return

    # оба не связаны — выдаём одно минимальное решение (ISO-перечисление
    # бесконечно; ограничиваемся базовым случаем)
    new_subst = open_tail.unify(Term('[]'), subst.copy(), depth=depth)
    if new_subst is not None:
        new_subst = n.unify(Number(count), new_subst, depth=depth)
        if new_subst is not None:
            yield new_subst


# ----------------------------------------------------------------------
# Ввод/вывод
# ----------------------------------------------------------------------
@prolog_predicate('write', 1)
def builtin_write(goal: Term, database: 'Database', subst: Dict, depth: int) -> List[Dict]:
    if len(goal.arguments) != 1:
        return []
    value = goal.arguments[0].substitute(subst)
    if isinstance(value, Term) and (
        (str(value.functor).startswith("'") and str(value.functor).endswith("'")) or
        (str(value.functor).startswith('"') and str(value.functor).endswith('"'))
    ):
        print(str(value.functor)[1:-1], end='')
    else:
        print(value, end='')
    return [subst.copy()]


@prolog_predicate('nl', 0)
def builtin_nl(goal: Term, database: 'Database', subst: Dict, depth: int) -> List[Dict]:
    if goal.arguments:
        return []
    print()
    return [subst.copy()]


# ----------------------------------------------------------------------
# Управляющие предикаты
# ----------------------------------------------------------------------
@prolog_predicate('repeat', 0)
def builtin_repeat(goal: Term, database: 'Database', subst: Dict, depth: int):
    if goal.arguments:
        return
    raise RepeatException(subst)


@prolog_predicate('fail', 0)
def builtin_fail(goal: Term, database: 'Database', subst: Dict, depth: int) -> List[Dict]:
    return []


@prolog_predicate('false', 0)
def builtin_false(goal: Term, database: 'Database', subst: Dict, depth: int) -> List[Dict]:
    return []


@prolog_predicate('true', 0)
def builtin_true(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if goal.arguments:
        return
    yield subst.copy()


@prolog_predicate('!', 0)
def builtin_cut(goal: Term, database: 'Database', subst: Dict, depth: int):
    if goal.arguments:
        return
    raise CutException(subst)


# ----------------------------------------------------------------------
# Отрицание
# ----------------------------------------------------------------------
@prolog_predicate(r'\+', 1)
def builtin_neg(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    subgoal = goal.arguments[0]
    concrete_goal = subgoal.substitute(subst)
    try:
        gen = database._solve(concrete_goal, database.rule_index, subst.copy(), depth)
        for _ in gen:
            return
        yield subst.copy()
    except Exception:
        yield subst.copy()


# ----------------------------------------------------------------------
# Предикаты базы данных
# ----------------------------------------------------------------------
@prolog_predicate('asserta', 1)
def builtin_asserta(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    term = goal.arguments[0].substitute(subst)
    rule = _term_to_rule(term)
    database.add_rule(rule, at_beginning=True)
    yield subst.copy()


@prolog_predicate('assertz', 1)
def builtin_assertz(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    term = goal.arguments[0].substitute(subst)
    rule = _term_to_rule(term)
    database.add_rule(rule, at_beginning=False)
    yield subst.copy()


@prolog_predicate('retract', 1)
def builtin_retract(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    term = goal.arguments[0].substitute(subst)
    template_rule = _term_to_rule(term)
    new_subst = database.retract_first(template_rule)
    if new_subst is not None:
        merged = subst.copy()
        merged.update(new_subst)
        yield merged


@prolog_predicate('retractall', 1)
def builtin_retractall(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    term = goal.arguments[0].substitute(subst)
    template_rule = _term_to_rule(term)
    database.retract_all(template_rule)
    yield subst.copy()


@prolog_predicate('abolish', 1)
def builtin_abolish(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    indicator = goal.arguments[0].substitute(subst)
    if not isinstance(indicator, Term) or indicator.functor != '/' or len(indicator.arguments) != 2:
        raise PrologError('type_error', 'predicate_indicator', indicator)
    name_term = indicator.arguments[0]
    arity_term = indicator.arguments[1]
    if not isinstance(name_term, Term) or name_term.arguments:
        raise PrologError('type_error', 'atom', name_term)
    name = name_term.functor
    try:
        arity = evaluate_expression(arity_term, subst)
    except PrologError:
        raise PrologError('type_error', 'integer', arity_term)
    if not isinstance(arity, int) or arity < 0:
        raise PrologError('type_error', 'integer', arity_term)
    database.abolish(name, arity)
    yield subst.copy()


@prolog_predicate('clause', 2)
def builtin_clause(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    head_template = goal.arguments[0]
    body_var = goal.arguments[1]
    head_inst = head_template.substitute(subst)
    if isinstance(head_inst, Variable):
        raise PrologError('instantiation_error', 'Head not sufficiently instantiated', head_template)
    if not isinstance(head_inst, Term):
        raise PrologError('type_error', 'callable', head_inst)
    name = head_inst.functor
    arity = len(head_inst.arguments)
    rules = database.get_rules_for_predicate(name, arity)
    for rule in rules:
        renamed_rule = database._rename_variables(rule, depth)
        new_subst = head_inst.unify(renamed_rule.head, subst.copy(), depth=depth)
        if new_subst is None:
            continue
        body = renamed_rule.body
        if isinstance(body, Term) and body.functor == 'true' and len(body.arguments) == 0:
            body_term = Term('true', [])
        else:
            body_term = body
        final_subst = body_var.unify(body_term, new_subst)
        if final_subst is not None:
            yield final_subst


@prolog_predicate('current_predicate', 1)
def builtin_current_predicate(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    indicator = goal.arguments[0].substitute(subst)
    if isinstance(indicator, Variable):
        seen = set()
        for key in database.rule_index.keys():
            name, arity = key
            if (name, arity) in seen:
                continue
            seen.add((name, arity))
            pred_ind = Term('/', [Term(name), Number(arity)])
            new_subst = indicator.unify(pred_ind, subst.copy(), depth=depth)
            if new_subst is not None:
                yield new_subst
        for name, arities in database.builtins.builtins.items():
            for arity in arities:
                if (name, arity) in seen:
                    continue
                seen.add((name, arity))
                pred_ind = Term('/', [Term(name), Number(arity)])
                new_subst = indicator.unify(pred_ind, subst.copy(), depth=depth)
                if new_subst is not None:
                    yield new_subst
    else:
        if not isinstance(indicator, Term) or indicator.functor != '/' or len(indicator.arguments) != 2:
            raise PrologError('type_error', 'predicate_indicator', indicator)
        name_term = indicator.arguments[0]
        arity_term = indicator.arguments[1]
        if not isinstance(name_term, Term) or name_term.arguments:
            raise PrologError('type_error', 'atom', name_term)
        name = name_term.functor
        try:
            arity = evaluate_expression(arity_term, subst)
        except PrologError:
            raise PrologError('type_error', 'integer', arity_term)
        if not isinstance(arity, int) or arity < 0:
            raise PrologError('type_error', 'integer', arity_term)
        if database.has_predicate(name, arity):
            yield subst.copy()


# ----------------------------------------------------------------------
# Мета-предикаты
#
# ОБЩИЙ ПРИНЦИП (см. findall):
#   * RepeatException / CutException — управляющие, пробрасываем как есть.
#   * PrologError                   — семантическая ошибка, пробрасываем.
#   * Прочие Exception              — печатаем traceback и пробрасываем.
# Раньше все они молча превращались в fail/return, из-за чего ошибка
# внутри Goal (например, TypeError в Python-действии distance/5)
# выглядела как «нет решений».
# ----------------------------------------------------------------------
def _call_with_args(g: Term, args: List[Term], database: 'Database',
                    subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if isinstance(g, Variable):
        if g in subst:
            g = subst[g]
        else:
            return
    if not isinstance(g, Term):
        return
    if g.arguments:
        all_args = g.arguments + [a.substitute(subst) for a in args]
        new_term = Term(g.functor, all_args)
    else:
        new_term = Term(g.functor, [a.substitute(subst) for a in args])
    try:
        yield from database._solve(new_term, database.rule_index, subst.copy(), depth)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('call/N', new_term, e)


@prolog_predicate('call', 1)
def builtin_call_1(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    call_goal = goal.arguments[0].substitute(subst)
    if isinstance(call_goal, Variable):
        if call_goal in subst:
            call_goal = subst[call_goal]
        else:
            return
    if not isinstance(call_goal, Term):
        return
    try:
        yield from database._solve(call_goal, database.rule_index, subst.copy(), depth)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('call/1', call_goal, e)


@prolog_predicate('call', 2)
def builtin_call_2(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    g = goal.arguments[0].substitute(subst)
    arg1 = goal.arguments[1]
    yield from _call_with_args(g, [arg1], database, subst, depth)


@prolog_predicate('call', 3)
def builtin_call_3(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    g = goal.arguments[0].substitute(subst)
    args = goal.arguments[1:]
    yield from _call_with_args(g, args, database, subst, depth)


@prolog_predicate('call', 4)
def builtin_call_4(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 4:
        return
    g = goal.arguments[0].substitute(subst)
    args = goal.arguments[1:]
    yield from _call_with_args(g, args, database, subst, depth)


@prolog_predicate('call', 5)
def builtin_call_5(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 5:
        return
    g = goal.arguments[0].substitute(subst)
    args = goal.arguments[1:]
    yield from _call_with_args(g, args, database, subst, depth)


@prolog_predicate('once', 1)
def builtin_once(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    subgoal = goal.arguments[0].substitute(subst)
    try:
        gen = database._solve(subgoal, database.rule_index, subst.copy(), depth)
        for sol in gen:
            yield sol
            return
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('once/1', subgoal, e)


@prolog_predicate('ignore', 1)
def builtin_ignore(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    # ignore/1 по ISO семантике глотает ЛЮБЫЕ ошибки и всегда успешен —
    # оставляем как есть.
    if len(goal.arguments) != 1:
        return
    subgoal = goal.arguments[0].substitute(subst)
    try:
        for _ in database._solve(subgoal, database.rule_index, subst.copy(), depth):
            pass
    except RepeatException:
        raise
    except Exception:
        pass
    yield subst.copy()


@prolog_predicate('forall', 2)
def builtin_forall(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    gen = goal.arguments[0].substitute(subst)
    test = goal.arguments[1]
    gen_solutions = []
    try:
        for sol in database._solve(gen, database.rule_index, subst.copy(), depth):
            gen_solutions.append(sol)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('forall/2 (generator)', gen, e)
    for gen_subst in gen_solutions:
        test_subst = gen_subst.copy()
        test_goal = test.substitute(test_subst)
        try:
            test_gen = database._solve(test_goal, database.rule_index, test_subst, depth)
            found = False
            for _ in test_gen:
                found = True
                break
            if not found:
                return
        except (RepeatException, CutException):
            raise
        except PrologError:
            raise
        except Exception as e:
            _reraise_unexpected('forall/2 (test)', test_goal, e)
    yield subst.copy()


# ----------------------------------------------------------------------
# findall / bagof / setof
#
# ИСПРАВЛЕНО (v5.3):
#   Раньше эти предикаты содержали `except Exception: return`, из-за чего
#   ЛЮБАЯ ошибка внутри Goal — в том числе TypeError/ValueError из
#   Python-действия вроде distance/5 — превращалась в пустой список.
#   Это делало невозможным отличить «нет решений» от «что-то упало
#   внутри» и приводило к «тихому» поведению всей рефлексивной логики.
#
#   Теперь:
#     * PrologError пробрасывается (ISO-семантика);
#     * прочие Exception печатаются с traceback и пробрасываются;
#     * RepeatException/CutException пробрасываются как управляющие.
# ----------------------------------------------------------------------
@prolog_predicate('findall', 3)
def builtin_findall(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """findall(Template, Goal, List)."""
    if len(goal.arguments) != 3:
        return
    template = goal.arguments[0]
    subgoal = goal.arguments[1].substitute(subst)
    list_var = goal.arguments[2]
    solutions_list = []
    try:
        for sol in database._solve(subgoal, database.rule_index, subst.copy(), depth):
            instance = template.substitute(sol)
            solutions_list.append(instance)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('findall', subgoal, e)
    result_list = _build_prolog_list(solutions_list)
    new_subst = list_var.unify(result_list, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('bagof', 3)
def builtin_bagof(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    template = goal.arguments[0]
    raw_goal = goal.arguments[1]
    list_var = goal.arguments[2]
    existential_vars, clean_goal = _extract_existential(raw_goal, subst)
    all_solutions = []
    try:
        for sol in database._solve(clean_goal, database.rule_index, subst.copy(), depth):
            all_solutions.append(sol)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('bagof', clean_goal, e)
    if not all_solutions:
        return
    goal_vars = _collect_variables(clean_goal)
    template_vars = _collect_variables(template)
    grouping_vars = (goal_vars - template_vars) - existential_vars
    groups = {}
    for sol in all_solutions:
        key = tuple(str(sol.get(var, var)) for var in grouping_vars)
        instance = template.substitute(sol)
        groups.setdefault(key, []).append((sol, instance))
    for key, group_data in groups.items():
        sample_subst = group_data[0][0]
        group_instances = [inst for _, inst in group_data]
        group_list = _build_prolog_list(group_instances)
        new_subst = subst.copy()
        for var in grouping_vars:
            if var in sample_subst:
                new_subst[var] = sample_subst[var]
        new_subst = list_var.unify(group_list, new_subst)
        if new_subst is not None:
            yield new_subst


@prolog_predicate('setof', 3)
def builtin_setof(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    template = goal.arguments[0]
    raw_goal = goal.arguments[1]
    list_var = goal.arguments[2]
    existential_vars, clean_goal = _extract_existential(raw_goal, subst)
    all_solutions = []
    try:
        for sol in database._solve(clean_goal, database.rule_index, subst.copy(), depth):
            all_solutions.append(sol)
    except (RepeatException, CutException):
        raise
    except PrologError:
        raise
    except Exception as e:
        _reraise_unexpected('setof', clean_goal, e)
    if not all_solutions:
        return
    goal_vars = _collect_variables(clean_goal)
    template_vars = _collect_variables(template)
    grouping_vars = (goal_vars - template_vars) - existential_vars
    groups = {}
    for sol in all_solutions:
        key = tuple(str(sol.get(var, var)) for var in grouping_vars)
        instance = template.substitute(sol)
        groups.setdefault(key, []).append(instance)
    for key, instances in groups.items():
        unique = []
        seen = set()
        for inst in instances:
            s = str(inst)
            if s not in seen:
                seen.add(s)
                unique.append(inst)
        unique.sort(key=str)
        group_list = _build_prolog_list(unique)
        sample_subst = {}
        for var in grouping_vars:
            for sol in all_solutions:
                if var in sol:
                    sample_subst[var] = sol[var]
                    break
        new_subst = subst.copy()
        new_subst.update(sample_subst)
        new_subst = list_var.unify(group_list, new_subst)
        if new_subst is not None:
            yield new_subst


# ----------------------------------------------------------------------
# Строковые и атомные предикаты
# ----------------------------------------------------------------------
@prolog_predicate('string_concat', 3)
def builtin_string_concat(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    a = goal.arguments[0].substitute(subst)
    b = goal.arguments[1].substitute(subst)
    c = goal.arguments[2]
    if not isinstance(a, Variable) and not isinstance(b, Variable):
        sa = _term_to_string(a)
        sb = _term_to_string(b)
        sc = sa + sb
        c_term = _string_to_term(sc, quoted=True)
        new_subst = c.unify(c_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return


@prolog_predicate('atomic_list_concat', 2)
def builtin_atomic_list_concat_2(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    lst = goal.arguments[0].substitute(subst)
    atom_var = goal.arguments[1]
    elements = _list_to_python(lst)
    if elements is None:
        return
    strings = [_term_to_string(e) for e in elements]
    result_str = ''.join(strings)
    result_term = _string_to_term(result_str, quoted=False)
    new_subst = atom_var.unify(result_term, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('atomic_list_concat', 3)
def builtin_atomic_list_concat_3(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    lst = goal.arguments[0].substitute(subst)
    sep = goal.arguments[1].substitute(subst)
    atom_var = goal.arguments[2]
    elements = _list_to_python(lst)
    if elements is None:
        return
    sep_str = _term_to_string(sep)
    strings = [_term_to_string(e) for e in elements]
    result_str = sep_str.join(strings)
    result_term = _string_to_term(result_str, quoted=False)
    new_subst = atom_var.unify(result_term, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('atom_string', 2)
def builtin_atom_string(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    arg1 = goal.arguments[0].substitute(subst)
    arg2 = goal.arguments[1].substitute(subst)
    
    if not isinstance(arg1, Variable) and isinstance(arg2, Variable):
        if isinstance(arg1, Term) and not arg1.arguments:
            s = _term_to_string(arg1)
        else:
            s = str(arg1)
        string_term = _string_to_term(s, quoted=True)
        new_subst = arg2.unify(string_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return
    
    if isinstance(arg1, Variable) and not isinstance(arg2, Variable):
        s = _term_to_string(arg2)
        atom_term = _string_to_term(s, quoted=False)
        new_subst = arg1.unify(atom_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return
    
    if not isinstance(arg1, Variable) and not isinstance(arg2, Variable):
        s1 = _term_to_string(arg1)
        s2 = _term_to_string(arg2)
        if s1 == s2:
            yield subst.copy()


@prolog_predicate('term_string', 2)
def builtin_term_string(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    term = goal.arguments[0].substitute(subst)
    str_var = goal.arguments[1]
    if isinstance(term, Variable):
        return
    term_str = str(term)
    string_term = _string_to_term(term_str, quoted=True)
    new_subst = str_var.unify(string_term, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('number_string', 2)
def builtin_number_string(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    arg1 = goal.arguments[0].substitute(subst)
    arg2 = goal.arguments[1].substitute(subst)
    if isinstance(arg1, Number) and isinstance(arg2, Variable):
        num_str = str(arg1.value)
        string_term = _string_to_term(num_str, quoted=True)
        new_subst = arg2.unify(string_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return
    if isinstance(arg1, Variable) and not isinstance(arg2, Variable):
        s = _term_to_string(arg2)
        try:
            if '.' in s or 'e' in s.lower():
                num = float(s)
            else:
                num = int(s)
        except ValueError:
            return
        num_term = Number(num)
        new_subst = arg1.unify(num_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return
    if isinstance(arg1, Number) and not isinstance(arg2, Variable):
        s = _term_to_string(arg2)
        if str(arg1.value) == s:
            yield subst.copy()


@prolog_predicate('split_string', 4)
def builtin_split_string(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 4:
        return
    string_term = goal.arguments[0].substitute(subst)
    sep_term = goal.arguments[1].substitute(subst)
    pad_term = goal.arguments[2].substitute(subst)
    list_var = goal.arguments[3]
    if isinstance(string_term, Variable) or isinstance(sep_term, Variable):
        return
    s = _term_to_string(string_term)
    sep = _term_to_string(sep_term)
    parts = s.split(sep)
    prolog_parts = [_string_to_term(p, quoted=True) for p in parts]
    prolog_list = _build_prolog_list(prolog_parts)
    new_subst = list_var.unify(prolog_list, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('sub_string', 5)
def builtin_sub_string(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 5:
        return
    args = [a.substitute(subst) for a in goal.arguments]
    s_term, before_term, length_term, after_term, sub_term = args
    if not isinstance(s_term, Variable):
        s = _term_to_string(s_term)
        if not isinstance(before_term, Variable) and not isinstance(length_term, Variable):
            try:
                before = int(_term_to_string(before_term)) if isinstance(before_term, (Number, Term)) else 0
                length = int(_term_to_string(length_term))
            except:
                return
            if before < 0 or length < 0 or before + length > len(s):
                return
            sub = s[before:before+length]
            after = len(s) - (before + length)
            sub_term_val = _string_to_term(sub, quoted=True)
            after_term_val = Number(after)
            new_subst = sub_term.unify(sub_term_val, subst.copy())
            if new_subst is None:
                return
            new_subst = after_term.unify(after_term_val, new_subst)
            if new_subst is None:
                return
            yield new_subst
            return
        elif not isinstance(sub_term, Variable):
            sub = _term_to_string(sub_term)
            pos = s.find(sub)
            if pos == -1:
                return
            before = pos
            length = len(sub)
            after = len(s) - (before + length)
            before_term_val = Number(before)
            length_term_val = Number(length)
            after_term_val = Number(after)
            new_subst = before_term.unify(before_term_val, subst.copy())
            if new_subst is None: return
            new_subst = length_term.unify(length_term_val, new_subst)
            if new_subst is None: return
            new_subst = after_term.unify(after_term_val, new_subst)
            if new_subst is None: return
            yield new_subst
            return


@prolog_predicate('memberchk', 2)
def builtin_memberchk(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    elem = goal.arguments[0]
    lst = goal.arguments[1].substitute(subst)
    member_goal = Term('member', [elem, lst])
    for sol in builtin_member(member_goal, database, subst, depth):
        yield sol
        return


# ----------------------------------------------------------------------
# Предикаты сравнения термов
# ----------------------------------------------------------------------
@prolog_predicate('==', 2)
def builtin_eq_eq(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if is_identical(t1, t2):
        yield subst.copy()


@prolog_predicate(r'\==', 2)
def builtin_not_eq_eq(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if not is_identical(t1, t2):
        yield subst.copy()


@prolog_predicate(r'\=', 2)
def builtin_not_unify(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    x = goal.arguments[0]
    y = goal.arguments[1]
    x_val = x.substitute(subst)
    y_val = y.substitute(subst)
    if isinstance(x_val, Number) and isinstance(y_val, Number):
        if type(x_val.value) is not type(y_val.value) or x_val.value != y_val.value:
            yield subst.copy()
        return
    test_subst = subst.copy()
    result = x.unify(y, test_subst, depth=depth)
    if result is None:
        yield subst.copy()


@prolog_predicate('@<', 2)
def builtin_at_lt(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if standard_order(t1, t2) < 0:
        yield subst.copy()


@prolog_predicate('@>', 2)
def builtin_at_gt(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if standard_order(t1, t2) > 0:
        yield subst.copy()


@prolog_predicate('@=<', 2)
def builtin_at_le(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if standard_order(t1, t2) <= 0:
        yield subst.copy()


@prolog_predicate('@>=', 2)
def builtin_at_ge(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 2:
        return
    t1 = goal.arguments[0]
    t2 = goal.arguments[1]
    if standard_order(t1, t2) >= 0:
        yield subst.copy()


@prolog_predicate('compare', 3)
def builtin_compare(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 3:
        return
    res_var = goal.arguments[0]
    t1 = goal.arguments[1]
    t2 = goal.arguments[2]
    cmp = standard_order(t1, t2)
    if cmp < 0:
        result_term = Term('<')
    elif cmp == 0:
        result_term = Term('=')
    else:
        result_term = Term('>')
    new_subst = res_var.unify(result_term, subst.copy())
    if new_subst is not None:
        yield new_subst


# ----------------------------------------------------------------------
# Предикаты конструкторы и деструкторы термов
# ----------------------------------------------------------------------
@prolog_predicate('functor', 3)
def builtin_functor(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    term, functor_arg, arity_arg = goal.arguments
    t = term.substitute(subst)
    f = functor_arg.substitute(subst)
    a = arity_arg.substitute(subst)
    if isinstance(t, Variable) and t.name != "_":
        if not isinstance(f, Term) or f.arguments:
            return
        if not isinstance(a, Number):
            return
        arity = int(a.value)
        if arity < 0:
            return
        new_args = [Variable(f"_G{i}") for i in range(arity)]
        new_term = Term(f.functor, new_args)
        new_subst = term.unify(new_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return
    if not isinstance(t, Term):
        return
    if isinstance(t, Number):
        functor_val = t.value
        arity_val = 0
    else:
        functor_val = t.functor
        arity_val = len(t.arguments)
    functor_term = Number(functor_val) if isinstance(functor_val, (int, float)) else Term(functor_val)
    arity_term = Number(arity_val)
    new_subst = functor_arg.unify(functor_term, subst.copy())
    if new_subst is None:
        return
    new_subst = arity_arg.unify(arity_term, new_subst)
    if new_subst is not None:
        yield new_subst


@prolog_predicate('arg', 3)
def builtin_arg(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    n_arg, term_arg, arg_val = goal.arguments
    n = n_arg.substitute(subst)
    t = term_arg.substitute(subst)
    if isinstance(n, Variable):
        if n.name == "_":
            return
        if not isinstance(t, Term):
            return
        args = t.arguments
        for idx, arg in enumerate(args, start=1):
            new_subst = arg_val.unify(arg, subst.copy())
            if new_subst is not None:
                new_subst = n.unify(Number(idx), new_subst)
                if new_subst is not None:
                    yield new_subst
        return
    if not isinstance(n, Number):
        return
    n = int(n.value)
    if n < 1:
        return
    if not isinstance(t, Term):
        return
    args = t.arguments
    if n > len(args):
        return
    nth = args[n-1]
    new_subst = arg_val.unify(nth, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('=..', 2)
def builtin_univ(goal: Term, database, subst: Dict, depth: int) -> Generator[Dict, None, None]:
    term_arg, list_arg = goal.arguments
    t = term_arg.substitute(subst)
    l = list_arg.substitute(subst)

    if isinstance(t, Variable) and t.name != "_":
        elements = _list_to_python(l)
        if not elements:
            return
        functor = elements[0]
        args = elements[1:]
        if isinstance(functor, Number):
            if args:
                return
            new_term = Number(functor.value)
        elif isinstance(functor, Term) and not functor.arguments:
            new_term = Term(functor.functor, args)
        else:
            return
        new_subst = term_arg.unify(new_term, subst.copy())
        if new_subst is not None:
            yield new_subst
        return

    if not isinstance(t, Variable) and isinstance(l, Variable) and l.name != "_":
        if isinstance(t, Number):
            functor_val = t
            args = []
        elif isinstance(t, Term):
            functor_val = Term(t.functor)
            args = t.arguments
        else:
            return
        lst = Term('[]')
        for arg in reversed(args):
            lst = Term('.', [arg, lst])
        lst = Term('.', [functor_val, lst])
        new_subst = list_arg.unify(lst, subst.copy())
        if new_subst is not None:
            yield new_subst
        return

    if not isinstance(t, Variable) and isinstance(l, Term) and (l.functor == '.' or l.functor == '[]'):
        if isinstance(t, Number):
            functor_val = t
            args = []
        elif isinstance(t, Term):
            functor_val = Term(t.functor)
            args = t.arguments
        else:
            return
        lst = Term('[]')
        for arg in reversed(args):
            lst = Term('.', [arg, lst])
        lst = Term('.', [functor_val, lst])
        new_subst = lst.unify(l, subst.copy())
        if new_subst is not None:
            yield new_subst


@prolog_predicate('copy_term', 2)
def builtin_copy_term(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    term, copy = goal.arguments
    t = term.substitute(subst)
    def copy_rec(t, var_map):
        if isinstance(t, Variable):
            if t.name == "_":
                return Variable("_")
            if t in var_map:
                return var_map[t]
            new_var = Variable(f"{t.name}_copy")
            var_map[t] = new_var
            return new_var
        if isinstance(t, Number):
            return Number(t.value)
        if isinstance(t, Term):
            new_args = [copy_rec(arg, var_map) for arg in t.arguments]
            return Term(t.functor, new_args)
        return t
    copied = copy_rec(t, {})
    new_subst = copy.unify(copied, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('term_variables', 2)
def builtin_term_variables(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    term, vars_list = goal.arguments
    t = term.substitute(subst)
    def collect(t, seen):
        if isinstance(t, Variable):
            if t.name != "_" and t not in seen:
                seen.append(t)
        elif isinstance(t, Term):
            for arg in t.arguments:
                collect(arg, seen)
    seen = []
    collect(t, seen)
    lst = Term('[]')
    for var in reversed(seen):
        lst = Term('.', [var, lst])
    new_subst = vars_list.unify(lst, subst.copy())
    if new_subst is not None:
        yield new_subst


@prolog_predicate('numbervars', 3)
def builtin_numbervars(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    term, start, end = goal.arguments
    t = term.substitute(subst)
    s = start.substitute(subst)
    if not isinstance(s, Number):
        return
    counter = int(s.value)
    def numbervars_rec(t, var_map):
        nonlocal counter
        if isinstance(t, Variable):
            if t.name == "_":
                return Variable("_")
            if t in var_map:
                return var_map[t]
            new_var = Term('$VAR', [Number(counter)])
            var_map[t] = new_var
            counter += 1
            return new_var
        if isinstance(t, Number):
            return Number(t.value)
        if isinstance(t, Term):
            new_args = [numbervars_rec(arg, var_map) for arg in t.arguments]
            return Term(t.functor, new_args)
        return t
    new_term = numbervars_rec(t, {})
    new_subst = term.unify(new_term, subst.copy())
    if new_subst is None:
        return
    new_subst = end.unify(Number(counter), new_subst)
    if new_subst is not None:
        yield new_subst


# ----------------------------------------------------------------------
# Предикаты проверки типов
# ----------------------------------------------------------------------
@prolog_predicate('var', 1)
def builtin_var(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Variable):
        yield subst.copy()


@prolog_predicate('nonvar', 1)
def builtin_nonvar(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if not isinstance(val, Variable):
        yield subst.copy()


@prolog_predicate('atom', 1)
def builtin_atom(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Истинно, если аргумент является атомом (не число, не переменная, не составной терм)."""
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Term) and not val.arguments and not isinstance(val, (Number, Variable)):
        yield subst.copy()


@prolog_predicate('integer', 1)
def builtin_integer(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Number) and isinstance(val.value, int):
        yield subst.copy()


@prolog_predicate('float', 1)
def builtin_float(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Number) and isinstance(val.value, float):
        yield subst.copy()


@prolog_predicate('number', 1)
def builtin_number(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Number):
        yield subst.copy()


@prolog_predicate('atomic', 1)
def builtin_atomic(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Истинно, если терм является числом, атомом или строкой (без переменных)."""
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Number):
        yield subst.copy()
    elif isinstance(val, Term) and not val.arguments and not isinstance(val, Variable):
        yield subst.copy()


@prolog_predicate('compound', 1)
def builtin_compound(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Term) and len(val.arguments) > 0:
        yield subst.copy()


@prolog_predicate('ground', 1)
def builtin_ground(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0]
    if _is_ground(val, subst):
        yield subst.copy()


@prolog_predicate('callable', 1)
def builtin_callable(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    if len(goal.arguments) != 1:
        return
    val = goal.arguments[0].substitute(subst)
    if isinstance(val, Term) and not isinstance(val, (Number, Variable)):
        yield subst.copy()


# ======================================================================
# ПРЕДИКАТЫ УПРАВЛЕНИЯ СЕМАНТИЧЕСКОЙ УНИФИКАЦИЕЙ
# ======================================================================
@prolog_predicate('semantic_on', 0)
def builtin_semantic_on_0(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Включить семантическую унификацию с параметрами по умолчанию."""
    if goal.arguments:
        return
    try:
        from prolog.embedding_unify import semantic_on
    except ImportError as e:
        raise PrologError('existence_error', f'embedding_unify module unavailable: {e}')
    semantic_on()
    yield subst.copy()


@prolog_predicate('semantic_on', 1)
def builtin_semantic_on_1(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Включить семантическую унификацию с заданным порогом: semantic_on(Threshold)."""
    if len(goal.arguments) != 1:
        return
    try:
        from prolog.embedding_unify import semantic_on
    except ImportError as e:
        raise PrologError('existence_error', f'embedding_unify module unavailable: {e}')
    t = goal.arguments[0].substitute(subst)
    if not isinstance(t, Number):
        raise PrologError('type_error', 'number', t)
    threshold = float(t.value)
    if not (0.0 <= threshold <= 1.0):
        raise PrologError('domain_error', 'probability', t)
    semantic_on(threshold=threshold)
    yield subst.copy()


@prolog_predicate('semantic_off', 0)
def builtin_semantic_off(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Отключить семантическую унификацию (перейти к строгой)."""
    if goal.arguments:
        return
    try:
        from prolog.embedding_unify import semantic_off
    except ImportError as e:
        raise PrologError('existence_error', f'embedding_unify module unavailable: {e}')
    semantic_off()
    yield subst.copy()


@prolog_predicate('semantic_threshold', 1)
def builtin_semantic_threshold(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Изменить порог семантической унификации у уже включённого унификатора."""
    if len(goal.arguments) != 1:
        return
    try:
        from prolog.embedding_unify import semantic_set_threshold, semantic_get_unifier
    except ImportError as e:
        raise PrologError('existence_error', f'embedding_unify module unavailable: {e}')
    if semantic_get_unifier() is None:
        return
    t = goal.arguments[0].substitute(subst)
    if not isinstance(t, Number):
        raise PrologError('type_error', 'number', t)
    threshold = float(t.value)
    if not (0.0 <= threshold <= 1.0):
        raise PrologError('domain_error', 'probability', t)
    semantic_set_threshold(threshold)
    yield subst.copy()


@prolog_predicate('semantic_status', 1)
def builtin_semantic_status(goal: Term, database: 'Database', subst: Dict, depth: int) -> Generator[Dict, None, None]:
    """Получить статус семантической унификации."""
    if len(goal.arguments) != 1:
        return
    try:
        from prolog.embedding_unify import semantic_get_unifier
    except ImportError:
        status_term = Term('off')
    else:
        u = semantic_get_unifier()
        if u is None:
            status_term = Term('off')
        else:
            status_term = Term('on', [Number(u.threshold)])
    new_subst = goal.arguments[0].unify(status_term, subst.copy(), depth=depth)
    if new_subst is not None:
        yield new_subst


# ======================================================================
# ДИАГНОСТИКА NUMBA-УСКОРЕНИЯ АРИФМЕТИКИ
# ======================================================================

def arith_numba_status() -> Dict[str, Any]:
    """Статус Numba-ускорения арифметики (для диагностики и тестов)."""
    return _numba_status()


def warmup_arith_numba() -> bool:
    """Прогреть JIT-ядро арифметики."""
    return _warmup_numba()


# ======================================================================
# АВТОМАТИЧЕСКОЕ ПОСТРОЕНИЕ РЕЕСТРА ПРЕДИКАТОВ
# ======================================================================
def _collect_builtins_from_module():
    """Собирает все функции с атрибутами _is_prolog_predicate в словарь."""
    import sys
    module = sys.modules[__name__]
    predicates = {}
    for name in dir(module):
        obj = getattr(module, name)
        if callable(obj) and getattr(obj, '_is_prolog_predicate', False):
            pred_name = obj._predicate_name
            arity = obj._predicate_arity
            key = (pred_name, arity)
            predicates[key] = obj
    return predicates


# Словарь для обратной совместимости (используется другими модулями)
STANDARD_PREDICATES = {}
for (name, arity), func in _collect_builtins_from_module().items():
    if name not in STANDARD_PREDICATES:
        STANDARD_PREDICATES[name] = {}
    STANDARD_PREDICATES[name][arity] = func