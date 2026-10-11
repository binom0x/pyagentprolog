"""PyAgentProlog — канонический мост Prolog ↔ Python.

Единственная точка преобразования между Prolog-термами и Python-объектами.
Реализация перенесена сюда из prolog/agent.py, чтобы устранить дублирование
с neuroagent/bridge.py и сохранить PyAgentProlog как самостоятельный,
независимый пакет (prolog не зависит от neuroagent).

Использование:
    from prolog.bridge import prolog_to_python, python_to_prolog

Совместимость:
    from prolog.agent import prolog_to_python       # тот же объект
    from neuroagent.bridge import prolog_to_python  # тот же объект

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""
from __future__ import annotations
from typing import Any, List

from prolog.terms import Term, Variable, Number


def prolog_list_to_python(term: Any) -> Any:
    """Преобразует Prolog-список (Term с . или []) в Python-список термов.

    Отличие от prolog_to_python: возвращает СПИСОК ТЕРМОВ (Term), а не
    Python-значений. Если term — не список, возвращает его как есть.
    """
    if isinstance(term, Term):
        if term.functor == '[]' and len(term.arguments) == 0:
            return []
        elif term.functor == '.' and len(term.arguments) == 2:
            elements = []
            current = term
            while (isinstance(current, Term)
                   and current.functor == '.'
                   and len(current.arguments) == 2):
                elements.append(current.arguments[0])
                current = current.arguments[1]
            if (isinstance(current, Term)
                    and current.functor == '[]'
                    and len(current.arguments) == 0):
                return elements
    return term


def prolog_to_python(term: Any) -> Any:
    """Преобразует Prolog-терм в Python-объект.

    Соответствия:
        Number            → int/float
        Variable          → None
        Строка в кавычках → str (без кавычек)
        Атом              → str
        Список            → list
        Составной терм    → str
    """
    if isinstance(term, Number):
        return term.value
    if isinstance(term, Variable):
        return None
    if isinstance(term, Term):
        # Пустой список
        if term.functor == '[]' and not term.arguments:
            return []
        # Непустой список
        if term.functor == '.' and len(term.arguments) == 2:
            elements = []
            cur = term
            while (isinstance(cur, Term)
                   and cur.functor == '.'
                   and len(cur.arguments) == 2):
                elements.append(prolog_to_python(cur.arguments[0]))
                cur = cur.arguments[1]
            if (isinstance(cur, Term)
                    and cur.functor == '[]'
                    and not cur.arguments):
                return elements
            return str(term)
        # Атом (возможно, число)
        if not term.arguments:
            f = term.functor
            if not isinstance(f, str):
                return f
            # Пробуем интерпретировать как число
            try:
                if '.' in f or 'e' in f.lower():
                    return float(f)
                return int(f)
            except ValueError:
                pass
            # Строка в кавычках — снимаем кавычки
            if len(f) >= 2 and f[0] == '"' and f[-1] == '"':
                return f[1:-1]
            if len(f) >= 2 and f[0] == "'" and f[-1] == "'":
                return f[1:-1]
            return f
        # Другие составные термы — строковое представление
        return str(term)
    return str(term)


def python_to_prolog(obj: Any) -> Term:
    """Преобразует Python-объект в Prolog-терм.

    Соответствия:
        Term         → как есть
        bool         → атом true/false
        int/float    → Number
        str          → атом в двойных кавычках
        list         → Prolog-список
        dict         → список пар '-'(K, V)
        None         → пустой список []
        прочее       → атом в двойных кавычках (str(obj))
    """
    if isinstance(obj, Term):
        return obj
    # bool ДО int: True/False должны быть атомами, а не числами
    if isinstance(obj, bool):
        return Term('true' if obj else 'false')
    if isinstance(obj, (int, float)):
        return Number(obj)
    if isinstance(obj, str):
        return Term(f'"{obj}"')
    if isinstance(obj, list):
        if not obj:
            return Term('[]')
        result = Term('[]')
        for item in reversed(obj):
            result = Term('.', [python_to_prolog(item), result])
        return result
    if isinstance(obj, dict):
        pairs: List[Term] = []
        for k, v in obj.items():
            pairs.append(Term('-', [python_to_prolog(k), python_to_prolog(v)]))
        return python_to_prolog(pairs)
    if obj is None:
        return Term('[]')
    return Term(f'"{obj}"')


__all__ = [
    'prolog_to_python',
    'python_to_prolog',
    'prolog_list_to_python',
]