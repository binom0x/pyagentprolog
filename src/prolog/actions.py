"""PyAgentProlog — канонические декораторы действий.

Декораторы позволяют подключить обычную Python-функцию как встроенный
предикат Prolog. Реализация перенесена сюда из prolog/agent.py, чтобы
устранить дублирование с neuroagent/actions.py и сохранить PyAgentProlog
как самостоятельный, независимый пакет (prolog не зависит от neuroagent).

Использование:
    from prolog.actions import simple_action, effector_action, generator_action

Совместимость:
    from prolog.agent import simple_action        # тот же объект
    from neuroagent.actions import simple_action  # тот же объект

Метаданные, которые вешаются на wrapper:
    _is_prolog_predicate, _predicate_name, _predicate_arity
        → для BuiltinRegistry.register_module
    _agent_action, _action_name, _action_arity,
    _action_description, _action_type
        → для Agent.register_action / register_action_module

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""
from __future__ import annotations
import functools
from typing import Optional

from prolog.terms import Term, Variable, Number
from prolog.bridge import prolog_to_python, python_to_prolog


def _tag(wrapper, name: str, arity: int, func, description: Optional[str],
         action_type: str) -> None:
    """Помечает wrapper метаданными действия и предиката.

    После _tag функция пригодна как для register_action_module (по
    _agent_action), так и для BuiltinRegistry.register_module
    (по _is_prolog_predicate).
    """
    wrapper._is_prolog_predicate = True
    wrapper._predicate_name = name
    wrapper._predicate_arity = arity

    wrapper._agent_action = True
    wrapper._action_name = name
    wrapper._action_arity = arity
    wrapper._action_description = description or func.__doc__ or ""
    wrapper._action_type = action_type


def simple_action(name: str, arity: int, output_arg: int = -1,
                  description: Optional[str] = None):
    """Действие, возвращающее одно значение.

    Args:
        name: имя предиката в Prolog.
        arity: общее число аргументов.
        output_arg: индекс (0-based) выходного аргумента, куда
            подставляется результат. -1 → эффектор (side effect,
            без возврата значения).
        description: описание (по умолчанию — docstring функции).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(goal, database, subst, depth):
            if len(goal.arguments) != arity:
                return []

            # Эффектор
            if output_arg == -1:
                inputs = []
                for i in range(arity):
                    arg = goal.arguments[i].substitute(subst)
                    if isinstance(arg, Variable):
                        return []
                    inputs.append(prolog_to_python(arg))
                try:
                    func(*inputs, context=database.context)
                except Exception:
                    return []
                return [subst.copy()]

            # Входные аргументы
            inputs = []
            for i in range(arity):
                if i == output_arg:
                    continue
                arg = goal.arguments[i].substitute(subst)
                if isinstance(arg, Variable):
                    return []
                inputs.append(prolog_to_python(arg))

            try:
                result = func(*inputs, context=database.context)
            except Exception:
                return []
            if result is None:
                return []

            result_term = python_to_prolog(result)
            out_arg = goal.arguments[output_arg]
            new_subst = out_arg.unify(result_term, subst.copy())
            if new_subst is None:
                return []
            return [new_subst]

        _tag(wrapper, name, arity, func, description, "simple")
        return wrapper
    return decorator


def effector_action(name: str, arity: int,
                    description: Optional[str] = None):
    """Действие-эффектор: только side effect, без возврата значения.

    Все аргументы считаются входными и должны быть конкретизированы.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(goal, database, subst, depth):
            if len(goal.arguments) != arity:
                return []
            inputs = []
            for i in range(arity):
                arg = goal.arguments[i].substitute(subst)
                if isinstance(arg, Variable):
                    return []
                inputs.append(prolog_to_python(arg))
            try:
                func(*inputs, context=database.context)
            except Exception:
                return []
            return [subst.copy()]

        _tag(wrapper, name, arity, func, description, "effector")
        return wrapper
    return decorator


def generator_action(name: str, arity: int, output_arg: int = -1,
                     description: Optional[str] = None):
    """Действие-генератор: функция возвращает итератор.

    Каждое значение итератора порождает отдельное решение Prolog.

    Если output_arg == -1, действие считается эффектором: функция
    вызывается один раз, а итератор игнорируется (совместимо с
    effector_action).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(goal, database, subst, depth):
            if len(goal.arguments) != arity:
                return []

            # Эффектор
            if output_arg == -1:
                inputs = []
                for i in range(arity):
                    arg = goal.arguments[i].substitute(subst)
                    if isinstance(arg, Variable):
                        return []
                    inputs.append(prolog_to_python(arg))
                try:
                    func(*inputs, context=database.context)
                except Exception:
                    return []
                return [subst.copy()]

            # Входные аргументы
            inputs = []
            for i in range(arity):
                if i == output_arg:
                    continue
                arg = goal.arguments[i].substitute(subst)
                if isinstance(arg, Variable):
                    return []
                inputs.append(prolog_to_python(arg))

            try:
                gen = func(*inputs, context=database.context)
            except Exception:
                return []

            results = []
            out_arg = goal.arguments[output_arg]
            for value in gen:
                result_term = python_to_prolog(value)
                new_subst = out_arg.unify(result_term, subst.copy())
                if new_subst is not None:
                    results.append(new_subst)
            return results

        _tag(wrapper, name, arity, func, description, "generator")
        return wrapper
    return decorator


__all__ = ['simple_action', 'effector_action', 'generator_action']