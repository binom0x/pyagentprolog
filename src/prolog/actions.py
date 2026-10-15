"""PyAgentProlog — canonical action decorators.

The decorators let you hook an ordinary Python function into Prolog as a
built-in predicate.

Usage:
    from prolog.actions import simple_action, effector_action, generator_action

Compatibility:
    from prolog.agent import simple_action        # same object
    from neuroagent.actions import simple_action  # same object

Metadata attached to the wrapper:
    _is_prolog_predicate, _predicate_name, _predicate_arity
        → for BuiltinRegistry.register_module
    _agent_action, _action_name, _action_arity,
    _action_description, _action_type
        → for Agent.register_action / register_action_module

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""
from __future__ import annotations
import functools
from typing import Optional

from prolog.terms import Term, Variable, Number
from prolog.bridge import prolog_to_python, python_to_prolog


def _tag(wrapper, name: str, arity: int, func, description: Optional[str],
         action_type: str) -> None:
    """Tag the wrapper with action and predicate metadata.

    After _tag, the function is suitable both for register_action_module
    (via _agent_action) and for BuiltinRegistry.register_module
    (via _is_prolog_predicate).
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
    """Action returning a single value.

    Args:
        name: predicate name in Prolog.
        arity: total number of arguments.
        output_arg: index (0-based) of the output argument where the
            result is placed. -1 → effector (side effect, no return value).
        description: description (defaults to the function's docstring).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(goal, database, subst, depth):
            if len(goal.arguments) != arity:
                return []

            # Effector
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

            # Input arguments
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
    """Effector action: side effect only, no return value.

    All arguments are treated as inputs and must be instantiated.
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
    """Generator action: the function returns an iterator.

    Each value of the iterator produces a separate Prolog solution.

    If output_arg == -1, the action is treated as an effector: the function
    is called once, and the iterator is ignored (compatible with
    effector_action).
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(goal, database, subst, depth):
            if len(goal.arguments) != arity:
                return []

            # Effector
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

            # Input arguments
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