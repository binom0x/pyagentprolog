"""PyAgentProlog — canonical Prolog ↔ Python bridge.

The single conversion point between Prolog terms and Python objects.

Usage:
    from prolog.bridge import prolog_to_python, python_to_prolog

Compatibility:
    from prolog.agent import prolog_to_python       # same object
    from neuroagent.bridge import prolog_to_python  # same object

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""
from __future__ import annotations
from typing import Any, List

from prolog.terms import Term, Variable, Number


def prolog_list_to_python(term: Any) -> Any:
    """Converts a Prolog list (Term with . or []) into a Python list of terms.

    Differs from prolog_to_python: returns a LIST OF TERMS (Term), not
    Python values. If term is not a list, returns it as is.
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
    """Converts a Prolog term into a Python object.

    Mappings:
        Number            → int/float
        Variable          → None
        Quoted string     → str (without quotes)
        Atom              → str
        List              → list
        Compound term     → str
    """
    if isinstance(term, Number):
        return term.value
    if isinstance(term, Variable):
        return None
    if isinstance(term, Term):
        # Empty list
        if term.functor == '[]' and not term.arguments:
            return []
        # Non-empty list
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
        # Atom (possibly a number)
        if not term.arguments:
            f = term.functor
            if not isinstance(f, str):
                return f
            # Try to interpret as a number
            try:
                if '.' in f or 'e' in f.lower():
                    return float(f)
                return int(f)
            except ValueError:
                pass
            # Quoted string — strip the quotes
            if len(f) >= 2 and f[0] == '"' and f[-1] == '"':
                return f[1:-1]
            if len(f) >= 2 and f[0] == "'" and f[-1] == "'":
                return f[1:-1]
            return f
        # Other compound terms — string representation
        return str(term)
    return str(term)


def python_to_prolog(obj: Any) -> Term:
    """Converts a Python object into a Prolog term.

    Mappings:
        Term         → as is
        bool         → atom true/false
        int/float    → Number
        str          → atom in double quotes
        list         → Prolog list
        dict         → list of pairs '-'(K, V)
        None         → empty list []
        other        → atom in double quotes (str(obj))
    """
    if isinstance(obj, Term):
        return obj
    # bool BEFORE int: True/False must be atoms, not numbers
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