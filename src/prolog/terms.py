""" PyAgentProlog

Base classes for terms with optimized unification (occurs_check disabled by
default) and reduced memory consumption (__slots__).

A pluggable unifier is supported (e.g., semantic on embeddings):
the public Term.unify(...) is a dispatcher entry point, while strict
unification is moved to Term.unify_strict(...) and does not go back into
the custom unifier.

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""

import sys
import threading

DEBUG_UNIFY = False   # Debugging can be enabled if needed


def debug(*args, indent=0, **kwargs):
    if DEBUG_UNIFY:
        print("  " * indent + "[UNIFY]", *args, **kwargs)


# =====================================================================
# Pluggable unifier
# =====================================================================
_custom_unifier = None
_thread_local = threading.local()


def set_custom_unifier(unifier) -> None:
    """Install a user-defined unifier (e.g., EmbeddingUnifier)."""
    global _custom_unifier
    _custom_unifier = unifier


def get_custom_unifier():
    """Get the current user-defined unifier (or None)."""
    return _custom_unifier


def clear_custom_unifier() -> None:
    """Revert to the default strict unification."""
    global _custom_unifier
    _custom_unifier = None
# =====================================================================


class Term:
    """Base class for Prolog terms"""
    __slots__ = ('functor', 'arguments')

    def __init__(self, functor, arguments=None):
        self.functor = functor
        self.arguments = arguments or []

    # -----------------------------------------------------------------
    # Public entry point — dispatcher
    # -----------------------------------------------------------------
    def unify(self, other, subst=None, depth=0, occurs_check=False):
        """
        Unification of terms.

        If a custom unifier (e.g., semantic on embeddings) is installed
        and we are not inside it — call it.
        Otherwise, strict unification unify_strict is performed.

        The occurs_check parameter controls the occurrence check (disabled by default).
        """
        if subst is None:
            subst = {}
        custom = _custom_unifier
        if custom is not None and not getattr(_thread_local, 'inside', False):
            _thread_local.inside = True
            try:
                return custom.unify(self, other, subst, depth, occurs_check)
            finally:
                _thread_local.inside = False
        return self.unify_strict(other, subst, depth, occurs_check)

    # -----------------------------------------------------------------
    # Strict unification (original algorithm unchanged)
    # -----------------------------------------------------------------
    def unify_strict(self, other, subst=None, depth=0, occurs_check=False):
        """
        Strict unification of terms (without calling the custom unifier).
        The occurs_check parameter controls the occurrence check (disabled by default).
        """
        if subst is None:
            subst = {}

        debug(f"unify({self}, {other})", indent=depth)
        debug(f"  subst = { {str(k):str(v) for k,v in subst.items()} }", indent=depth)

        # If this is the same variable
        if self is other:
            debug(f"  same reference -> {subst}", indent=depth)
            return subst

        # If self is a variable
        if isinstance(self, Variable):
            debug(f"  self is variable {self.name}", indent=depth)
            res = self.unify_with(other, subst, depth + 1, occurs_check)
            debug(f"  variable unification result: {res}", indent=depth)
            return res

        # If other is a variable
        if isinstance(other, Variable):
            debug(f"  other is variable {other.name}", indent=depth)
            res = other.unify_with(self, subst, depth + 1, occurs_check)
            debug(f"  variable unification result: {res}", indent=depth)
            return res

        # If self is a number
        if isinstance(self, Number):
            debug(f"  self is number {self.value}", indent=depth)
            if isinstance(other, Number):
                if self.value == other.value:
                    debug(f"    numbers are equal -> {subst}", indent=depth)
                    return subst
            elif isinstance(other, Term):
                try:
                    val = float(other.functor) if '.' in other.functor else int(other.functor)
                    if self.value == val:
                        debug(f"    term {other} coerces to number {val} -> {subst}", indent=depth)
                        return subst
                except:
                    pass
            debug(f"    numbers/terms not equal -> None", indent=depth)
            return None

        # If other is a number
        if isinstance(other, Number):
            debug(f"  other is number {other.value}", indent=depth)
            return other.unify_strict(self, subst, depth + 1, occurs_check)

        # If both are ordinary terms
        if isinstance(self, Term) and isinstance(other, Term):
            # Special handling of lists
            if self.functor == '.' and other.functor == '.':
                debug(f"  list unification: {self} vs {other}", indent=depth)
                head1, tail1 = self.arguments[0], self.arguments[1]
                head2, tail2 = other.arguments[0], other.arguments[1]
                debug(f"    head1={head1}, head2={head2}", indent=depth + 1)
                debug(f"    tail1={tail1}, tail2={tail2}", indent=depth + 1)
                subst1 = head1.unify_strict(head2, subst, depth + 1, occurs_check)
                if subst1 is None:
                    debug(f"    head unification failed -> None", indent=depth)
                    return None
                debug(f"    after head unification: {subst1}", indent=depth + 1)
                subst2 = tail1.unify_strict(tail2, subst1, depth + 1, occurs_check)
                if subst2 is None:
                    debug(f"    tail unification failed -> None", indent=depth)
                    return None
                debug(f"    list unification succeeded -> {subst2}", indent=depth)
                return subst2

            # Ordinary terms
            debug(f"  both are ordinary terms: {self.functor}/{len(self.arguments)} vs {other.functor}/{len(other.arguments)}",
                  indent=depth)
            if self.functor != other.functor or len(self.arguments) != len(other.arguments):
                debug(f"    functors or arities do not match -> None", indent=depth)
                return None

            current_subst = subst.copy()
            for i, (arg1, arg2) in enumerate(zip(self.arguments, other.arguments)):
                debug(f"    unifying argument {i}: {arg1} vs {arg2}", indent=depth + 1)
                result = arg1.unify_strict(arg2, current_subst, depth + 1, occurs_check)
                if result is None:
                    debug(f"    argument {i} not unified -> None", indent=depth + 1)
                    return None
                current_subst = result
                debug(f"    substitution after argument {i}: {current_subst}", indent=depth + 1)

            debug(f"  term unification succeeded -> {current_subst}", indent=depth)
            return current_subst

        debug(f"  failed to unify -> None", indent=depth)
        return None

    def substitute(self, subst):
        """Substitution of variable values (without cycle protection)"""
        if isinstance(subst, dict):
            if isinstance(self, Variable):
                return self.substitute_from_dict(subst)
            elif isinstance(self, Number):
                return self
            elif isinstance(self, Term):
                new_args = [arg.substitute(subst) for arg in self.arguments]
                return Term(self.functor, new_args)
        return self

    def substitute_from_dict(self, subst, visited=None):
        """Substitution from a dict with cycle protection"""
        if visited is None:
            visited = set()
        if isinstance(self, Variable):
            if self in subst:
                if self in visited:
                    # Cyclic reference — return the variable as is
                    return self
                visited.add(self)
                value = subst[self]
                return value.substitute_from_dict(subst, visited)
            return self
        elif isinstance(self, Number):
            return self
        elif isinstance(self, Term):
            new_args = [arg.substitute_from_dict(subst, visited) for arg in self.arguments]
            return Term(self.functor, new_args)
        return self

    def is_list(self):
        """Checks whether the term is a list"""
        if self.functor == '[]' and len(self.arguments) == 0:
            return True
        elif self.functor == '.' and len(self.arguments) == 2:
            return self.arguments[1].is_list()
        return False

    def get_list_elements(self):
        """Returns the elements of the list if the term is a list"""
        elements = []
        current = self
        while isinstance(current, Term) and current.functor == '.' and len(current.arguments) == 2:
            elements.append(current.arguments[0])
            current = current.arguments[1]
        if isinstance(current, Term) and current.functor == '[]' and len(current.arguments) == 0:
            return elements
        else:
            return None

    def __str__(self):
        # Conjunction handling
        if self.functor == ',' and len(self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            return f"{left}, {right}"
        # Disjunction handling
        if self.functor == ';' and len(self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            return f"{left} ; {right}"
        # Negation handling
        if self.functor == '\\+' and len(self.arguments) == 1:
            return f"\\+({self.arguments[0]})"
        # Arithmetic operator handling
        if self.functor in ['+', '-', '*', '/', 'is', '=', '=:=', r'=\=', '<', '>', '=<', '>='] and len(
                self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            precedence = {
                'is': 1,
                '*': 2, '/': 2,
                '+': 3, '-': 3,
                '=:=': 4, r'=\=': 4, '<': 4, '>': 4, '=<': 4, '>=': 4,
                '=': 5
            }
            if isinstance(self.arguments[0], Term) and self.arguments[0].functor in precedence:
                if precedence.get(self.arguments[0].functor, 10) > precedence.get(self.functor, 10):
                    left = f"({left})"
            if isinstance(self.arguments[1], Term) and self.arguments[1].functor in precedence:
                if precedence.get(self.arguments[1].functor, 10) > precedence.get(self.functor, 10):
                    right = f"({right})"
                elif self.functor in ['-', '/'] and precedence.get(self.arguments[1].functor, 10) >= precedence.get(
                        self.functor, 10):
                    right = f"({right})"
            return f"{left} {self.functor} {right}"
        # List handling
        if self.functor == '.' and len(self.arguments) == 2:
            elements = []
            current = self
            while isinstance(current, Term) and current.functor == '.' and len(current.arguments) == 2:
                elements.append(str(current.arguments[0]))
                current = current.arguments[1]
            if isinstance(current, Term) and current.functor == '[]' and len(current.arguments) == 0:
                return '[' + ','.join(elements) + ']'
            else:
                return '[' + ','.join(elements) + '|' + str(current) + ']'
        # Quoted string handling
        if len(self.arguments) == 0:
            functor_str = str(self.functor)
            if (functor_str.startswith("'") and functor_str.endswith("'")) or \
                    (functor_str.startswith('"') and functor_str.endswith('"')):
                return functor_str
            return functor_str
        else:
            args_str = ', '.join(str(arg) for arg in self.arguments)
            return f"{self.functor}({args_str})"

    def __repr__(self):
        return str(self)


class Variable(Term):
    """Prolog variable with optimized unification (occurs_check disabled by default)"""
    __slots__ = ('name',)

    def __init__(self, name):
        super().__init__(name, [])   # <-- call the Term constructor to set attributes
        self.name = name

    def unify_with(self, other, subst, depth=0, occurs_check=False):
        """
        Unify a variable with another term.
        The occurs_check parameter enables the occurrence check (default False).

        This method is internal for strict unification: it is called only
        from Term.unify_strict and does not go into the custom unifier.
        """
        debug(f"unify_with: {self.name} with {other}", indent=depth)

        # An anonymous variable always unifies
        if self.name == "_":
            debug(f"  anonymous variable -> {subst}", indent=depth)
            return subst

        # If the variable already has a value in the substitution
        if self in subst:
            debug(f"  variable {self.name} is already bound to {subst[self]}", indent=depth)
            res = subst[self].unify_strict(other, subst, depth + 1, occurs_check)
            debug(f"  recursive unification result: {res}", indent=depth)
            return res

        # If other is a variable that already has a value
        if isinstance(other, Variable) and other in subst:
            debug(f"  other is variable {other.name}, already bound to {subst[other]}", indent=depth)
            res = self.unify_with(subst[other], subst, depth + 1, occurs_check)
            debug(f"  recursive unification result: {res}", indent=depth)
            return res

        # If other is a variable, and the very same one (by object or by name)
        if isinstance(other, Variable):
            if self is other or self.name == other.name:
                debug(f"  unifying variable {self.name} with itself -> {subst}", indent=depth)
                return subst

        # Occurs-check — disabled by default for performance
        if occurs_check and self.occurs_in(other, subst):
            debug(f"  occurs_check failed: {self.name} occurs in {other}", indent=depth)
            return None

        # Create a new substitution
        new_subst = subst.copy()
        new_subst[self] = other
        debug(f"  new substitution created: { {str(k):str(v) for k,v in new_subst.items()} }", indent=depth)
        return new_subst

    def occurs_in(self, term, subst):
        """Check if the variable occurs in the term (used only with occurs_check=True)"""
        if self is term:
            return False
        if self == term:
            return False
        if isinstance(term, Variable) and term in subst:
            return self.occurs_in(subst[term], subst)
        if isinstance(term, Term):
            for arg in term.arguments:
                if self.occurs_in(arg, subst):
                    return True
        return False

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class Number(Term):
    """Prolog number"""
    __slots__ = ('value',)

    def __init__(self, value):
        super().__init__(str(value), [])   # <-- call the Term constructor
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Number):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, Number):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, Number):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, Number):
            return self.value >= other.value
        return self.value >= other

    def __hash__(self):
        return hash(self.value)


class Rule:
    """Prolog rule"""
    __slots__ = ('head', 'body')

    def __init__(self, head, body):
        self.head = head
        self.body = body

    def __str__(self):
        if isinstance(self.body, Term) and self.body.functor == "true":
            return f"{self.head}."
        else:
            return f"{self.head} :- {self.body}."

    def __repr__(self):
        return str(self)