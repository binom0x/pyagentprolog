""" PyAgentProlog

Framework for building AI agents on PyAgentProlog with Python actions.

Canonical implementations have been moved out:
    - prolog_to_python, python_to_prolog, prolog_list_to_python
      → prolog.bridge
    - simple_action, effector_action, generator_action
      → prolog.actions

This module re-exports them for backward compatibility and
provides the Agent class and its built-in predicates set_state/1,
update_state/1.

(c) 2026 Andrey Vorfolomeev

version 1.0.1 (canonical bridge/actions)
"""

import pickle
import gzip
import functools
import sys
import ast
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator

from prolog.solver import Solver
from prolog.parser import Parser
from prolog.terms import Term, Variable, Number, Rule
from prolog.predicates import prolog_predicate

# Canonical implementations live in prolog.bridge / prolog.actions.
# Re-export for backward compatibility:
#   from prolog.agent import prolog_to_python  → prolog.bridge.prolog_to_python
#   from prolog.agent import simple_action     → prolog.actions.simple_action
from prolog.bridge import (
    prolog_to_python,
    python_to_prolog,
    prolog_list_to_python,
)
from prolog.actions import (
    simple_action,
    effector_action,
    generator_action,
)

DEBUG = False


# ------------------------------------------------------------
# Legacy @agent_action decorator (backward compatibility)
# ------------------------------------------------------------
def agent_action(name: str, arity: int, description: str = None,
                 action_type: str = "generic"):
    """Legacy decorator for registering a Python function as a
    Prolog predicate-action of an agent.

    Kept for backward compatibility. New code should use
    simple_action / effector_action / generator_action
    from prolog.actions.

    Returns a wrapper function with metadata:
        _agent_action, _action_name, _action_arity,
        _action_description, _action_type,
        _is_prolog_predicate, _predicate_name, _predicate_arity.
    """
    def decorator(func):
        wrapped = prolog_predicate(name, arity)(func)
        wrapped._agent_action = True
        wrapped._action_name = name
        wrapped._action_arity = arity
        wrapped._action_description = description
        wrapped._action_type = action_type

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if DEBUG:
                print(f"Agent action called: {name}/{arity}")
            return func(*args, **kwargs)

        # functools.wraps copied __dict__ from func, where prolog_predicate
        # and subsequent assignments placed all the metadata. Therefore the
        # wrapper also has them and will be found by register_action_module.
        return wrapper
    return decorator


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def is_ground(term) -> bool:
    """Checks whether the term contains no free variables."""
    if isinstance(term, Variable):
        return False
    elif isinstance(term, Number):
        return True
    elif isinstance(term, Term):
        return all(is_ground(arg) for arg in term.arguments)
    else:
        return True


# ------------------------------------------------------------
# Built-in agent predicates (actions)
# ------------------------------------------------------------
@agent_action('set_state', 1,
              description="Replace all facts with the same functor/arity "
                          "by this ground fact")
def set_state(goal, database, subst, depth):
    """Replaces all facts with the given functor/arity with the new fact."""
    if len(goal.arguments) != 1:
        return []

    term = goal.arguments[0].substitute(subst)
    if not isinstance(term, Term) or not is_ground(term):
        return []

    from prolog.terms import Variable

    # Create a head template with variables
    functor = term.functor
    arity = len(term.arguments)
    vars = [Variable('_') for _ in range(arity)]
    head_template = Term(functor, vars)
    template_rule = Rule(head_template, Term('true', []))

    # Remove all matching facts
    database.retract_all(template_rule)

    # Add the new fact
    database.add_rule(Rule(term, Term('true', [])))

    return [subst.copy()]


@agent_action('update_state', 1,
              description="Add a fact (without deleting old ones)")
def update_state(goal, database, subst, depth):
    """Adds a fact without deleting the old ones."""
    if len(goal.arguments) != 1:
        return []

    term = goal.arguments[0].substitute(subst)
    if not isinstance(term, Term) or not is_ground(term):
        return []

    database.add_rule(Rule(term, Term('true', [])))

    return [subst.copy()]


# ------------------------------------------------------------
# Main Agent class
# ------------------------------------------------------------
class Agent:
    """
    AI agent driven by a Prolog program with the ability to call
    Python actions via user-defined predicates.
    """

    def __init__(self, program: str = "", pl_file: str = "",
                 context: dict = None):
        """
        Initializes the agent.

        Args:
            program: Prolog program text.
            pl_file: File with a Prolog program to load.
            context: Dictionary with external resources (LLM clients, APIs, etc.)
                     accessible from actions via database.context.
        """
        pl_prog = program
        if not program and pl_file:
            with open(pl_file, encoding='utf-8') as f:
                pl_prog = f.read()

        self.solver = Solver(pl_prog, context=context)
        self._actions: Dict[Tuple[str, int], Dict] = {}

        # Automatically register built-in agent predicates
        self._register_builtins()

    def _register_builtins(self):
        """Registers the built-in predicates set_state/1 and update_state/1."""
        current_module = sys.modules[__name__]
        self.register_action_module(current_module)

    def register_action(self, func):
        """Registers a single action decorated with @agent_action
        or with the new decorators (simple_action / effector_action /
        generator_action)."""
        if not hasattr(func, '_agent_action'):
            raise ValueError("Function is not decorated with @agent_action")

        name = func._action_name
        arity = func._action_arity
        self.solver.database.register_custom_predicate(name, arity, func)
        self._actions[(name, arity)] = {
            'description': func._action_description,
            'type': func._action_type,
            'function': func
        }
        if DEBUG:
            print(f"Registered agent action: {name}/{arity}")

    def register_action_module(self, module):
        """Registers all actions from the given module."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and hasattr(attr, '_agent_action'):
                self.register_action(attr)

    def list_actions(self) -> List[Dict]:
        """Returns a list of registered actions with metadata."""
        return [
            {
                'name': name,
                'arity': arity,
                **meta
            }
            for (name, arity), meta in self._actions.items()
        ]

    def query(self, query_text: str, with_explanation: bool = False):
        """
        Executes a Prolog query.

        Args:
            query_text: Query text (must end with a period).
            with_explanation: If True, also returns an explanation object.

        Returns:
            If with_explanation=False: a dictionary with results.
            If with_explanation=True: a tuple (dictionary with results,
                explanation object).
        """
        result = self.solver.find_solutions(
            query_text, with_explanation=with_explanation
        )
        if with_explanation:
            solutions, explanation = result
            parsed = self._parse_solutions(query_text, solutions)
            return parsed, explanation
        else:
            return self._parse_solutions(query_text, result)

    def query_one(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Returns the first solution of the query, or None.

        Distinguishes "query failed" from "query succeeded but without free
        variables". In the first case, returns None; in the second, a dict
        with the marker '__success__'.
        """
        result = self.query(query_text)
        if isinstance(result, dict):
            # Success without free variables: query() returned True.
            if result.get('solution') is True:
                return {'__success__': True}
            if ('solutions' in result
                    and isinstance(result['solutions'], list)
                    and result['solutions']):
                return result['solutions'][0]
            if 'solution' in result:
                sol = result['solution']
                if isinstance(sol, list) and sol:
                    return sol[0]
                elif isinstance(sol, dict):
                    return sol
        return None

    def _parse_solutions(self, query_text, raw_solutions):
        """Overrides solution parsing to return Python objects."""
        from prolog.solver import ParseSolution
        # Standard parsing (dict with keys 'solution', 'output', etc.)
        parsed = ParseSolution(query_text, raw_solutions, explanation=None)
        # Convert string values to Python objects
        if 'solution' in parsed:
            sol = parsed['solution']
            if isinstance(sol, dict):
                parsed['solution'] = self._convert_solution(sol)
            elif isinstance(sol, list):
                parsed['solution'] = [
                    self._convert_solution(s) for s in sol if s is not None
                ]
        return parsed

    def _convert_solution(self, sol_dict):
        """
        Recursively converts string values to Python objects
        (numbers, lists). Uses ast.literal_eval for safe
        parsing of literals.
        """
        if not isinstance(sol_dict, dict):
            return sol_dict
        converted = {}
        for var, val in sol_dict.items():
            if isinstance(val, str):
                try:
                    # ast.literal_eval safely parses numbers, strings,
                    # lists, tuples, dicts, etc.
                    parsed = ast.literal_eval(val)
                    val = parsed
                except (ValueError, SyntaxError):
                    # If parsing fails, keep it as a string
                    pass
            converted[var] = val
        return converted

    def add_fact(self, fact_text: str):
        """Adds a single fact to the knowledge base (compatibility).

        Delegates to add_facts to avoid duplicating parsing logic.
        """
        self.add_facts(fact_text)

    def add_facts(self, facts_text: str):
        """Adds a batch of facts/rules in a SINGLE parser call.

        Unlike add_fact, it is designed for multi-line text: each
        rule is separated by a period. This gives ~3–5× speedup when
        loading 5+ facts, because Parser is created and performs
        regex tokenization once for the whole batch, not per fact.

        Example:
            agent.add_facts('''
                my_pos(5.0, 5.0).
                my_mode(search).
                observed(bravo, search, 10.0, 5.0, 17).
            ''')
        """
        text = facts_text.strip()
        if not text:
            return
        if not text.endswith('.'):
            text = text + '.'
        rules = Parser(text).parse_rules()
        for rule in rules:
            self.solver.database.add_rule(rule)
            if DEBUG:
                print(f"Added fact: {rule}")

    def retract_fact(self, fact_template: str) -> Optional[Dict]:
        """Removes the first rule that unifies with the template."""
        if not fact_template.strip().endswith('.'):
            fact_template = fact_template.strip() + '.'
        rules = Parser(fact_template).parse_rules()
        if not rules:
            return None
        template_rule = rules[0]
        return self.solver.database.retract_first(template_rule)

    def retract_all(self, fact_template: str) -> int:
        """Removes all rules that unify with the template."""
        if not fact_template.strip().endswith('.'):
            fact_template = fact_template.strip() + '.'
        rules = Parser(fact_template).parse_rules()
        if not rules:
            return 0
        template_rule = rules[0]
        return self.solver.database.retract_all(template_rule)

    @property
    def context(self) -> dict:
        return self.solver.database.context

    @context.setter
    def context(self, value: dict):
        self.solver.database.context = value

    def stream(self, query_text: str, with_explanation: bool = False):
        """
        Executes a Prolog query and returns a generator that yields
        solutions as they are found. Useful for long-running queries or
        when results need to be processed in real time.

        Args:
            query_text: Query text (must end with a period).
            with_explanation: Reserved (not implemented yet).

        Yields:
            - If the query contains variables: a dict
              {variable_name: value} for each solution.
            - If the query has no variables: True on success,
              False on failure.
            - On error: a dict with the key 'error'.
        """
        from prolog.parser import Parser
        from prolog.terms import Variable

        try:
            query = Parser(query_text).parse_query()
        except Exception as e:
            yield {'error': f'Query parse error: {e}'}
            return

        # Find all variables in the query
        variables = {}

        def find_vars(term):
            if isinstance(term, Variable) and term.name != '_':
                variables[term.name] = term
            elif isinstance(term, Term):
                for arg in term.arguments:
                    find_vars(arg)

        find_vars(query)

        has_vars = bool(variables)
        found = False

        try:
            # Use the internal solution generator of the knowledge base
            for subst in self.solver.database._solve(
                query, self.solver.database.rule_index, {}, 0
            ):
                found = True
                if has_vars:
                    solution = {}
                    for var_name, var in variables.items():
                        val = var.substitute(subst)
                        solution[var_name] = prolog_to_python(val)
                    yield solution
                else:
                    yield True
        except Exception as e:
            yield {'error': f'Query execution error: {e}'}
        finally:
            if not has_vars and not found:
                yield False

    def __repr__(self):
        return f"Agent(actions={len(self._actions)})"

    def save_kb(self, filename: str, format: str = 'prolog') -> None:
        """
        Saves the current knowledge base to a file in the given format.

        Args:
            filename: Name of the file to save.
            format:   'prolog' – text Prolog format (each rule
                      ending with a period);
                      'binary' – binary format (pickle) for fast
                      loading.
        """
        if format == 'prolog':
            with open(filename, 'w', encoding='utf-8') as f:
                for rule in self.solver.database.rules:
                    f.write(str(rule) + '\n')
            if DEBUG:
                print(f"Knowledge base saved in text format: {filename}")

        elif format == 'binary':
            # Serialize the list of rules with a version for future changes
            data = {'version': 1, 'rules': self.solver.database.rules}
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            if DEBUG:
                print(f"Knowledge base saved in binary format: {filename}")

        else:
            raise ValueError("format must be 'prolog' or 'binary'")

    def load_kb(self, filename: str, format: str = 'prolog',
                clear: bool = False) -> None:
        """
        Loads the knowledge base from a file in the given format.

        Args:
            filename: Name of the file to load.
            format:   'prolog' – text Prolog format (parsed);
                      'binary' – binary format (pickle, no parsing).
            clear:    If True, the current base is fully cleared
                      before loading.
        """
        if clear:
            self.clear_kb()

        if format == 'prolog':
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            from prolog.parser import Parser
            try:
                rules = Parser(content).parse_rules()
            except Exception as e:
                raise ValueError(
                    f"Parse error in file {filename}: {e}"
                )

            for rule in rules:
                self.solver.database.add_rule(rule)

            if DEBUG:
                print(f"Loaded {len(rules)} rules from text file "
                      f"{filename}")

        elif format == 'binary':
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Error loading binary file {filename}: {e}"
                )

            if not isinstance(data, dict) or data.get('version') != 1:
                raise ValueError(
                    "Invalid binary file format or unsupported version"
                )

            rules = data['rules']
            for rule in rules:
                self.solver.database.add_rule(rule)

            if DEBUG:
                print(f"Loaded {len(rules)} rules from binary file "
                      f"{filename}")

        else:
            raise ValueError("format must be 'prolog' or 'binary'")

    def clear_kb(self) -> None:
        """Fully clears the knowledge base."""
        self.solver.database.rules.clear()
        self.solver.database.build_rule_index()
        if DEBUG:
            print("Knowledge base cleared")


# ------------------------------------------------------------
# Export
# ------------------------------------------------------------
# Original __all__ preserved (backward compatibility for
# `from prolog.agent import *`). Re-exported bridge functions
# are available via explicit import:
#     from prolog.agent import prolog_to_python, python_to_prolog
__all__ = [
    'Agent',
    'agent_action',
    'simple_action',
    'effector_action',
    'generator_action',
]