# Complete Guide to PyAgentProlog

**Purpose:** A Prolog engine in Python with agents, tracing, and semantic unification

---

# Table of Contents

1. [Introduction](#1-introduction)
2. [Installation and Quick Start](#2-installation-and-quick-start)
3. [Package Architecture](#3-package-architecture)
4. [Module terms — Terms and Unification](#4-module-terms--terms-and-unification)
5. [Module parser — Prolog Syntax](#5-module-parser--prolog-syntax)
6. [Module interpreter — Solver Core](#6-module-interpreter--solver-core)
7. [Module predicates — Built-in Predicates](#7-module-predicates--built-in-predicates)
8. [Module solver — High-Level Interface](#8-module-solver--high-level-interface)
9. [Module bridge — Prolog ↔ Python Bridge](#9-module-bridge--prolog--python-bridge)
10. [Module actions — Python Actions](#10-module-actions--python-actions)
11. [Module agent — Agent Class](#11-module-agent--agent-class)
12. [Module tracer — Proof Tracing](#12-module-tracer--proof-tracing)
13. [Module embedding_unify — Semantic Unification](#13-module-embedding_unify--semantic-unification)
14. [Module arith_numba — Arithmetic Acceleration](#14-module-arith_numba--arithmetic-acceleration)
15. [Module prologchain — LangChain Integration](#15-module-prologchain--langchain-integration)
16. [Module proagent_gradio — Gradio Sessions](#16-module-proagent_gradio--gradio-sessions)
17. [Prolog Subset Reference](#17-prolog-subset-reference)
18. [Practical Examples](#18-practical-examples)
19. [Debugging and Diagnostics](#19-debugging-and-diagnostics)
20. [Best Practices](#20-best-practices)
21. [Limitations and Known Features](#21-limitations-and-known-features)

---

# 1. Introduction

## 1.1. What Is PyAgentProlog

**PyAgentProlog** is a standalone Prolog engine implemented in pure Python, independent of external logic systems. The package does not require SWI-Prolog, GNU Prolog, or any other external solvers — the entire core runs on Python.

Key features:

- **Logical inference**: a full cycle of unification, backtracking search, and cut.
- **Iterative solver with TCO**: tail-call optimization, elimination of stack overflow.
- **Built-in and meta-predicates**: arithmetic, lists, findall/setof/bagof, call/N, forall.
- **Proof tracing**: construction of a derivation tree with unifications and rules.
- **Agents with Python actions**: connecting ordinary Python functions as Prolog predicates.
- **Semantic unification**: comparing atoms via embeddings (Ollama, sentence-transformers, custom).
- **LangChain integration**: Runnable chains with Prolog queries.
- **Gradio sessions**: interactive UI sessions for the agent.
- **Numba arithmetic acceleration**: JIT compilation of large arithmetic expressions.
- **Optional dependencies**: the core works without numpy, langchain, gradio, or sentence-transformers.

## 1.2. What to Use It For

- Expert systems with logical inference.
- Neuro-symbolic agents (LLM + logic).
- Learning Prolog in Python.
- Rapid prototyping of rules and ontologies.
- Natural language processing with semantic unification.
- Integrating logical inference into LangChain pipelines.

## 1.3. License and Authorship

The package is distributed with attribution to Andrey Vorfolomeev. Version 1.0.1.

---

# 2. Installation and Quick Start

## 2.1. Requirements

- Python 3.8+
- No mandatory external dependencies for the core.

Optional dependencies:

| Feature | Package |
|---|---|
| Numba arithmetic acceleration | `numpy`, `numba` |
| Semantic unification via Ollama | `langchain-ollama` or `langchain-community` |
| Semantic unification via ST | `sentence-transformers` |
| LangChain integration | `langchain-core`, `pydantic` |
| Gradio sessions | `gradio` (for UI) |

## 2.2. Installation

```bash
# Core (no dependencies)
pip install pyagentprolog

# With Numba
pip install pyagentprolog[numba]

# With semantics (sentence-transformers)
pip install pyagentprolog[semantic]

# With LangChain
pip install pyagentprolog[langchain]

# Everything at once
pip install pyagentprolog[all]
```

If the package is distributed as a local module, place the `prolog/` directory in `PYTHONPATH`.

## 2.3. Quick Start

```python
from prolog.agent import Agent

agent = Agent("""
parent(ivan, alexey).
parent(ivan, maria).
parent(alexey, petr).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
""")

print(agent.query("grandparent(ivan, X)."))
# {'solution': [{'X': 'petr'}]}
```

## 2.4. Minimal Sanity Check

```python
from prolog.solver import Solver

solver = Solver("""
fact(1).
fact(2).
fact(3).
""")

print(solver.find_solutions("fact(X)."))
# [{'X': 1}, {'X': 2}, {'X': 3}]
```

---

# 3. Package Architecture

## 3.1. Layers

```
┌─────────────────────────────────────────────────────────┐
│  Application layer                                      │
│  proagent_gradio, prologchain, agent                    │
├─────────────────────────────────────────────────────────┤
│  Actions layer                                          │
│  actions (simple/effector/generator)                    │
├─────────────────────────────────────────────────────────┤
│  Unification layer                                      │
│  terms.unify → embedding_unify (optional)               │
├─────────────────────────────────────────────────────────┤
│  Inference layer                                        │
│  solver → interpreter (Database, Frame, TCO, cut)       │
├─────────────────────────────────────────────────────────┤
│  Predicates layer                                       │
│  predicates, arith_numba                                │
├─────────────────────────────────────────────────────────┤
│  Parsing layer                                          │
│  parser → terms                                         │
└─────────────────────────────────────────────────────────┘
```

## 3.2. Import Order

In `__init__.py`, a strict order is maintained:

```
terms → bridge → parser → tracer → predicates
      → interpreter → solver → actions → agent
```

Optional modules (`embedding_unify`, `prologchain`, `proagent_gradio`) are imported inside `try/except ImportError`.

## 3.3. Key Entities

| Entity | Module | Purpose |
|---|---|---|
| `Term` | `terms` | Compound Prolog term |
| `Variable` | `terms` | Prolog variable |
| `Number` | `terms` | Number |
| `Rule` | `terms` | Rule (head + body) |
| `Parser` | `parser` | Prolog text parser |
| `Database` | `interpreter` | Knowledge base + solver |
| `BuiltinRegistry` | `interpreter` | Built-in predicate registry |
| `Frame` | `interpreter` | Search stack frame |
| `Solver` | `solver` | High-level solver |
| `Agent` | `agent` | Agent with actions |
| `ProofTracer` | `tracer` | Tracer |
| `EmbeddingUnifier` | `embedding_unify` | Semantic unifier |

---

# 4. Module terms — Terms and Unification

## 4.1. Term Classes

### 4.1.1. `Term`

Base class for compound terms.

```python
from prolog.terms import Term

t = Term('parent', [Term('ivan'), Term('alexey')])
print(t)           # parent(ivan, alexey)
print(t.functor)   # parent
print(t.arguments) # [Term(ivan), Term(alexey)]
```

- `Term(functor, arguments=None)` — constructor.
- `functor` — a string (functor name) or another value.
- `arguments` — a list of subterms.

### 4.1.2. `Variable`

Prolog variable. Inherits from `Term`.

```python
from prolog.terms import Variable

x = Variable('X')
print(x.name)  # X
```

- `Variable(name)` — constructor.
- A variable named `_` is anonymous.
- `__eq__` and `__hash__` are based on the name.

### 4.1.3. `Number`

Prolog number.

```python
from prolog.terms import Number

n = Number(42)
f = Number(3.14)
print(n.value)   # 42
print(f.value)   # 3.14
```

- `Number(value)` — constructor.
- Supports `int` and `float`.
- Compared with numbers via `__eq__`, `__lt__`, etc.

### 4.1.4. `Rule`

Prolog rule.

```python
from prolog.terms import Term, Rule

head = Term('ancestor', [Variable('X'), Variable('Y')])
body = Term('parent', [Variable('X'), Variable('Y')])
rule = Rule(head, body)
print(rule)  # ancestor(X, Y) :- parent(X, Y).
```

- `Rule(head, body)` — constructor.
- Fact: `body = Term('true', [])`.
- `__str__` returns a correct representation.

## 4.2. Unification

### 4.2.1. Public API

```python
Term.unify(other, subst=None, depth=0, occurs_check=False)
```

- `other` — the second term.
- `subst` — substitution dictionary.
- `depth` — depth (for debugging and tracing).
- `occurs_check` — occurs check (default `False`).

Returns a new substitution or `None`.

### 4.2.2. Dispatch

`Term.unify` is the public entry point:

1. If a custom unifier is installed (`set_custom_unifier`) and we are not inside it — `custom.unify(...)` is called.
2. Otherwise — `unify_strict(...)`.

This allows plugging in semantic unification without changing the term code.

### 4.2.3. Examples

```python
from prolog.terms import Term, Variable, Number

X = Variable('X')
t1 = Term('f', [X])
t2 = Term('f', [Number(42)])

subst = t1.unify(t2)
print(subst)   # {X: 42}
```

```python
# List
l1 = Term('.', [Number(1), Term('.', [Number(2), Term('[]')])])
l2 = Term('.', [X, Term('.', [Number(2), Term('[]')])])
print(l1.unify(l2))   # {X: 1}
```

### 4.2.4. `unify_strict`

Fully strict unification that never delegates to a custom unifier. Used when strict behavior must be explicitly guaranteed (e.g., inside `EmbeddingUnifier`).

## 4.3. Substitution

```python
term.substitute(subst)
```

Recursively applies a substitution.

```python
X = Variable('X')
t = Term('f', [X, Number(1)])
print(t.substitute({X: Number(42)}))   # f(42, 1)
```

## 4.4. Lists

A Prolog list is represented as:

- `Term('[]', [])` — empty list.
- `Term('.', [head, tail])` — cons cell.

```python
from prolog.terms import Term, Number

lst = Term('.', [Number(1), Term('.', [Number(2), Term('[]')])])
print(lst)                          # [1,2]
print(lst.is_list())                # True
print(lst.get_list_elements())      # [Number(1), Number(2)]
```

## 4.5. Custom Unifier

```python
from prolog.terms import set_custom_unifier, get_custom_unifier, clear_custom_unifier

class MyUnifier:
    def unify(self, t1, t2, subst, depth, occurs_check):
        # your logic
        return t1.unify_strict(t2, subst, depth, occurs_check)

set_custom_unifier(MyUnifier())
# ... usage ...
clear_custom_unifier()
```

---

# 5. Module parser — Prolog Syntax

## 5.1. Class `Parser`

```python
from prolog.parser import Parser

parser = Parser("parent(ivan, alexey).")
rules = parser.parse_rules()
```

> ⚠️ **Note:** a `Parser` instance can be used only once. For new text, create a new `Parser`.

## 5.2. Methods

### 5.2.1. `parse_rules(return_directives=False)`

Returns a list of `Rule`. If `return_directives=True`, returns a tuple `(rules, directives)`, where `directives` is a list of directive terms.

```python
rules, directives = Parser(text).parse_rules(return_directives=True)
```

### 5.2.2. `parse_query()`

Parses a query (without `?-`, but with a trailing period).

```python
query = Parser("parent(ivan, X).").parse_query()
```

### 5.2.3. `parse_query_from_string(text)` (module function)

```python
from prolog.parser import parse_query_from_string
q = parse_query_from_string("parent(ivan, X).")
```

## 5.3. Syntax

### 5.3.1. Atoms

- Start with a lowercase letter: `parent`, `ivan`, `москва`.
- Quoted: `'strange atom'`, `"string"`.
- Contain letters, digits, `_`.

### 5.3.2. Variables

- Start with an uppercase letter or `_`: `X`, `Y`, `_Foo`, `_`.
- `_` — anonymous variable (each occurrence is a new one).

### 5.3.3. Numbers

- Integers: `42`, `-7`.
- Floats: `3.14`, `-0.5`.
- Exponential: `1e-3`, `2.5E+10`.

### 5.3.4. Compound Terms

```prolog
f(X, Y).
parent(ivan, alexey).
tree(leaf(1), leaf(2)).
```

### 5.3.5. Lists

```prolog
[].
[a, b, c].
[H | T].
[a, b | T].
```

### 5.3.6. Comments

```prolog
% Single-line
/* Multi-line
   comment */
/* Nested /* comment */ still inside */
```

### 5.3.7. Operators

Full list of priorities:

| Operator | Priority | Assoc. |
|---|---:|---|
| `:-` | 1200 | xfx |
| `;` | 900 | xfy |
| `->` | 950 | xfy |
| `,` | 1000 | xfy |
| `=..` | 1100 | xfx |
| `is` | 1100 | xfx |
| `=`, `\=` | 1100 | xfx |
| `==`, `\==` | 1100 | xfx |
| `=:=`, `=\=` | 1100 | xfx |
| `<`, `>`, `=<`, `>=` | 1100 | xfx |
| `@<`, `@>`, `@=<`, `@>=` | 1100 | xfx |
| `+`, `-` | 1200 | yfx |
| `*`, `/` | 1300 | yfx |
| `mod`, `div` | 1300 | xfx |
| `**` | 1300 | xfy |
| `//`, `>>`, `<<` | 1200 | yfx |
| `^` | 1200 | xfy |
| `\+` | — | fy |

## 5.4. Directives

Only semantic unification directives are supported:

```prolog
:- semantic_on.
:- semantic_on(0.85).
:- semantic_off.
:- semantic_threshold(0.75).
```

Directives are returned by `parse_rules(return_directives=True)` and applied in `Solver._apply_directives`.

## 5.5. Error Handling

`Parser._error(...)` builds a message indicating line, column, and a caret:

```
Error: Expected '.' at end of query, but got 'foo'
   at line 3, column 15:
   parent(ivan, X) foo
                  ^
```

---

# 6. Module interpreter — Solver Core

## 6.1. Class `Frame`

Search stack frame. Attributes:

| Attribute | Purpose |
|---|---|
| `goals` | List of goals |
| `subst` | Substitution |
| `depth` | Depth |
| `rule_index` | Index of the current rule |
| `total_rules` | Total number of rules |
| `tail` | Tail flag |
| `trace_step` | Trace step |
| `cut_id` | Cut identifier |
| `disj_step` | Disjunction step |
| `ifthen_meta` | If-then-else metadata |

## 6.2. Class `BuiltinRegistry`

Registry of built-in predicates.

```python
registry = BuiltinRegistry()
registry.register(Predicate('my_pred', 1, my_func))
registry.register_custom('my_pred', 2, my_func2)
registry.register_module(my_module)
```

## 6.3. Class `Database`

Knowledge base + solver.

### 6.3.1. Creation

```python
from prolog.interpreter import Database

db = Database(rules=[], context={"llm": client})
```

### 6.3.2. Indexing

When `build_rule_index()` is called, an index is created:

```python
rule_index = {
    (functor, arity): {
        'all': [rule1, rule2, ...],
        'const': {
            const_value: [rule_with_const_first_arg, ...]
        }
    }
}
```

This allows quickly selecting rules by the first constant argument.

### 6.3.3. Working with the Database

```python
db.add_rule(rule)                       # append
db.add_rule(rule, at_beginning=True)    # prepend
db.retract_first(template_rule)         # remove first match
db.retract_all(template_rule)           # remove all matches
db.abolish(name, arity)                 # remove predicate
db.get_rules_for_predicate(name, arity)
db.has_predicate(name, arity)
```

### 6.3.4. Queries

```python
for solution in db.query(goal):
    print(solution)

solutions, tracer = db.query_with_explanation(goal)
```

## 6.4. Iterative Solver

The `_solve_iterative(initial_goal, initial_subst, depth, with_trace)` method implements:

- **TCO** (Tail Call Optimization): on a tail call, depth is not increased.
- **Cut**: via `CutException` and `cut_id`.
- **If-then-else**: via `_IfThenBarrier`.
- **Subst pruning** (Patch 5.8): `_prune_subst_to_goals` keeps only bindings reachable from new goals and anchor variables.

## 6.5. Cut

```prolog
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
```

Works via `CutException`, which is raised from `builtin_cut` and caught in `_solve_iterative`.

## 6.6. If-then-else

```prolog
classify(X, positive) :- X > 0 -> true ; classify_neg(X).
```

Implemented via `_IfThenBarrier`. Guarantees that after `Cond` succeeds, only `Then` is chosen and `Else` is discarded.

## 6.7. Repeat

```prolog
loop :- repeat, read(X), (X = end -> ! ; process(X), fail).
```

Implemented via `RepeatException`.

## 6.8. Substitution Pruning (Patch 5.8)

The `_prune_subst_to_goals(subst, goals, keep_vars)` method:

1. Seed = variables from `goals` + `keep_vars`.
2. Closure with chain resolution (`v -> w -> x -> 42` → `v -> 42`).
3. Return a new dictionary.

This keeps `subst` bounded in size regardless of recursion depth → O(N) instead of O(N²).

---

# 7. Module predicates — Built-in Predicates

## 7.1. Decorator `@prolog_predicate`

```python
from prolog.predicates import prolog_predicate

@prolog_predicate('my_pred', 2)
def my_pred(goal, database, subst, depth):
    # your logic
    yield subst.copy()
```

The function receives:
- `goal` — the goal term.
- `database` — the knowledge base (to access `context`).
- `subst` — the current substitution.
- `depth` — the depth.

Returns an iterator/list of substitutions.

## 7.2. Exceptions

| Exception | Purpose |
|---|---|
| `CutException(subst)` | Cut `!` |
| `RepeatException(subst)` | `repeat/0` |
| `PrologError(type, detail, term)` | ISO errors |

## 7.3. Arithmetic

### 7.3.1. `evaluate_expression(expr, subst)`

Evaluates an arithmetic expression. Uses the Numba path if available.

### 7.3.2. Supported Operators

- **Binary**: `+`, `-`, `*`, `/`, `mod`, `div`, `max`, `min`, `**`, `pow`, `rem`, `//`, `>>`, `<<`, `xor`.
- **Unary**: `-`, `abs`, `sign`, `round`, `floor`, `ceiling`, `truncate`, `float`, `integer`, `sin`, `cos`, `atan`, `exp`, `log`, `sqrt`.

### 7.3.3. Example

```prolog
X is 2 + 3 * 4.        % X = 14
Y is sqrt(16).         % Y = 4.0
Z is max(10, 20).      % Z = 20
```

## 7.4. Lists

### 7.4.1. `member/2`

Iterative traversal without `substitute` at each step.

```prolog
member(X, [a, b, c]).   % X = a ; X = b ; X = c
```

### 7.4.2. `append/3`

Relational semantics: works in all main modes.

```prolog
append([1, 2], [3, 4], L).       % L = [1,2,3,4]
append(A, B, [1, 2, 3]).         % A = [], B = [1,2,3] ; ...
```

### 7.4.3. `length/2`

Iterative counting O(len(List)).

```prolog
length([a, b, c], N).   % N = 3
length(L, 3).           % L = [_, _, _]
```

### 7.4.4. `memberchk/2`

Like `member/2`, but returns only the first solution.

## 7.5. Meta-Predicates

### 7.5.1. `findall/3`

```prolog
findall(X, parent(ivan, X), L).   % L = [alexey, maria]
```

Important: **does not swallow exceptions**. If a `PrologError` occurs inside `Goal`, it is propagated.

### 7.5.2. `bagof/3`

Like `findall/3`, but with grouping by free variables.

```prolog
bagof(X, parent(P, X), L).
```

### 7.5.3. `setof/3`

Like `bagof/3`, but sorts and removes duplicates.

### 7.5.4. `forall/2`

```prolog
forall(parent(ivan, X), write(X), nl).
```

### 7.5.5. `call/1` — `call/5`

```prolog
call(parent(ivan, X)).
call(my_pred, X, Y).
```

### 7.5.6. `once/1`

First solution.

### 7.5.7. `ignore/1`

Swallows errors and always succeeds.

## 7.6. Database

### 7.6.1. `asserta/1`, `assertz/1`

```prolog
asserta(fact(1)).
assertz(fact(2)).
```

### 7.6.2. `retract/1`, `retractall/1`

```prolog
retract(fact(X)).
retractall(fact(_)).
```

### 7.6.3. `abolish/1`

```prolog
abolish(fact/1).
```

### 7.6.4. `clause/2`

```prolog
clause(ancestor(X, Y), Body).
```

### 7.6.5. `current_predicate/1`

```prolog
current_predicate(fact/1).
```

## 7.7. Working with Terms

- `functor/3`, `arg/3`, `=../2`
- `copy_term/2`
- `term_variables/2`
- `numbervars/3`

```prolog
functor(parent(ivan, alexey), F, A).   % F = parent, A = 2
arg(1, parent(ivan, alexey), X).       % X = ivan
parent(ivan, alexey) =.. L.            % L = [parent, ivan, alexey]
```

## 7.8. Type Checking

- `var/1`, `nonvar/1`
- `atom/1`, `integer/1`, `float/1`, `number/1`, `atomic/1`
- `compound/1`, `ground/1`, `callable/1`

## 7.9. Comparison

- `==/2`, `\==/2`, `\=/2`
- `@</2`, `@>/2`, `@=</2`, `@>=/2`
- `compare/3`

```prolog
compare(<, a, b).      % true
compare(>, 2, 1).      % true
```

## 7.10. Strings

- `string_concat/3`
- `atomic_list_concat/2`, `atomic_list_concat/3`
- `atom_string/2`
- `term_string/2`
- `number_string/2`
- `split_string/4`
- `sub_string/5`

## 7.11. Semantic Unification

- `semantic_on/0`, `semantic_on/1`
- `semantic_off/0`
- `semantic_threshold/1`
- `semantic_status/1`

## 7.12. Numba Diagnostics

```python
from prolog.predicates import arith_numba_status, warmup_arith_numba
print(arith_numba_status())
warmup_arith_numba()
```

---

# 8. Module solver — High-Level Interface

## 8.1. Class `Solver`

```python
from prolog.solver import Solver

solver = Solver("""
parent(ivan, alexey).
parent(alexey, petr).
""", context={"key": "value"})
```

### 8.1.1. Attributes

- `solver.database` — the `Database` object.
- `solver.last_tracer` — the last tracer.

### 8.1.2. Methods

#### `find_solutions(query_text, with_explanation=False)`

```python
result = solver.find_solutions("parent(ivan, X).")
# [{'X': Term('alexey')}]

result, tracer = solver.find_solutions("parent(ivan, X).", with_explanation=True)
```

#### `get_explanation(format="text")`

```python
print(solver.get_explanation(format="text"))
print(solver.get_explanation(format="dict"))
print(solver.get_explanation(format="tree"))
```

## 8.2. `ParseSolution(query, solution, explanation=None)`

Formats the result for output.

```python
from prolog.solver import ParseSolution

parsed = ParseSolution("parent(ivan, X).", [{'X': Term('alexey')}])
print(parsed['output'])   # X = alexey
print(parsed['solution']) # [{'X': 'alexey'}]
```

Result keys:
- `query`
- `solution`
- `output`
- `count` (if multiple solutions)
- `explanation` (if provided)

## 8.3. Formatting Explanations

```python
from prolog.solver import format_explanation_for_human, format_dict_explanation

print(format_explanation_for_human(explanation))
```

---

# 9. Module bridge — Prolog ↔ Python Bridge

## 9.1. `prolog_to_python(term)`

```python
from prolog.bridge import prolog_to_python
from prolog.terms import Term, Number, Variable

prolog_to_python(Number(42))           # 42
prolog_to_python(Term('ivan'))          # 'ivan'
prolog_to_python(Variable('X'))         # None
prolog_to_python(Term('.', [Number(1), Term('[]')]))  # [1]
```

Correspondences:

| Prolog | Python |
|---|---|
| `Number` | `int` / `float` |
| `Variable` | `None` |
| atom | `str` |
| quoted string | `str` without quotes |
| list | `list` |
| compound term | `str` |

## 9.2. `python_to_prolog(obj)`

```python
from prolog.bridge import python_to_prolog

python_to_prolog(42)          # Number(42)
python_to_prolog("ivan")      # Term('"ivan"')
python_to_prolog([1, 2, 3])   # Prolog list
python_to_prolog(True)        # Term('true')
python_to_prolog(None)        # Term('[]')
python_to_prolog({'a': 1})    # list of '-'(K, V) pairs
```

## 9.3. `prolog_list_to_python(term)`

Returns a list of **terms** (not Python values).

```python
prolog_list_to_python(Term('.', [Number(1), Term('.', [Number(2), Term('[]')])]))
# [Number(1), Number(2)]
```

---

# 10. Module actions — Python Actions

## 10.1. General

The decorators `simple_action`, `effector_action`, `generator_action` turn Python functions into Prolog predicates.

All of them:
- attach metadata `_is_prolog_predicate`, `_predicate_name`, `_predicate_arity`;
- attach metadata `_agent_action`, `_action_name`, `_action_arity`, `_action_description`, `_action_type`;
- receive `context=database.context`.

## 10.2. `simple_action(name, arity, output_arg=-1, description=None)`

```python
from prolog.actions import simple_action

@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2
```

Rules:
- `output_arg` — index of the output argument (0-based).
- `output_arg=-1` → effector.
- Input arguments must be instantiated.
- Returns a single value.

## 10.3. `effector_action(name, arity, description=None)`

```python
from prolog.actions import effector_action

@effector_action('print_msg', 1)
def print_msg(msg, context):
    print("MSG:", msg)
```

- All arguments are input.
- Returns `None`.
- One solution on success.

## 10.4. `generator_action(name, arity, output_arg=-1, description=None)`

```python
from prolog.actions import generator_action

@generator_action('count', 2, output_arg=1)
def count(n, context):
    for i in range(1, n + 1):
        yield i
```

- The function returns an iterator.
- Each value → a separate solution.
- `output_arg=-1` → effector (the iterator is ignored).

## 10.5. Usage

```python
from prolog.agent import Agent

agent = Agent("""
run :- double(21, X), print_msg(X), count(3, Y), write(Y), nl, fail.
run.
""")

agent.register_action(double)
agent.register_action(print_msg)
agent.register_action(count)

agent.query("run.")
```

## 10.6. Error Handling

Any exception inside an action → the predicate simply yields no solutions (fail). This avoids crashing the entire query due to an error in one action.

---

# 11. Module agent — Agent Class

## 11.1. Creation

```python
from prolog.agent import Agent

agent = Agent(program="...", pl_file="...", context={"llm": client})
```

- `program` — program text.
- `pl_file` — file with the program.
- `context` — dictionary of external resources.

## 11.2. Registering Actions

```python
agent.register_action(func)
agent.register_action_module(module)
```

## 11.3. Listing Actions

```python
agent.list_actions()
# [{'name': 'double', 'arity': 2, 'description': ..., 'type': 'simple', ...}]
```

## 11.4. Queries

### 11.4.1. `query(query_text, with_explanation=False)`

```python
result = agent.query("parent(ivan, X).")
# {'solution': [{'X': 'alexey'}]}
```

### 11.4.2. `query_one(query_text)`

```python
result = agent.query_one("parent(ivan, X).")
# {'X': 'alexey'}
```

Distinguishes "failure" (`None`) from "success without variables" (`{'__success__': True}`).

### 11.4.3. `stream(query_text)`

```python
for solution in agent.stream("parent(ivan, X)."):
    print(solution)
```

## 11.5. Working with the Database

```python
agent.add_fact("parent(ivan, alexey).")
agent.add_facts("""
    parent(ivan, maria).
    parent(alexey, petr).
""")
agent.retract_fact("parent(ivan, alexey).")
agent.retract_all("parent(ivan, _).")
```

## 11.6. Saving and Loading

```python
agent.save_kb("kb.pl", format="prolog")
agent.save_kb("kb.pkl", format="binary")
agent.load_kb("kb.pl", format="prolog", clear=True)
agent.clear_kb()
```

## 11.7. Context

```python
agent.context['key'] = 'value'
print(agent.context)
```

## 11.8. Built-in Predicates

### 11.8.1. `set_state/1`

```prolog
set_state(mode(search)).
```

Replaces all facts with the given functor/arity.

### 11.8.2. `update_state/1`

```prolog
update_state(observed(bravo, search, 10.0, 5.0, 17)).
```

Adds a fact without removing old ones.

## 11.9. Legacy Decorator `agent_action`

```python
from prolog.agent import agent_action

@agent_action('my_action', 1, description="...", action_type="generic")
def my_action(goal, database, subst, depth):
    yield subst.copy()
```

Kept for backward compatibility. New code should use `simple_action`/`effector_action`/`generator_action`.

---

# 12. Module tracer — Proof Tracing

## 12.1. Class `ProofStep`

```python
@dataclass
class ProofStep:
    id: str
    goal: str
    rule_used: Optional[str]
    unification: List[Tuple[str, str]]
    success: bool
    children: List['ProofStep']
    subst_before: Dict[str, str]
    subst_after: Dict[str, str]
    depth: int
    builtin: bool
```

Methods:
- `to_dict()`
- `to_tree_string()`
- `__str__`

## 12.2. Class `ProofTracer`

```python
from prolog.tracer import ProofTracer

tracer = ProofTracer()
```

Methods:
- `start_goal(goal_str, depth)` — begin a step.
- `end_goal(success)` — end a step.
- `add_unification(var, value)`
- `set_rule(rule_str)`
- `set_builtin(builtin)`
- `set_substitutions(before, after)`
- `get_proof_tree()` — text tree.
- `get_proof_dict()` — dictionary.
- `clear()`

## 12.3. Example

```python
agent = Agent("""
parent(ivan, alexey).
parent(alexey, petr).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
""")

result, explanation = agent.query("ancestor(ivan, X).", with_explanation=True)
print(result)
print(explanation.get_proof_tree())
```

Output:

```
PROOF TREE:
==================================================
ancestor(ivan, X)
  parent(ivan, alexey)
    ...
```

---

# 13. Module embedding_unify — Semantic Unification

## 13.1. Concept

Semantic unification compares atoms not by exact equality but by meaning (embeddings). This allows unifying `москва` ↔ `moscow`, `Масква` ↔ `москва`, etc.

## 13.2. Matching Layers

1. **Layer 1**: text normalization (`strip`, unquoting, `lower()`).
2. **Layer 2a**: nearly identical strings (`lev_ratio <= lev_typo_max`) — match without the model.
3. **Layer 3**: hard — `cos >= threshold`.
4. **Layer 2b**: soft — `soft_cos_min <= cos < threshold` and `lev_ratio <= lev_max`.

## 13.3. Backends

### 13.3.1. Ollama

```python
from prolog.embedding_unify import semantic_on

semantic_on(
    threshold=0.82,
    backend='ollama',
    model='paraphrase-multilingual:latest',
    base_url='http://localhost:11434'
)
```

### 13.3.2. sentence-transformers

```python
semantic_on(
    threshold=0.82,
    backend='st',
    model='paraphrase-multilingual-minilm',
    st_device='cpu'
)
```

### 13.3.3. Custom

```python
def my_embed(text: str):
    return my_model.encode(text)

semantic_on(
    threshold=0.82,
    backend='custom',
    embed_fn=my_embed,
    text_prefix='query: '
)
```

## 13.4. Parameters

| Parameter | Default | Purpose |
|---|---|---|
| `threshold` | 0.82 | Hard cos threshold |
| `soft_cos_min` | 0.55 | Lower cos bound for soft |
| `lev_max` | 0.35 | Maximum lev_ratio for soft |
| `lev_typo_max` | 0.20 | Upper lev_ratio bound for typo |
| `text_prefix` | auto | Embedder prefix |
| `max_cache` | 10000 | Embedding cache size |

## 13.5. API

- `semantic_on(...)` — enable.
- `semantic_off()` — disable.
- `semantic_status()` — status.
- `semantic_set_threshold(value)` — change threshold.
- `semantic_set_corridor(...)` — change corridor.
- `semantic_set_prefix(prefix)` — change prefix.
- `semantic_get_unifier()` — get the unifier.
- `install_embedding_unifier(...)` — low-level install.
- `uninstall_embedding_unifier()` — uninstall.

## 13.6. Control from Prolog

```prolog
:- semantic_on(0.82).
semantic_status(S).
:- semantic_threshold(0.75).
:- semantic_off.
```

## 13.7. Example

```python
from prolog.agent import Agent
from prolog.embedding_unify import semantic_on, semantic_off

semantic_on(threshold=0.82, backend='st', model='paraphrase-multilingual-minilm')

agent = Agent("""
city(moscow).
city(paris).
""")

print(agent.query("city(moskva)."))
semantic_off()
```

## 13.8. Suppressing ST Output

By default, all service messages from `sentence-transformers` / `transformers` / HF Hub are disabled. To restore them:

```bash
export PROLOG_ST_VERBOSE=1
```

## 13.9. Known Prefixes

| Model | Prefix |
|---|---|
| `multilingual-e5` | `query: ` |
| `intfloat-e5` | `query: ` |
| `nomic-embed-text-v2` | `search_query: ` |
| `nomic-embed-text-v1` | `search_query: ` |

## 13.10. Model Aliases

```python
from prolog.embedding_unify import resolve_model_id

resolve_model_id('multilingual-e5-large')
# 'intfloat/multilingual-e5-large'
```

---

# 14. Module arith_numba — Arithmetic Acceleration

## 14.1. Principle

A Prolog term is compiled into flat numpy arrays ONCE (cached by `id(term)`). The JIT core is a non-recursive stack machine. Works on any Numba version and any ARM variant.

## 14.2. API

```python
from prolog.arith_numba import (
    try_eval_fast, evaluate_expression_numba,
    warmup_arith_numba, arith_numba_status, clear_arith_cache,
    NOT_ATTEMPTED
)
```

### `try_eval_fast(expr, subst)`

Returns:
- a number — successful computation via JIT;
- `NOT_ATTEMPTED` — Numba is not applicable;
- raises `PrologError` — semantic error.

### `warmup_arith_numba()`

Warms up the JIT core. Recommended to call once at startup.

### `arith_numba_status()`

Snapshot of the state.

### `clear_arith_cache()`

Clears the cache.

## 14.3. Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `PROLOG_NUMBA_THRESHOLD` | 20/30/40 | Node count threshold |
| `PROLOG_NUMBA_FASTMATH` | 0 | Enable fastmath |
| `PROLOG_NUMBA_CACHE` | 10000 | Cache size |
| `PROLOG_NUMBA_DISABLE` | 0 | Disable completely |

## 14.4. Example

```python
from prolog.predicates import arith_numba_status, warmup_arith_numba

print(arith_numba_status())
# {'available': True, 'active': True, 'numba_version': (0, 57, 0), ...}

warmup_arith_numba()
```

## 14.5. Platform Thresholds

| Platform | Threshold |
|---|---:|
| x86_64 | 20 |
| 64-bit ARM | 30 |
| 32-bit ARM | 40 |

---

# 15. Module prologchain — LangChain Integration

## 15.1. Class `PrologQueryTemplate`

```python
from prolog.prologchain import PrologQueryTemplate

template = PrologQueryTemplate.from_template("parent({person}, X).")
result = template.invoke({"person": "ivan"})
# {'query': 'parent(ivan, X).'}
```

Methods:
- `from_template(template)`
- `format(**kwargs)`
- `format_prompt(**kwargs)`
- `invoke(input)`
- `__or__(other)` — supports `|`.

## 15.2. Class `PrologChain`

```python
from prolog.prologchain import PrologChain

chain = PrologChain(
    program="parent(ivan, alexey).",
    context=None,
    include_explanation=False
)

result = chain.invoke({"query": "parent(ivan, X)."})
# {'success': True, 'query': ..., 'program': ..., 'output': ..., 'count': 1, ...}
```

## 15.3. Class `PrologOutputFormatter`

```python
from prolog.prologchain import PrologOutputFormatter

formatter = PrologOutputFormatter(include_query=True, include_stats=True)
print(formatter.invoke(result))
```

## 15.4. Ready-Made Chain

```python
from prolog.prologchain import PrologQueryTemplate, PrologChain

chain = PrologQueryTemplate.from_template("parent({person}, X).") | PrologChain(
    program="parent(ivan, alexey).",
    context=None
)

result = chain.invoke({"person": "ivan"})
print(result["output"])
```

## 15.5. `prolog_pipeline`

```python
from prolog.prologchain import prolog_pipeline

chain = prolog_pipeline(
    program="parent(ivan, alexey).",
    query_template="parent({person}, X)."
)

result = chain.invoke({"person": "ivan"})
```

## 15.6. Result Structure

| Key | Description |
|---|---|
| `success` | `True` / `False` |
| `query` | Query text |
| `program` | Program |
| `output` | Formatted output |
| `count` | Number of solutions |
| `solutions` | List of solutions |
| `result` | `True` / `False` (if no variables) |
| `explanation` | Explanation (if enabled) |
| `error` | Error text (if `success=False`) |

---

# 16. Module proagent_gradio — Gradio Sessions

## 16.1. Session Classes

### 16.1.1. `BaseGradioSession`

Base class with input/output queues.

### 16.1.2. `AgentGradioSession`

Session for the agent. Attributes:
- `history`
- `ui_values`

Methods:
- `start(agent, query)`
- `get_input()`
- `get_ui_value(name)`
- `set_ui_value(name, value)`
- `is_running()`
- `stop()`

## 16.2. Session Registry

```python
from prolog.proagent_gradio import (
    register_session, get_session, remove_session
)

register_session(session_id, session)
session = get_session(session_id)
remove_session(session_id)
```

## 16.3. Functions for the Agent

```python
from prolog.proagent_gradio import (
    create_agent_session, start_agent, send_input_to_agent,
    get_agent_outputs, is_session_running, stop_session
)

session_id = create_agent_session()
start_agent(session_id, agent, "run.")
send_input_to_agent(session_id, "hello")
outputs = get_agent_outputs(session_id)
stop_session(session_id)
```

## 16.4. Actions

| Action | Purpose |
|---|---|
| `gradio_print/1` | Output to UI |
| `gradio_read_input/1` | Read input |
| `gradio_get_ui_value/2` | Get a UI value |
| `gradio_set_ui_value/2` | Set a UI value |
| `format/3` | Format a string |

## 16.5. Registration

```python
from prolog.proagent_gradio import register_gradio_actions

register_gradio_actions(agent)
```

## 16.6. Prolog Example

```prolog
run :-
    gradio_print('Enter something:'),
    gradio_read_input(Input),
    format("You entered: ~w", [Input], Msg),
    gradio_print(Msg).
```

## 16.7. `format/3`

```prolog
format("Hello, ~w!", ["World"], Result).
% Result = "Hello, World!"
```

Supports `~w`, `~d`, `~f`, `~s`.

---

# 17. Prolog Subset Reference

## 17.1. Syntax

### 17.1.1. Terms

| Construct | Example |
|---|---|
| Atom | `foo`, `'bar baz'`, `"string"` |
| Variable | `X`, `_`, `_Foo` |
| Number | `42`, `3.14`, `1e-3` |
| Compound term | `f(a, b)` |
| List | `[a, b, c]`, `[H|T]` |
| Operator | `a + b`, `X is Y * 2` |

### 17.1.2. Rules

```prolog
head :- body1, body2, body3.
head.
```

### 17.1.3. Queries

```prolog
?- goal1, goal2.
```

`Agent.query` and `Solver.find_solutions` use the form without `?-`, but with a trailing period.

## 17.2. Control Constructs

| Construct | Syntax |
|---|---|
| Conjunction | `A, B` |
| Disjunction | `A ; B` |
| If-then | `Cond -> Then` |
| If-then-else | `(Cond -> Then ; Else)` |
| Cut | `!` |
| Negation | `\+ Goal` |
| True | `true` |
| Fail | `fail`, `false` |
| Repeat | `repeat` |

## 17.3. Built-in Predicates

### 17.3.1. Arithmetic

`is/2`, `>/2`, `</2`, `>=/2`, `=</2`, `=:=/2`, `=\=/2`

### 17.3.2. Lists

`member/2`, `append/3`, `length/2`, `memberchk/2`

### 17.3.3. Input/Output

`write/1`, `nl/0`

### 17.3.4. Database

`asserta/1`, `assertz/1`, `retract/1`, `retractall/1`, `abolish/1`, `clause/2`, `current_predicate/1`

### 17.3.5. Meta-Predicates

`call/1..5`, `once/1`, `ignore/1`, `forall/2`, `findall/3`, `bagof/3`, `setof/3`

### 17.3.6. Terms

`functor/3`, `arg/3`, `=../2`, `copy_term/2`, `term_variables/2`, `numbervars/3`

### 17.3.7. Types

`var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `atomic/1`, `compound/1`, `ground/1`, `callable/1`

### 17.3.8. Comparison

`==/2`, `\==/2`, `\=/2`, `@</2`, `@>/2`, `@=</2`, `@>=/2`, `compare/3`

### 17.3.9. Strings

`string_concat/3`, `atomic_list_concat/2`, `atomic_list_concat/3`, `atom_string/2`, `term_string/2`, `number_string/2`, `split_string/4`, `sub_string/5`

### 17.3.10. Semantic Unification

`semantic_on/0`, `semantic_on/1`, `semantic_off/0`, `semantic_threshold/1`, `semantic_status/1`

## 17.4. Not Implemented

- Module system (`:- module`, `:- use_module`).
- DCG (`-->`).
- User-defined operators (`:- op(...)`).
- `catch/3`, `throw/1`.
- `between/3`, `succ/2`, `plus/3`.
- `sort/2`, `msort/2`, `keysort/2`.
- `read/1`, `open/3`, `close/1`, `read_term/2`, `write_term/2`.
- `atom_length/2`, `atom_concat/3`, `atom_codes/2`, `char_code/2`, `atom_chars/2`.
- Tabling and constraint programming (`library(clpfd)`).
- `ground/1` works, but `ground/1` inside `bagof/3` is only partially accounted for.

## 17.5. Partially Implemented Predicates

| Predicate | Limitation |
|---|---|
| `length/2` | With an open list and unbound N, yields one minimal solution |
| `append/3` | Not complete in all exotic modes |
| `retract/1` | Mostly matches the rule head |
| `sub_string/5` | Not all ISO modes |
| `bagof/3` | Grouping by string key |
| `setof/3` | Sorting by string representation |

---

# 18. Practical Examples

## 18.1. Family Relations

```python
from prolog.agent import Agent

agent = Agent("""
parent(ivan, alexey).
parent(ivan, maria).
parent(alexey, petr).
parent(alexey, anna).
parent(maria, sergey).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
""")

print(agent.query("grandparent(ivan, X)."))
print(agent.query("ancestor(ivan, X)."))
```

## 18.2. Lists and findall

```python
agent = Agent("""
parent(ivan, alexey).
parent(ivan, maria).
""")

print(agent.query("findall(X, parent(ivan, X), L)."))
# L = ['alexey', 'maria']
```

## 18.3. Arithmetic

```python
agent = Agent("""
fact(0, 1).
fact(N, F) :- N > 0, N1 is N - 1, fact(N1, F1), F is N * F1.
""")

print(agent.query("fact(5, F)."))
# F = 120
```

## 18.4. Cut

```python
agent = Agent("""
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
""")

print(agent.query("max(3, 5, M)."))   # M = 5
print(agent.query("max(7, 2, M)."))   # M = 7
```

## 18.5. If-then-else

```python
agent = Agent("""
sign(X, positive) :- X > 0 -> true ; sign_neg(X).
sign_neg(X, negative) :- X < 0 -> true ; true.
sign_neg(_, zero).
""")

print(agent.query("sign(5, S)."))    # S = positive
print(agent.query("sign(-3, S)."))   # S = negative
print(agent.query("sign(0, S)."))    # S = zero
```

## 18.6. Agent Actions

```python
from prolog.agent import Agent, simple_action, effector_action, generator_action

@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2

@effector_action('print_msg', 1)
def print_msg(msg, context):
    print("MSG:", msg)

@generator_action('count', 2, output_arg=1)
def count(n, context):
    for i in range(1, n + 1):
        yield i

agent = Agent("""
run :- double(21, X), print_msg(X), count(3, Y), write(Y), nl, fail.
run.
""")

agent.register_action(double)
agent.register_action(print_msg)
agent.register_action(count)
agent.query("run.")
```

## 18.7. Semantic Unification

```python
from prolog.agent import Agent
from prolog.embedding_unify import semantic_on, semantic_off

semantic_on(
    threshold=0.82,
    backend='st',
    model='paraphrase-multilingual-minilm'
)

agent = Agent("""
city(moscow).
city(paris).
city(london).
""")

print(agent.query("city(moskva)."))
print(agent.query("city(paris)."))
semantic_off()
```

## 18.8. Tracing

```python
agent = Agent("""
parent(ivan, alexey).
parent(alexey, petr).
ancestor(X, Y) :- parent(X, Y).
ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
""")

result, explanation = agent.query("ancestor(ivan, X).", with_explanation=True)
print(result)
print(explanation.get_proof_tree())
```

## 18.9. LangChain

```python
from prolog.prologchain import PrologChain, PrologQueryTemplate

program = """
parent(ivan, alexey).
parent(ivan, maria).
"""

chain = PrologQueryTemplate.from_template("parent({person}, X).") | PrologChain(
    program=program,
    context=None
)

result = chain.invoke({"person": "ivan"})
print(result["output"])
```

## 18.10. Gradio

```python
from prolog.agent import Agent
from prolog.proagent_gradio import (
    create_agent_session, start_agent, get_agent_outputs,
    is_session_running, stop_session, register_gradio_actions
)

agent = Agent("""
run :-
    gradio_print('Starting work...'),
    gradio_read_input(Input),
    format("Received: ~w", [Input], Msg),
    gradio_print(Msg).
run.
""")

register_gradio_actions(agent)

session_id = create_agent_session()
start_agent(session_id, agent, "run.")

import time
time.sleep(0.5)
print(get_agent_outputs(session_id))

stop_session(session_id)
```

## 18.11. Saving/Loading the Database

```python
agent = Agent("fact(1). fact(2).")
agent.add_fact("fact(3).")
agent.save_kb("kb.pl", format="prolog")
agent.save_kb("kb.pkl", format="binary")

agent2 = Agent()
agent2.load_kb("kb.pl", format="prolog")
print(agent2.query("fact(X)."))
```

## 18.12. Numba Acceleration

```python
import os
os.environ['PROLOG_NUMBA_THRESHOLD'] = '10'

from prolog.predicates import warmup_arith_numba, arith_numba_status

warmup_arith_numba()
print(arith_numba_status())

from prolog.agent import Agent

agent = Agent("""
compute(X) :-
    X is 2 + 3 * 4 + 5 * 6 + 7 * 8 + 9 * 10 + 11 * 12 + 13 * 14.
""")

print(agent.query("compute(X)."))
```

---

# 19. Debugging and Diagnostics

## 19.1. Module Flags

| Module | Flag |
|---|---|
| `terms` | `DEBUG_UNIFY = True` |
| `interpreter` | `DEBUG = True` |
| `agent` | `DEBUG = True` |

```python
import prolog.terms as terms
terms.DEBUG_UNIFY = True

import prolog.interpreter as interp
interp.DEBUG = True
```

## 19.2. Numba Diagnostics

```python
from prolog.predicates import arith_numba_status
print(arith_numba_status())
```

## 19.3. Semantic Unification Diagnostics

```python
from prolog.embedding_unify import semantic_status
print(semantic_status())
```

## 19.4. Action Diagnostics

```python
agent.list_actions()
```

## 19.5. Proof Tracing

```python
result, explanation = agent.query("goal.", with_explanation=True)
print(explanation.get_proof_tree())
```

## 19.6. Exception Handling

- `PrologError` — semantic error.
- `CutException` / `RepeatException` — control flow.
- Other `Exception`s in meta-predicates are printed with traceback and propagated.

---

# 20. Best Practices

## 20.1. Prolog Programming

1. **Use cut carefully** — it can hide alternative solutions.
2. **Prefer if-then-else** for deterministic branching.
3. **Check goal order** — Prolog searches left to right.
4. **Index rules** — put a constant first argument.
5. **Don't overuse `assert`/`retract`** — they change the index.
6. **Use `findall` instead of manual collection**.
7. **Check `ground/1`** before arithmetic.

## 20.2. Python Actions

1. **Use `simple_action`/`effector_action`/`generator_action`**, not legacy `agent_action`.
2. **Instantiate input arguments** — otherwise the action won't fire.
3. **Return `None`** for failure (effector).
4. **Catch exceptions inside the action** if you want a silent fail.
5. **Use `context`** to access external resources.

## 20.3. Performance

1. **Use TCO** — write recursion in tail form.
2. **Avoid deep non-tail recursion** — use accumulators.
3. **Index the first argument** — constants in the head speed up search.
4. **Enable Numba** for large arithmetic expressions.
5. **Cache embeddings** — the cache size is configurable.
6. **Use `stream`** for large result sets.

## 20.4. Semantic Unification

1. **Tune the threshold** empirically (usually 0.75–0.85).
2. **Use prefixes** for E5/nomic models.
3. **Cache embeddings** — `max_cache` matters.
4. **Disable semantics** where strictness is required.
5. **Check `semantic_status()`** for diagnostics.

## 20.5. Security

1. **Do not execute untrusted queries** — Prolog can enter an infinite search.
2. **Limit depth** via `depth` in custom predicates.
3. **Isolate the context** — don't give actions access to everything.
4. **Validate input** before parsing.

---

# 21. Limitations and Known Features

## 21.1. Language Limitations

- No modules.
- No DCG.
- No user-defined operators.
- No `catch/3`/`throw/1`.
- No `between/3`, `succ/2`.
- No `sort/2`, `msort/2`.
- No `read/1`, `open/3`.
- No `atom_length/2`, `atom_concat/3`, `atom_codes/2`.
- No CLP(FD).

## 21.2. Implementation Features

- **Occurs check is disabled by default** — for performance.
- **Anonymous `_`** always unifies.
- **Variables are compared by name**, not by object.
- **Subst pruning** (Patch 5.8) may affect variables not reachable from the goals.
- **`if-then-else`** is implemented via `_IfThenBarrier`, not via cut.
- **Meta-predicates do not swallow exceptions** (since version 5.3).

## 21.3. Compatibility

- Python 3.8+.
- Numba — optional.
- sentence-transformers — optional.
- LangChain — optional.
- Gradio — optional.

## 21.4. Known Bugs and Workarounds

| Problem | Workaround |
|---|---|
| `length/2` with an open list | Use `length(L, N)` with a concrete `N` |
| `append/3` in non-standard modes | Use `member/2` or `findall` |
| Semantic unification is slow | Reduce `max_cache`, use caching |
| Numba crashes on ARM | Set `PROLOG_NUMBA_DISABLE=1` |
| TCO doesn't trigger | Write recursion in tail form |

---

# Conclusion

PyAgentProlog is a full-featured Prolog engine in Python that combines classical logical inference with modern capabilities:

- **Agents with Python actions** — integration with any Python libraries.
- **Semantic unification** — working with fuzzy concepts.
- **Proof tracing** — explainability of inference.
- **LangChain and Gradio** — integration with LLMs and UI.
- **Numba** — arithmetic acceleration on weak hardware.

The package is suitable both for learning Prolog and for industrial neuro-symbolic applications.