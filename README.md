# PyAgentProlog

A Prolog engine in Python with proof tracing, agents based on Python actions, an expert system, and semantic unification on embeddings.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author

Andrey Vorfolomeev

## Features

- **Prolog core**: parser, iterative solver with TCO (tail-call optimization),
  cut (`!`), `if-then-else`, rule indexing by the first argument.
- **Proof tracing** — an inference tree with unifications and rules.
- **Numba-accelerated arithmetic** — a non-recursive stack machine, runs on ARM/Raspberry Pi.
- **Agents** — Python functions as built-in predicates (`@simple_action`,
  `@effector_action`, `@generator_action`).
- **Expert system** — interactive mode, extraction of conditions from natural language
  via `pymorphy3`, contradiction resolution, glossary.
- **Semantic unification** — matching atoms via embeddings
 (Ollama or sentence-transformers).
- **Integration with LangChain** — `PrologChain` as `Runnable` in LCEL chains.
- **Gradio sessions** — for interactive UIs.

## Installation

### Basic (core only)

```bash
pip install pyagentprolog
```

### With additional features

```bash
# Numba acceleration for arithmetic
pip install pyagentprolog[numba]

# Semantic unification
pip install pyagentprolog[embeddings]

# LangChain
pip install pyagentprolog[langchain]

# Expert system (pymorphy2)
pip install pyagentprolog[expert]

# Everything at once
pip install pyagentprolog[all]
```

## Quick start

### Simple query

```python
from prolog import Solver

program = """
parent(ivan, alexey).
parent(alexey, petr).

grandfather(D, V) :-
    parent(D, R),
    parent(R, V).
"""

solver = Solver(program)
solutions = solver.find_solutions("grandfather(ivan, X).")
print(solutions)  # [{'X': 'petr'}]
```

### Proof trace

```python
solutions, explanation = solver.find_solutions("grandfather(ivan, X).",
                                                with_explanation=True)
print(solver.get_explanation())
```

### Agent with Python actions

```python
from prolog import Agent, simple_action

@simple_action('add', 3, output_arg=2)
def add(a, b, context=None):
    return a + b

agent = Agent(program="run(X) :- add(2, 3, X).")
agent.register_action(add)
print(agent.query_one("run(X)."))  # {'X': 5}

### Expert System

Requires pyagentprolog[expert]

```python
from prolog import Expert

expert = Expert(domain_file="knowledge.pl", use_pymorphy=True)
answer = expert.ask("I want to learn programming")
print(answer)
```

## Requirements

- Python 3.9+
- Mandatory: `numpy`
- Optional: `numba`, `langchain-core`, `sentence-transformers`,
  `langchain-ollama`, `gradio`, `pymorphy2`

## License

MIT — see the [LICENSE](LICENSE) file.