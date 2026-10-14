# PyAgentProlog

Prolog-движок на Python с трассировкой доказательств, агентами на Python-действиях,
экспертной системой и семантической унификацией на эмбеддингах.

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Автор

Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

## Возможности

- **Ядро Prolog**: парсер, итеративный решатель с TCO (tail-call optimization),
  отсечение (`!`), `if-then-else`, индексация правил по первому аргументу.
- **Трассировка доказательств** — дерево вывода с унификациями и правилами.
- **Numba-ускорение арифметики** — нерекурсивная стек-машина, работает на ARM/Raspberry Pi.
- **Агенты** — Python-функции как встроенные предикаты (`@simple_action`,
  `@effector_action`, `@generator_action`).
- **Экспертная система** — диалоговый режим, извлечение условий из естественного языка
  через `pymorphy3`, разрешение противоречий, глоссарий.
- **Семантическая унификация** — сопоставление атомов через эмбеддинги
  (Ollama или sentence-transformers).
- **Интеграция с LangChain** — `PrologChain` как `Runnable` в LCEL-цепочках.
- **Gradio-сессии** — для интерактивных UI.

## Установка

### Базовая (только ядро)

```bash
pip install pyagentprolog
```

### С дополнительными возможностями

```bash
# Numba-ускорение арифметики
pip install pyagentprolog[numba]

# Семантическая унификация
pip install pyagentprolog[embeddings]

# LangChain
pip install pyagentprolog[langchain]

# Экспертная система (pymorphy3)
pip install pyagentprolog[expert]

# Всё сразу
pip install pyagentprolog[all]
```

## Быстрый старт

### Простой запрос

```python
from prolog import Solver

program = """
родитель(иван, алексей).
родитель(алексей, петр).

дедушка(Д, В) :-
    родитель(Д, Р),
    родитель(Р, В).
"""

solver = Solver(program)
solutions = solver.find_solutions("дедушка(иван, X).")
print(solutions)  # [{'X': 'петр'}]
```

### Трассировка доказательства

```python
solutions, explanation = solver.find_solutions("дедушка(иван, X).",
                                                with_explanation=True)
print(solver.get_explanation())
```

### Агент с Python-действиями

```python
from prolog import Agent, simple_action

@simple_action('add', 3, output_arg=2)
def add(a, b, context=None):
    return a + b

agent = Agent(program="run(X) :- add(2, 3, X).")
agent.register_action(add)
print(agent.query_one("run(X)."))  # {'X': 5}
```

### Экспертная система

Требует pyagentprolog[expert]

```python
from prolog import Expert

expert = Expert(domain_file="knowledge.pl", use_pymorphy=True)
answer = expert.ask("я хочу учиться программированию")
print(answer)
```

## Требования

- Python 3.9+
- Обязательно: `numpy`
- Опционально: `numba`, `langchain-core`, `sentence-transformers`,
  `langchain-ollama`, `gradio`, `pymorphy3`

## Лицензия

MIT — см. файл [LICENSE](LICENSE).

