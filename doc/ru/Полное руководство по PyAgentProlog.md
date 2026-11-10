
**Назначение:** Prolog-движок на Python с агентами, трассировкой и семантической унификацией

---

# Содержание

1. [Введение](#1-введение)
2. [Установка и быстрый старт](#2-установка-и-быстрый-старт)
3. [Архитектура пакета](#3-архитектура-пакета)
4. [Модуль terms — термы и унификация](#4-модуль-terms--термы-и-унификация)
5. [Модуль parser — синтаксис Prolog](#5-модуль-parser--синтаксис-prolog)
6. [Модуль interpreter — ядро решателя](#6-модуль-interpreter--ядро-решателя)
7. [Модуль predicates — встроенные предикаты](#7-модуль-predicates--встроенные-предикаты)
8. [Модуль solver — высокоуровневый интерфейс](#8-модуль-solver--высокоуровневый-интерфейс)
9. [Модуль bridge — мост Prolog ↔ Python](#9-модуль-bridge--мост-prolog--python)
10. [Модуль actions — Python-действия](#10-модуль-actions--python-действия)
11. [Модуль agent — класс Agent](#11-модуль-agent--класс-agent)
12. [Модуль tracer — трассировка доказательств](#12-модуль-tracer--трассировка-доказательств)
13. [Модуль embedding_unify — семантическая унификация](#13-модуль-embedding_unify--семантическая-унификация)
14. [Модуль arith_numba — ускорение арифметики](#14-модуль-arith_numba--ускорение-арифметики)
15. [Модуль prologchain — интеграция с LangChain](#15-модуль-prologchain--интеграция-с-langchain)
16. [Модуль proagent_gradio — Gradio-сессии](#16-модуль-proagent_gradio--gradio-сессии)
17. [Справочник по подмножеству Prolog](#17-справочник-по-подмножеству-prolog)
18. [Практические примеры](#18-практические-примеры)
19. [Отладка и диагностика](#19-отладка-и-диагностика)
20. [Лучшие практики](#20-лучшие-практики)
21. [Ограничения и известные особенности](#21-ограничения-и-известные-особенности)

---

# 1. Введение

## 1.1. Что такое PyAgentProlog

**PyAgentProlog** — это самостоятельный, независимый от внешних логических систем Prolog-движок, реализованный на чистом Python. Пакет не требует установки SWI-Prolog, GNU Prolog или других внешних решателей — всё ядро работает на Python.

Ключевые возможности:

- **Логический вывод**: полный цикл унификации, поиска с возвратом, отсечения.
- **Итеративный решатель с TCO**: оптимизация хвостовых вызовов, устранение переполнения стека.
- **Встроенные и мета-предикаты**: арифметика, списки, findall/setof/bagof, call/N, forall.
- **Трассировка доказательств**: построение дерева вывода с унификациями и правилами.
- **Агенты с Python-действиями**: подключение обычных Python-функций как предикатов Prolog.
- **Семантическая унификация**: сравнение атомов через эмбеддинги (Ollama, sentence-transformers, кастомные).
- **Интеграция с LangChain**: цепочки Runnable с Prolog-запросами.
- **Gradio-сессии**: интерактивные UI-сессии для агента.
- **Numba-ускорение арифметики**: JIT-компиляция больших арифметических выражений.
- **Опциональные зависимости**: ядро работает без numpy, langchain, gradio, sentence-transformers.

## 1.2. Для чего использовать

- Экспертные системы с логическим выводом.
- Нейро-символические агенты (LLM + логика).
- Обучение Prolog на Python.
- Быстрое прототипирование правил и онтологий.
- Обработка естественного языка с семантической унификацией.
- Интеграция логического вывода в LangChain-пайплайны.

## 1.3. Лицензия и авторство

Пакет распространяется с сохранением авторства Andrey Vorfolomeev. Версия 1.0.1.

---

# 2. Установка и быстрый старт

## 2.1. Требования

- Python 3.8+
- Обязательных внешних зависимостей нет для ядра.

Опциональные зависимости:

| Функция | Пакет |
|---|---|
| Numba-ускорение арифметики | `numpy`, `numba` |
| Семантическая унификация через Ollama | `langchain-ollama` или `langchain-community` |
| Семантическая унификация через ST | `sentence-transformers` |
| LangChain-интеграция | `langchain-core`, `pydantic` |
| Gradio-сессии | `gradio` (для UI) |

## 2.2. Установка

```bash
# Ядро (без зависимостей)
pip install pyagentprolog

# С Numba
pip install pyagentprolog[numba]

# С семантикой (sentence-transformers)
pip install pyagentprolog[semantic]

# С LangChain
pip install pyagentprolog[langchain]

# Всё сразу
pip install pyagentprolog[all]
```

Если пакет распространяется как локальный модуль, положите директорию `prolog/` в `PYTHONPATH`.

## 2.3. Быстрый старт

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

## 2.4. Минимальная проверка работоспособности

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

# 3. Архитектура пакета

## 3.1. Слои

```
┌─────────────────────────────────────────────────────────┐
│  Прикладной слой                                        │
│  proagent_gradio, prologchain, agent                    │
├─────────────────────────────────────────────────────────┤
│  Слой действий                                          │
│  actions (simple/effector/generator)                    │
├─────────────────────────────────────────────────────────┤
│  Слой унификации                                        │
│  terms.unify → embedding_unify (опционально)            │
├─────────────────────────────────────────────────────────┤
│  Слой вывода                                            │
│  solver → interpreter (Database, Frame, TCO, cut)       │
├─────────────────────────────────────────────────────────┤
│  Слой предикатов                                        │
│  predicates, arith_numba                                │
├─────────────────────────────────────────────────────────┤
│  Слой парсинга                                          │
│  parser → terms                                         │
└─────────────────────────────────────────────────────────┘
```

## 3.2. Порядок импорта

В `__init__.py` соблюдён строгий порядок:

```
terms → bridge → parser → tracer → predicates
      → interpreter → solver → actions → agent
```

Опциональные модули (`embedding_unify`, `prologchain`, `proagent_gradio`) импортируются в `try/except ImportError`.

## 3.3. Ключевые сущности

| Сущность | Модуль | Назначение |
|---|---|---|
| `Term` | `terms` | Составной терм Prolog |
| `Variable` | `terms` | Переменная Prolog |
| `Number` | `terms` | Число |
| `Rule` | `terms` | Правило (голова + тело) |
| `Parser` | `parser` | Парсер текста Prolog |
| `Database` | `interpreter` | База знаний + решатель |
| `BuiltinRegistry` | `interpreter` | Реестр встроенных предикатов |
| `Frame` | `interpreter` | Кадр стека поиска |
| `Solver` | `solver` | Высокоуровневый решатель |
| `Agent` | `agent` | Агент с действиями |
| `ProofTracer` | `tracer` | Трассировщик |
| `EmbeddingUnifier` | `embedding_unify` | Семантический унификатор |

---

# 4. Модуль terms — термы и унификация

## 4.1. Классы термов

### 4.1.1. `Term`

Базовый класс для составных термов.

```python
from prolog.terms import Term

t = Term('parent', [Term('ivan'), Term('alexey')])
print(t)           # parent(ivan, alexey)
print(t.functor)   # parent
print(t.arguments) # [Term(ivan), Term(alexey)]
```

- `Term(functor, arguments=None)` — конструктор.
- `functor` — строка (имя функтора) или другое значение.
- `arguments` — список подтермов.

### 4.1.2. `Variable`

Переменная Prolog. Наследуется от `Term`.

```python
from prolog.terms import Variable

x = Variable('X')
print(x.name)  # X
```

- `Variable(name)` — конструктор.
- Переменная с именем `_` — анонимная.
- `__eq__` и `__hash__` основаны на имени.

### 4.1.3. `Number`

Число Prolog.

```python
from prolog.terms import Number

n = Number(42)
f = Number(3.14)
print(n.value)   # 42
print(f.value)   # 3.14
```

- `Number(value)` — конструктор.
- Поддерживаются `int` и `float`.
- Сравнивается с числами через `__eq__`, `__lt__` и т.д.

### 4.1.4. `Rule`

Правило Prolog.

```python
from prolog.terms import Term, Rule

head = Term('ancestor', [Variable('X'), Variable('Y')])
body = Term('parent', [Variable('X'), Variable('Y')])
rule = Rule(head, body)
print(rule)  # ancestor(X, Y) :- parent(X, Y).
```

- `Rule(head, body)` — конструктор.
- Факт: `body = Term('true', [])`.
- `__str__` возвращает корректную запись.

## 4.2. Унификация

### 4.2.1. Публичный API

```python
Term.unify(other, subst=None, depth=0, occurs_check=False)
```

- `other` — второй терм.
- `subst` — словарь подстановки.
- `depth` — глубина (для отладки и трассировки).
- `occurs_check` — проверка вхождения (по умолчанию `False`).

Возвращает новую подстановку или `None`.

### 4.2.2. Диспетчеризация

`Term.unify` — публичная точка входа:

1. Если установлен кастомный унификатор (`set_custom_unifier`) и мы не внутри него — вызывается `custom.unify(...)`.
2. Иначе — `unify_strict(...)`.

Это позволяет подключать семантическую унификацию без изменения кода термов.

### 4.2.3. Примеры

```python
from prolog.terms import Term, Variable, Number

X = Variable('X')
t1 = Term('f', [X])
t2 = Term('f', [Number(42)])

subst = t1.unify(t2)
print(subst)   # {X: 42}
```

```python
# Список
l1 = Term('.', [Number(1), Term('.', [Number(2), Term('[]')])])
l2 = Term('.', [X, Term('.', [Number(2), Term('[]')])])
print(l1.unify(l2))   # {X: 1}
```

### 4.2.4. `unify_strict`

Полностью строгая унификация, никогда не уходит в кастомный унификатор. Используется, когда нужно явно гарантировать строгое поведение (например, внутри `EmbeddingUnifier`).

## 4.3. Подстановка

```python
term.substitute(subst)
```

Рекурсивно применяет подстановку.

```python
X = Variable('X')
t = Term('f', [X, Number(1)])
print(t.substitute({X: Number(42)}))   # f(42, 1)
```

## 4.4. Списки

Prolog-список представляется как:

- `Term('[]', [])` — пустой список.
- `Term('.', [head, tail])` — cons-ячейка.

```python
from prolog.terms import Term, Number

lst = Term('.', [Number(1), Term('.', [Number(2), Term('[]')])])
print(lst)                          # [1,2]
print(lst.is_list())                # True
print(lst.get_list_elements())      # [Number(1), Number(2)]
```

## 4.5. Кастомный унификатор

```python
from prolog.terms import set_custom_unifier, get_custom_unifier, clear_custom_unifier

class MyUnifier:
    def unify(self, t1, t2, subst, depth, occurs_check):
        # ваша логика
        return t1.unify_strict(t2, subst, depth, occurs_check)

set_custom_unifier(MyUnifier())
# ... использование ...
clear_custom_unifier()
```

---

# 5. Модуль parser — синтаксис Prolog

## 5.1. Класс `Parser`

```python
from prolog.parser import Parser

parser = Parser("parent(ivan, alexey).")
rules = parser.parse_rules()
```

> ⚠️ **Внимание:** экземпляр `Parser` можно использовать только один раз. Для нового текста создавайте новый `Parser`.

## 5.2. Методы

### 5.2.1. `parse_rules(return_directives=False)`

Возвращает список `Rule`. Если `return_directives=True`, возвращает кортеж `(rules, directives)`, где `directives` — список термов-директив.

```python
rules, directives = Parser(text).parse_rules(return_directives=True)
```

### 5.2.2. `parse_query()`

Разбирает запрос (без `?-`, но с точкой в конце).

```python
query = Parser("parent(ivan, X).").parse_query()
```

### 5.2.3. `parse_query_from_string(text)` (функция модуля)

```python
from prolog.parser import parse_query_from_string
q = parse_query_from_string("parent(ivan, X).")
```

## 5.3. Синтаксис

### 5.3.1. Атомы

- Начинаются со строчной буквы: `parent`, `ivan`, `москва`.
- В кавычках: `'странный атом'`, `"строка"`.
- Содержат буквы, цифры, `_`.

### 5.3.2. Переменные

- Начинаются с заглавной буквы или `_`: `X`, `Y`, `_Foo`, `_`.
- `_` — анонимная переменная (каждое вхождение — новая).

### 5.3.3. Числа

- Целые: `42`, `-7`.
- Вещественные: `3.14`, `-0.5`.
- Экспоненциальные: `1e-3`, `2.5E+10`.

### 5.3.4. Составные термы

```prolog
f(X, Y).
parent(ivan, alexey).
tree(leaf(1), leaf(2)).
```

### 5.3.5. Списки

```prolog
[].
[a, b, c].
[H | T].
[a, b | T].
```

### 5.3.6. Комментарии

```prolog
% Однострочный
/* Многострочный
   комментарий */
/* Вложенный /* комментарий */ всё ещё внутри */
```

### 5.3.7. Операторы

Полный список приоритетов:

| Оператор | Приоритет | Ассоц. |
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

## 5.4. Директивы

Поддерживаются только директивы семантической унификации:

```prolog
:- semantic_on.
:- semantic_on(0.85).
:- semantic_off.
:- semantic_threshold(0.75).
```

Директивы возвращаются `parse_rules(return_directives=True)` и применяются в `Solver._apply_directives`.

## 5.5. Обработка ошибок

`Parser._error(...)` формирует сообщение с указанием строки, колонки и каретки:

```
Ошибка: Ожидалась '.' в конце запроса, но получено 'foo'
   в строке 3, колонка 15:
   parent(ivan, X) foo
                  ^
```

---

# 6. Модуль interpreter — ядро решателя

## 6.1. Класс `Frame`

Кадр стека поиска. Атрибуты:

| Атрибут | Назначение |
|---|---|
| `goals` | Список целей |
| `subst` | Подстановка |
| `depth` | Глубина |
| `rule_index` | Индекс текущего правила |
| `total_rules` | Всего правил |
| `tail` | Хвостовой флаг |
| `trace_step` | Шаг трассировки |
| `cut_id` | Идентификатор отсечения |
| `disj_step` | Шаг дизъюнкции |
| `ifthen_meta` | Метаданные if-then-else |

## 6.2. Класс `BuiltinRegistry`

Реестр встроенных предикатов.

```python
registry = BuiltinRegistry()
registry.register(Predicate('my_pred', 1, my_func))
registry.register_custom('my_pred', 2, my_func2)
registry.register_module(my_module)
```

## 6.3. Класс `Database`

База знаний + решатель.

### 6.3.1. Создание

```python
from prolog.interpreter import Database

db = Database(rules=[], context={"llm": client})
```

### 6.3.2. Индексация

При `build_rule_index()` создаётся индекс:

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

Это позволяет быстро отбирать правила по первому константному аргументу.

### 6.3.3. Работа с базой

```python
db.add_rule(rule)                       # в конец
db.add_rule(rule, at_beginning=True)    # в начало
db.retract_first(template_rule)         # удалить первое совпадение
db.retract_all(template_rule)           # удалить все совпадения
db.abolish(name, arity)                 # удалить предикат
db.get_rules_for_predicate(name, arity)
db.has_predicate(name, arity)
```

### 6.3.4. Запросы

```python
for solution in db.query(goal):
    print(solution)

solutions, tracer = db.query_with_explanation(goal)
```

## 6.4. Итеративный решатель

Метод `_solve_iterative(initial_goal, initial_subst, depth, with_trace)` реализует:

- **TCO** (Tail Call Optimization): при хвостовом вызове глубина не увеличивается.
- **Cut**: через `CutException` и `cut_id`.
- **If-then-else**: через `_IfThenBarrier`.
- **Обрезку subst** (Патч 5.8): `_prune_subst_to_goals` оставляет только связки, достижимые из новых целей и якорных переменных.

## 6.5. Отсечение

```prolog
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
```

Работает через `CutException`, которая поднимается из `builtin_cut` и перехватывается в `_solve_iterative`.

## 6.6. If-then-else

```prolog
classify(X, positive) :- X > 0 -> true ; classify_neg(X).
```

Реализуется через `_IfThenBarrier`. Гарантирует, что после успеха `Cond` выбирается только `Then`, а `Else` отбрасывается.

## 6.7. Repeat

```prolog
loop :- repeat, read(X), (X = end -> ! ; process(X), fail).
```

Реализуется через `RepeatException`.

## 6.8. Обрезка подстановки (Патч 5.8)

Метод `_prune_subst_to_goals(subst, goals, keep_vars)`:

1. Seed = переменные из `goals` + `keep_vars`.
2. Замыкание с разрешением цепочек (`v -> w -> x -> 42` → `v -> 42`).
3. Возврат нового словаря.

Это делает `subst` ограниченным по размеру независимо от глубины рекурсии → O(N) вместо O(N²).

---

# 7. Модуль predicates — встроенные предикаты

## 7.1. Декоратор `@prolog_predicate`

```python
from prolog.predicates import prolog_predicate

@prolog_predicate('my_pred', 2)
def my_pred(goal, database, subst, depth):
    # ваша логика
    yield subst.copy()
```

Функция получает:
- `goal` — терм цели.
- `database` — база знаний (для доступа к `context`).
- `subst` — текущая подстановка.
- `depth` — глубина.

Возвращает итератор/список подстановок.

## 7.2. Исключения

| Исключение | Назначение |
|---|---|
| `CutException(subst)` | Отсечение `!` |
| `RepeatException(subst)` | `repeat/0` |
| `PrologError(type, detail, term)` | ISO-ошибки |

## 7.3. Арифметика

### 7.3.1. `evaluate_expression(expr, subst)`

Вычисляет арифметическое выражение. Использует Numba-путь, если доступен.

### 7.3.2. Поддерживаемые операторы

- **Бинарные**: `+`, `-`, `*`, `/`, `mod`, `div`, `max`, `min`, `**`, `pow`, `rem`, `//`, `>>`, `<<`, `xor`.
- **Унарные**: `-`, `abs`, `sign`, `round`, `floor`, `ceiling`, `truncate`, `float`, `integer`, `sin`, `cos`, `atan`, `exp`, `log`, `sqrt`.

### 7.3.3. Пример

```prolog
X is 2 + 3 * 4.        % X = 14
Y is sqrt(16).         % Y = 4.0
Z is max(10, 20).      % Z = 20
```

## 7.4. Списки

### 7.4.1. `member/2`

Итеративный обход без `substitute` на каждом шаге.

```prolog
member(X, [a, b, c]).   % X = a ; X = b ; X = c
```

### 7.4.2. `append/3`

Реляционная семантика: работает во всех основных режимах.

```prolog
append([1, 2], [3, 4], L).       % L = [1,2,3,4]
append(A, B, [1, 2, 3]).         % A = [], B = [1,2,3] ; ...
```

### 7.4.3. `length/2`

Итеративный подсчёт O(len(List)).

```prolog
length([a, b, c], N).   % N = 3
length(L, 3).           % L = [_, _, _]
```

### 7.4.4. `memberchk/2`

Как `member/2`, но возвращает только первое решение.

## 7.5. Мета-предикаты

### 7.5.1. `findall/3`

```prolog
findall(X, parent(ivan, X), L).   % L = [alexey, maria]
```

Важно: **не глотает исключения**. Если внутри `Goal` возникнет `PrologError` — пробрасывается.

### 7.5.2. `bagof/3`

Как `findall/3`, но с группировкой по свободным переменным.

```prolog
bagof(X, parent(P, X), L).
```

### 7.5.3. `setof/3`

Как `bagof/3`, но сортирует и удаляет дубликаты.

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

Первое решение.

### 7.5.7. `ignore/1`

Проглатывает ошибки и всегда успешен.

## 7.6. База данных

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

## 7.7. Работа с термами

- `functor/3`, `arg/3`, `=../2`
- `copy_term/2`
- `term_variables/2`
- `numbervars/3`

```prolog
functor(parent(ivan, alexey), F, A).   % F = parent, A = 2
arg(1, parent(ivan, alexey), X).       % X = ivan
parent(ivan, alexey) =.. L.            % L = [parent, ivan, alexey]
```

## 7.8. Проверка типов

- `var/1`, `nonvar/1`
- `atom/1`, `integer/1`, `float/1`, `number/1`, `atomic/1`
- `compound/1`, `ground/1`, `callable/1`

## 7.9. Сравнение

- `==/2`, `\==/2`, `\=/2`
- `@</2`, `@>/2`, `@=</2`, `@>=/2`
- `compare/3`

```prolog
compare(<, a, b).      % true
compare(>, 2, 1).      % true
```

## 7.10. Строки

- `string_concat/3`
- `atomic_list_concat/2`, `atomic_list_concat/3`
- `atom_string/2`
- `term_string/2`
- `number_string/2`
- `split_string/4`
- `sub_string/5`

## 7.11. Семантическая унификация

- `semantic_on/0`, `semantic_on/1`
- `semantic_off/0`
- `semantic_threshold/1`
- `semantic_status/1`

## 7.12. Диагностика Numba

```python
from prolog.predicates import arith_numba_status, warmup_arith_numba
print(arith_numba_status())
warmup_arith_numba()
```

---

# 8. Модуль solver — высокоуровневый интерфейс

## 8.1. Класс `Solver`

```python
from prolog.solver import Solver

solver = Solver("""
parent(ivan, alexey).
parent(alexey, petr).
""", context={"key": "value"})
```

### 8.1.1. Атрибуты

- `solver.database` — объект `Database`.
- `solver.last_tracer` — последний трассировщик.

### 8.1.2. Методы

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

Форматирует результат для вывода.

```python
from prolog.solver import ParseSolution

parsed = ParseSolution("parent(ivan, X).", [{'X': Term('alexey')}])
print(parsed['output'])   # X = alexey
print(parsed['solution']) # [{'X': 'alexey'}]
```

Ключи результата:
- `query`
- `solution`
- `output`
- `count` (если несколько решений)
- `explanation` (если передано)

## 8.3. Форматирование объяснений

```python
from prolog.solver import format_explanation_for_human, format_dict_explanation

print(format_explanation_for_human(explanation))
```

---

# 9. Модуль bridge — мост Prolog ↔ Python

## 9.1. `prolog_to_python(term)`

```python
from prolog.bridge import prolog_to_python
from prolog.terms import Term, Number, Variable

prolog_to_python(Number(42))           # 42
prolog_to_python(Term('ivan'))          # 'ivan'
prolog_to_python(Variable('X'))         # None
prolog_to_python(Term('.', [Number(1), Term('[]')]))  # [1]
```

Соответствия:

| Prolog | Python |
|---|---|
| `Number` | `int` / `float` |
| `Variable` | `None` |
| атом | `str` |
| строка в кавычках | `str` без кавычек |
| список | `list` |
| составной терм | `str` |

## 9.2. `python_to_prolog(obj)`

```python
from prolog.bridge import python_to_prolog

python_to_prolog(42)          # Number(42)
python_to_prolog("ivan")      # Term('"ivan"')
python_to_prolog([1, 2, 3])   # список Prolog
python_to_prolog(True)        # Term('true')
python_to_prolog(None)        # Term('[]')
python_to_prolog({'a': 1})    # список пар '-'(K, V)
```

## 9.3. `prolog_list_to_python(term)`

Возвращает список **термов** (а не Python-значений).

```python
prolog_list_to_python(Term('.', [Number(1), Term('.', [Number(2), Term('[]')])]))
# [Number(1), Number(2)]
```

---

# 10. Модуль actions — Python-действия

## 10.1. Общее

Декораторы `simple_action`, `effector_action`, `generator_action` превращают Python-функции в предикаты Prolog.

Все они:
- вешают метаданные `_is_prolog_predicate`, `_predicate_name`, `_predicate_arity`;
- вешают метаданные `_agent_action`, `_action_name`, `_action_arity`, `_action_description`, `_action_type`;
- принимают `context=database.context`.

## 10.2. `simple_action(name, arity, output_arg=-1, description=None)`

```python
from prolog.actions import simple_action

@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2
```

Правила:
- `output_arg` — индекс выходного аргумента (0-based).
- `output_arg=-1` → эффектор.
- Входные аргументы должны быть конкретизированы.
- Возвращает одно значение.

## 10.3. `effector_action(name, arity, description=None)`

```python
from prolog.actions import effector_action

@effector_action('print_msg', 1)
def print_msg(msg, context):
    print("MSG:", msg)
```

- Все аргументы входные.
- Возвращает `None`.
- Одно решение при успехе.

## 10.4. `generator_action(name, arity, output_arg=-1, description=None)`

```python
from prolog.actions import generator_action

@generator_action('count', 2, output_arg=1)
def count(n, context):
    for i in range(1, n + 1):
        yield i
```

- Функция возвращает итератор.
- Каждое значение → отдельное решение.
- `output_arg=-1` → эффектор (итератор игнорируется).

## 10.5. Использование

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

## 10.6. Обработка ошибок

Любое исключение внутри действия → предикат просто не даёт решений (fail). Это позволяет избежать падения всего запроса из-за ошибки в одном действии.

---

# 11. Модуль agent — класс Agent

## 11.1. Создание

```python
from prolog.agent import Agent

agent = Agent(program="...", pl_file="...", context={"llm": client})
```

- `program` — текст программы.
- `pl_file` — файл с программой.
- `context` — словарь внешних ресурсов.

## 11.2. Регистрация действий

```python
agent.register_action(func)
agent.register_action_module(module)
```

## 11.3. Список действий

```python
agent.list_actions()
# [{'name': 'double', 'arity': 2, 'description': ..., 'type': 'simple', ...}]
```

## 11.4. Запросы

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

Отличает «провал» (`None`) от «успех без переменных» (`{'__success__': True}`).

### 11.4.3. `stream(query_text)`

```python
for solution in agent.stream("parent(ivan, X)."):
    print(solution)
```

## 11.5. Работа с базой

```python
agent.add_fact("parent(ivan, alexey).")
agent.add_facts("""
    parent(ivan, maria).
    parent(alexey, petr).
""")
agent.retract_fact("parent(ivan, alexey).")
agent.retract_all("parent(ivan, _).")
```

## 11.6. Сохранение и загрузка

```python
agent.save_kb("kb.pl", format="prolog")
agent.save_kb("kb.pkl", format="binary")
agent.load_kb("kb.pl", format="prolog", clear=True)
agent.clear_kb()
```

## 11.7. Контекст

```python
agent.context['key'] = 'value'
print(agent.context)
```

## 11.8. Встроенные предикаты

### 11.8.1. `set_state/1`

```prolog
set_state(mode(search)).
```

Заменяет все факты с данным функтором/арностью.

### 11.8.2. `update_state/1`

```prolog
update_state(observed(bravo, search, 10.0, 5.0, 17)).
```

Добавляет факт, не удаляя старые.

## 11.9. Legacy-декоратор `agent_action`

```python
from prolog.agent import agent_action

@agent_action('my_action', 1, description="...", action_type="generic")
def my_action(goal, database, subst, depth):
    yield subst.copy()
```

Оставлен для обратной совместимости. Новый код должен использовать `simple_action`/`effector_action`/`generator_action`.

---

# 12. Модуль tracer — трассировка доказательств

## 12.1. Класс `ProofStep`

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

Методы:
- `to_dict()`
- `to_tree_string()`
- `__str__`

## 12.2. Класс `ProofTracer`

```python
from prolog.tracer import ProofTracer

tracer = ProofTracer()
```

Методы:
- `start_goal(goal_str, depth)` — начать шаг.
- `end_goal(success)` — завершить шаг.
- `add_unification(var, value)`
- `set_rule(rule_str)`
- `set_builtin(builtin)`
- `set_substitutions(before, after)`
- `get_proof_tree()` — текстовое дерево.
- `get_proof_dict()` — словарь.
- `clear()`

## 12.3. Пример

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

Вывод:

```
ДЕРЕВО ДОКАЗАТЕЛЬСТВА:
==================================================
ancestor(ivan, X)
  parent(ivan, alexey)
    ...
```

---

# 13. Модуль embedding_unify — семантическая унификация

## 13.1. Концепция

Семантическая унификация сравнивает атомы не по точному равенству, а по смыслу (эмбеддингам). Это позволяет унифицировать `москва` ↔ `moscow`, `Масква` ↔ `москва` и т.д.

## 13.2. Слои сопоставления

1. **Layer 1**: нормализация текста (`strip`, снятие кавычек, `lower()`).
2. **Layer 2a**: почти идентичные строки (`lev_ratio <= lev_typo_max`) — match без модели.
3. **Layer 3**: hard — `cos >= threshold`.
4. **Layer 2b**: soft — `soft_cos_min <= cos < threshold` и `lev_ratio <= lev_max`.

## 13.3. Backend’ы

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

### 13.3.3. Кастомный

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

## 13.4. Параметры

| Параметр | По умолчанию | Назначение |
|---|---|---|
| `threshold` | 0.82 | Hard-порог cos |
| `soft_cos_min` | 0.55 | Нижняя граница cos для soft |
| `lev_max` | 0.35 | Максимальный lev_ratio для soft |
| `lev_typo_max` | 0.20 | Верхняя граница lev_ratio для typo |
| `text_prefix` | авто | Префикс эмбеддера |
| `max_cache` | 10000 | Размер кэша эмбеддингов |

## 13.5. API

- `semantic_on(...)` — включить.
- `semantic_off()` — выключить.
- `semantic_status()` — статус.
- `semantic_set_threshold(value)` — изменить порог.
- `semantic_set_corridor(...)` — изменить коридор.
- `semantic_set_prefix(prefix)` — изменить префикс.
- `semantic_get_unifier()` — получить унификатор.
- `install_embedding_unifier(...)` — низкоуровневая установка.
- `uninstall_embedding_unifier()` — снять.

## 13.6. Управление из Prolog

```prolog
:- semantic_on(0.82).
semantic_status(S).
:- semantic_threshold(0.75).
:- semantic_off.
```

## 13.7. Пример

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

## 13.8. Подавление вывода ST

По умолчанию все служебные сообщения `sentence-transformers` / `transformers` / HF Hub отключены. Чтобы вернуть их:

```bash
export PROLOG_ST_VERBOSE=1
```

## 13.9. Известные префиксы

| Модель | Префикс |
|---|---|
| `multilingual-e5` | `query: ` |
| `intfloat-e5` | `query: ` |
| `nomic-embed-text-v2` | `search_query: ` |
| `nomic-embed-text-v1` | `search_query: ` |

## 13.10. Алиасы моделей

```python
from prolog.embedding_unify import resolve_model_id

resolve_model_id('multilingual-e5-large')
# 'intfloat/multilingual-e5-large'
```

---

# 14. Модуль arith_numba — ускорение арифметики

## 14.1. Принцип

Терм Prolog компилируется в плоские numpy-массивы ОДИН раз (кэш по `id(term)`). JIT-ядро — нерекурсивная стек-машина. Работает на любых версиях Numba и любых ARM-вариантах.

## 14.2. API

```python
from prolog.arith_numba import (
    try_eval_fast, evaluate_expression_numba,
    warmup_arith_numba, arith_numba_status, clear_arith_cache,
    NOT_ATTEMPTED
)
```

### `try_eval_fast(expr, subst)`

Возвращает:
- число — успешное вычисление через JIT;
- `NOT_ATTEMPTED` — Numba не применима;
- бросает `PrologError` — семантическая ошибка.

### `warmup_arith_numba()`

Прогревает JIT-ядро. Рекомендуется вызвать один раз на старте.

### `arith_numba_status()`

Снимок состояния.

### `clear_arith_cache()`

Очистка кэша.

## 14.3. Переменные окружения

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `PROLOG_NUMBA_THRESHOLD` | 20/30/40 | Порог по числу узлов |
| `PROLOG_NUMBA_FASTMATH` | 0 | Включить fastmath |
| `PROLOG_NUMBA_CACHE` | 10000 | Размер кэша |
| `PROLOG_NUMBA_DISABLE` | 0 | Полностью отключить |

## 14.4. Пример

```python
from prolog.predicates import arith_numba_status, warmup_arith_numba

print(arith_numba_status())
# {'available': True, 'active': True, 'numba_version': (0, 57, 0), ...}

warmup_arith_numba()
```

## 14.5. Пороги по платформе

| Платформа | Порог |
|---|---:|
| x86_64 | 20 |
| 64-bit ARM | 30 |
| 32-bit ARM | 40 |

---

# 15. Модуль prologchain — интеграция с LangChain

## 15.1. Класс `PrologQueryTemplate`

```python
from prolog.prologchain import PrologQueryTemplate

template = PrologQueryTemplate.from_template("parent({person}, X).")
result = template.invoke({"person": "ivan"})
# {'query': 'parent(ivan, X).'}
```

Методы:
- `from_template(template)`
- `format(**kwargs)`
- `format_prompt(**kwargs)`
- `invoke(input)`
- `__or__(other)` — поддержка `|`.

## 15.2. Класс `PrologChain`

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

## 15.3. Класс `PrologOutputFormatter`

```python
from prolog.prologchain import PrologOutputFormatter

formatter = PrologOutputFormatter(include_query=True, include_stats=True)
print(formatter.invoke(result))
```

## 15.4. Готовая цепь

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

## 15.6. Структура результата

| Ключ | Описание |
|---|---|
| `success` | `True` / `False` |
| `query` | Текст запроса |
| `program` | Программа |
| `output` | Форматированный вывод |
| `count` | Количество решений |
| `solutions` | Список решений |
| `result` | `True` / `False` (если без переменных) |
| `explanation` | Объяснение (если включено) |
| `error` | Текст ошибки (если `success=False`) |

---

# 16. Модуль proagent_gradio — Gradio-сессии

## 16.1. Классы сессий

### 16.1.1. `BaseGradioSession`

Базовый класс с очередями ввода/вывода.

### 16.1.2. `AgentGradioSession`

Сессия для агента. Атрибуты:
- `history`
- `ui_values`

Методы:
- `start(agent, query)`
- `get_input()`
- `get_ui_value(name)`
- `set_ui_value(name, value)`
- `is_running()`
- `stop()`

## 16.2. Реестр сессий

```python
from prolog.proagent_gradio import (
    register_session, get_session, remove_session
)

register_session(session_id, session)
session = get_session(session_id)
remove_session(session_id)
```

## 16.3. Функции для агента

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

## 16.4. Действия

| Действие | Назначение |
|---|---|
| `gradio_print/1` | Вывод в UI |
| `gradio_read_input/1` | Чтение ввода |
| `gradio_get_ui_value/2` | Получить значение UI |
| `gradio_set_ui_value/2` | Установить значение UI |
| `format/3` | Форматирование строки |

## 16.5. Регистрация

```python
from prolog.proagent_gradio import register_gradio_actions

register_gradio_actions(agent)
```

## 16.6. Пример Prolog

```prolog
run :-
    gradio_print('Введите что-нибудь:'),
    gradio_read_input(Input),
    format("Вы ввели: ~w", [Input], Msg),
    gradio_print(Msg).
```

## 16.7. `format/3`

```prolog
format("Hello, ~w!", ["World"], Result).
% Result = "Hello, World!"
```

Поддерживаются `~w`, `~d`, `~f`, `~s`.

---

# 17. Справочник по подмножеству Prolog

## 17.1. Синтаксис

### 17.1.1. Термы

| Конструкция | Пример |
|---|---|
| Атом | `foo`, `'bar baz'`, `"string"` |
| Переменная | `X`, `_`, `_Foo` |
| Число | `42`, `3.14`, `1e-3` |
| Составной терм | `f(a, b)` |
| Список | `[a, b, c]`, `[H|T]` |
| Оператор | `a + b`, `X is Y * 2` |

### 17.1.2. Правила

```prolog
head :- body1, body2, body3.
head.
```

### 17.1.3. Запросы

```prolog
?- goal1, goal2.
```

В `Agent.query` и `Solver.find_solutions` используется форма без `?-`, но с точкой.

## 17.2. Управляющие конструкции

| Конструкция | Синтаксис |
|---|---|
| Конъюнкция | `A, B` |
| Дизъюнкция | `A ; B` |
| If-then | `Cond -> Then` |
| If-then-else | `(Cond -> Then ; Else)` |
| Отсечение | `!` |
| Отрицание | `\+ Goal` |
| True | `true` |
| Fail | `fail`, `false` |
| Repeat | `repeat` |

## 17.3. Встроенные предикаты

### 17.3.1. Арифметика

`is/2`, `>/2`, `</2`, `>=/2`, `=</2`, `=:=/2`, `=\=/2`

### 17.3.2. Списки

`member/2`, `append/3`, `length/2`, `memberchk/2`

### 17.3.3. Ввод/вывод

`write/1`, `nl/0`

### 17.3.4. База данных

`asserta/1`, `assertz/1`, `retract/1`, `retractall/1`, `abolish/1`, `clause/2`, `current_predicate/1`

### 17.3.5. Мета-предикаты

`call/1..5`, `once/1`, `ignore/1`, `forall/2`, `findall/3`, `bagof/3`, `setof/3`

### 17.3.6. Термы

`functor/3`, `arg/3`, `=../2`, `copy_term/2`, `term_variables/2`, `numbervars/3`

### 17.3.7. Типы

`var/1`, `nonvar/1`, `atom/1`, `integer/1`, `float/1`, `number/1`, `atomic/1`, `compound/1`, `ground/1`, `callable/1`

### 17.3.8. Сравнение

`==/2`, `\==/2`, `\=/2`, `@</2`, `@>/2`, `@=</2`, `@>=/2`, `compare/3`

### 17.3.9. Строки

`string_concat/3`, `atomic_list_concat/2`, `atomic_list_concat/3`, `atom_string/2`, `term_string/2`, `number_string/2`, `split_string/4`, `sub_string/5`

### 17.3.10. Семантическая унификация

`semantic_on/0`, `semantic_on/1`, `semantic_off/0`, `semantic_threshold/1`, `semantic_status/1`

## 17.4. Не реализовано

- Модульная система (`:- module`, `:- use_module`).
- DCG (`-->`).
- Пользовательские операторы (`:- op(...)`).
- `catch/3`, `throw/1`.
- `between/3`, `succ/2`, `plus/3`.
- `sort/2`, `msort/2`, `keysort/2`.
- `read/1`, `open/3`, `close/1`, `read_term/2`, `write_term/2`.
- `atom_length/2`, `atom_concat/3`, `atom_codes/2`, `char_code/2`, `atom_chars/2`.
- Таблицы и constraint-программирование (`library(clpfd)`).
- `ground/1` работает, но `ground/1` внутри `bagof/3` учитывается частично.

## 17.5. Частично реализованные предикаты

| Предикат | Ограничение |
|---|---|
| `length/2` | С открытым списком и свободным N выдаёт одно минимальное решение |
| `append/3` | Не полон во всех экзотических режимах |
| `retract/1` | Сопоставляет в основном голову правила |
| `sub_string/5` | Не все режимы ISO |
| `bagof/3` | Группировка по строковому ключу |
| `setof/3` | Сортировка по строковому представлению |

---

# 18. Практические примеры

## 18.1. Родственные отношения

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

## 18.2. Списки и findall

```python
agent = Agent("""
parent(ivan, alexey).
parent(ivan, maria).
""")

print(agent.query("findall(X, parent(ivan, X), L)."))
# L = ['alexey', 'maria']
```

## 18.3. Арифметика

```python
agent = Agent("""
fact(0, 1).
fact(N, F) :- N > 0, N1 is N - 1, fact(N1, F1), F is N * F1.
""")

print(agent.query("fact(5, F)."))
# F = 120
```

## 18.4. Отсечение

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

## 18.6. Действия агента

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

## 18.7. Семантическая унификация

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

## 18.8. Трассировка

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
    gradio_print('Начинаем работу...'),
    gradio_read_input(Input),
    format("Получено: ~w", [Input], Msg),
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

## 18.11. Сохранение/загрузка базы

```python
agent = Agent("fact(1). fact(2).")
agent.add_fact("fact(3).")
agent.save_kb("kb.pl", format="prolog")
agent.save_kb("kb.pkl", format="binary")

agent2 = Agent()
agent2.load_kb("kb.pl", format="prolog")
print(agent2.query("fact(X)."))
```

## 18.12. Numba-ускорение

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

# 19. Отладка и диагностика

## 19.1. Флаги модулей

| Модуль | Флаг |
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

## 19.2. Диагностика Numba

```python
from prolog.predicates import arith_numba_status
print(arith_numba_status())
```

## 19.3. Диагностика семантической унификации

```python
from prolog.embedding_unify import semantic_status
print(semantic_status())
```

## 19.4. Диагностика действий

```python
agent.list_actions()
```

## 19.5. Трассировка доказательств

```python
result, explanation = agent.query("goal.", with_explanation=True)
print(explanation.get_proof_tree())
```

## 19.6. Обработка исключений

- `PrologError` — семантическая ошибка.
- `CutException` / `RepeatException` — управляющие.
- Прочие `Exception` в мета-предикатах печатаются с traceback и пробрасываются.

---

# 20. Лучшие практики

## 20.1. Программирование на Prolog

1. **Используйте отсечение осторожно** — оно может скрыть альтернативные решения.
2. **Предпочитайте if-then-else** для детерминированных ветвлений.
3. **Проверяйте порядок целей** — Prolog ищет слева направо.
4. **Индексируйте правила** — ставьте константный первый аргумент.
5. **Не злоупотребляйте `assert`/`retract`** — они меняют индекс.
6. **Используйте `findall` вместо ручного сбора**.
7. **Проверяйте `ground/1`** перед арифметикой.

## 20.2. Python-действия

1. **Используйте `simple_action`/`effector_action`/`generator_action`**, а не legacy `agent_action`.
2. **Конкретизируйте входные аргументы** — иначе действие не сработает.
3. **Возвращайте `None`** для отказа (эффектор).
4. **Ловите исключения внутри действия**, если хотите тихий fail.
5. **Используйте `context`** для доступа к внешним ресурсам.

## 20.3. Производительность

1. **Используйте TCO** — пишите рекурсию в хвостовой форме.
2. **Избегайте глубокой нехвостовой рекурсии** — используйте аккумуляторы.
3. **Индексируйте первый аргумент** — константы в голове ускоряют поиск.
4. **Включайте Numba** для больших арифметических выражений.
5. **Кэшируйте эмбеддинги** — размер кэша настраивается.
6. **Используйте `stream`** для больших результатов.

## 20.4. Семантическая унификация

1. **Подбирайте порог** эмпирически (обычно 0.75–0.85).
2. **Используйте префиксы** для E5/nomic моделей.
3. **Кэшируйте эмбеддинги** — `max_cache` важен.
4. **Отключайте семантику** там, где нужна строгость.
5. **Проверяйте `semantic_status()`** для диагностики.

## 20.5. Безопасность

1. **Не выполняйте непроверенные запросы** — Prolog может уйти в бесконечный поиск.
2. **Ограничивайте глубину** через `depth` в кастомных предикатах.
3. **Изолируйте контекст** — не давайте действиям доступ ко всему.
4. **Проверяйте входные данные** перед парсингом.

---

# 21. Ограничения и известные особенности

## 21.1. Ограничения языка

- Нет модулей.
- Нет DCG.
- Нет пользовательских операторов.
- Нет `catch/3`/`throw/1`.
- Нет `between/3`, `succ/2`.
- Нет `sort/2`, `msort/2`.
- Нет `read/1`, `open/3`.
- Нет `atom_length/2`, `atom_concat/3`, `atom_codes/2`.
- Нет CLP(FD).

## 21.2. Особенности реализации

- **Occurs check по умолчанию выключен** — для производительности.
- **Анонимная `_`** унифицируется всегда.
- **Переменные сравниваются по имени**, а не по объекту.
- **Обрезка `subst`** (Патч 5.8) может влиять на переменные, не достижимые из целей.
- **`if-then-else`** реализован через `_IfThenBarrier`, а не через cut.
- **Мета-предикаты не глотают исключения** (с версии 5.3).

## 21.3. Совместимость

- Python 3.8+.
- Numba — опционально.
- sentence-transformers — опционально.
- LangChain — опционально.
- Gradio — опционально.

## 21.4. Известные баги и workaround

| Проблема | Workaround |
|---|---|
| `length/2` с открытым списком | Используйте `length(L, N)` с конкретным `N` |
| `append/3` в нестандартных режимах | Используйте `member/2` или `findall` |
| Семантическая унификация медленная | Уменьшите `max_cache`, используйте кэш |
| Numba падает на ARM | Установите `PROLOG_NUMBA_DISABLE=1` |
| TCO не срабатывает | Пишите рекурсию в хвостовой форме |

---

# Заключение

PyAgentProlog — это полноценный Prolog-движок на Python, который сочетает классический логический вывод с современными возможностями:

- **Агенты с Python-действиями** — интеграция с любыми Python-библиотеками.
- **Семантическая унификация** — работа с нечёткими понятиями.
- **Трассировка доказательств** — объяснимость вывода.
- **LangChain и Gradio** — интеграция с LLM и UI.
- **Numba** — ускорение арифметики на слабом железе.

Пакет подходит как для обучения Prolog, так и для промышленных нейро-символических приложений.

