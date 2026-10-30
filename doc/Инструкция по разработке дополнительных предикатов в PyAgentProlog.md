
Фреймворк позволяет легко расширять набор предикатов, не изменяя исходный код. Для этого используются:

- Декоратор `@prolog_predicate(name, arity)` из модуля `prolog.predicates`.
- Методы `register_custom_predicate` у `Database` или `register_action` у `Agent` (для действий агента).

## 1. Регистрация через декоратор `@prolog_predicate`

Этот способ подходит для предикатов, которые работают на уровне базы знаний (как стандартные `is/2`, `=/2`, `assert/1`). Функция должна принимать параметры: `(goal, database, subst, depth)` и возвращать список подстановок.

```python
from prolog.predicates import prolog_predicate
from prolog.terms import Term, Number

@prolog_predicate('my_plus', 3)
def my_plus(goal, database, subst, depth):
    """my_plus(A, B, C) :- C is A + B."""
    if len(goal.arguments) != 3:
        return []
    # Получаем аргументы с учётом текущей подстановки
    a = goal.arguments[0].substitute(subst)
    b = goal.arguments[1].substitute(subst)
    c_var = goal.arguments[2]

    # Проверяем, что A и B — числа
    if not isinstance(a, Number) or not isinstance(b, Number):
        return []
    result = Number(a.value + b.value)

    # Унифицируем результат с третьим аргументом
    new_subst = c_var.unify(result, subst.copy())
    return [new_subst] if new_subst is not None else []

# Регистрация в солвере
solver = Solver(program)
solver.database.register_custom_predicate('my_plus', 3, my_plus)
# Теперь можно использовать my_plus(1,2,X) в запросах
```

## 2. Регистрация через `Agent.register_action`

Если предикат предназначен для использования **в агенте** (с доступом к контексту и возможностью использовать декораторы `simple_action` и т.д.), его можно зарегистрировать через метод `register_action`. При этом функция автоматически получит доступ к `context` из агента.

```python
from prolog.agent import Agent, simple_action

@simple_action('my_greet', 1, output_arg=0)
def my_greet(name, context=None):
    return f"Hello, {name}!"

agent = Agent(program="...")
agent.register_action(my_greet)
# В Prolog: my_greet('World', Greeting). -> Greeting = 'Hello, World!'
```

## 3. Регистрация целого модуля

Если у вас есть модуль, содержащий несколько декорированных предикатов (как `@prolog_predicate`), их можно зарегистрировать одной командой:

```python
import my_custom_predicates
solver.database.register_predicate_module(my_custom_predicates)
```

Или для агента:

```python
agent.register_action_module(my_custom_predicates)
```

## 4. Пример: предикат для работы с LLM

```python
# llm_predicates.py
from prolog.predicates import prolog_predicate
from prolog.terms import Term

@prolog_predicate('llm_generate', 2)
def llm_generate(goal, database, subst, depth):
    """llm_generate(+Prompt, -Response)"""
    prompt_arg = goal.arguments[0].substitute(subst)
    response_var = goal.arguments[1]

    if not isinstance(prompt_arg, Term) or prompt_arg.arguments:
        return []
    prompt = str(prompt_arg.functor).strip('"')

    # Получаем LLM-клиент из контекста базы данных
    llm = database.context.get('llm')
    if not llm:
        return []
    response = llm.generate(prompt)

    response_term = Term(f'"{response}"')
    new_subst = response_var.unify(response_term, subst.copy())
    return [new_subst] if new_subst is not None else []

# Использование:
solver = Solver(program, context={'llm': my_llm_client})
solver.database.register_custom_predicate('llm_generate', 2, llm_generate)
```

## 5. Важные замечания

- Функция встроенного предиката **не должна быть генератором** (возвращать `yield`). Если нужно множество решений, возвращайте список подстановок.
- Для реализации генераторов с backtracking используйте `RepeatException` (как в `repeat/0`) или возвращайте несколько подстановок в списке.
- Предикаты, которые изменяют базу знаний (`assert`, `retract`), должны вызывать `database.add_rule()` и `database.retract_all()` и возвращать подстановку.
- Для отсечения альтернатив используйте `raise CutException(subst)`.
