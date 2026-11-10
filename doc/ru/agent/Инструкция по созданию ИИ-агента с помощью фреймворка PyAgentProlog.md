Фреймворк **proagent** позволяет создавать интеллектуальных агентов, логика поведения которых описывается на языке Prolog, а вычислительно-ёмкие или внешние действия (вызов LLM, работа с API, сенсоры, эффекторы) реализуются на Python. Такой подход обеспечивает декларативность, лёгкость модификации и мощь логического вывода, сохраняя при этом доступ ко всей экосистеме Python.

## 1. Установка и зависимости

Фреймворк PyAgentProlog использует интерпретатор Prolog v3.3 и выше (пакет `prolog`). Убедитесь, что этот пакет доступен для импорта (он должен находиться в директории, указанной в `sys.path`). Сам proagent распространяется в виде модуля `prolog.agent`.

Для работы примеров вам также потребуется Python 3.7+.

## 2. Основные компоненты агента

Агент состоит из двух чётко разделённых частей:

- **Prolog-программа** – факты и правила, описывающие логику переходов, обработку событий, выбор действий. Состояние агента хранится в базе знаний в виде динамических фактов.
- **Python-действия** – функции, которые вызываются из Prolog как встроенные предикаты. Они могут обращаться к внешним ресурсам (LLM, базы данных, файловая система и т.п.) через контекст агента.

Связующим звеном является класс `Agent`, который содержит интерпретатор Prolog и реестр зарегистрированных действий.

## 3. Определение действий агента (Python)

Действия создаются с помощью специальных декораторов, которые скрывают детали взаимодействия с Prolog и позволяют писать функции в естественном для Python стиле.
[[Инструкция по программированию действий агента во фреймворке PyAgentProlog]]

### 3.1. Декоратор `@simple_action`

Используется для действий, которые принимают входные аргументы и возвращают **одно значение**. Это значение будет унифицировано с выходным аргументом предиката.

```python
from prolog.agent import simple_action

@simple_action('read_input', 1, output_arg=0)
def read_input(context):
    """
    read_input(-Text)
    Считывает строку из стандартного ввода и возвращает её.
    """
    line = input(">> ").strip()
    return line
```

- `'read_input'` – имя предиката в Prolog.
- `1` – общая арность (количество аргументов).
- `output_arg=0` – индекс аргумента (с нуля), в который будет помещён результат. В данном случае предикат имеет один аргумент, который является выходным.
- Функция принимает все входные аргументы (кроме выходного) в том же порядке, что и в Prolog, плюс параметр `context`. В примере входных аргументов нет, поэтому функция принимает только `context`.
- Возвращаемое значение будет автоматически преобразовано в Prolog-терм (строки оборачиваются в кавычки, списки – в списки Prolog, числа – в числа).

### 3.2. Декоратор `@effector_action`

Для действий, которые только выполняют побочный эффект и **не возвращают значения**. Все аргументы считаются входными.

```python
from prolog.agent import effector_action

@effector_action('print_message', 1)
def print_message(msg, context):
    """
    print_message(+Message)
    Выводит сообщение на экран.
    """
    print(f"[Агент]: {msg}")
```

- Функция принимает все аргументы предиката (в порядке их следования) и `context`.
- Возвращаемое значение не используется; действие считается успешным, если функция выполнилась без исключений.

### 3.3. Декоратор `@generator_action`

Для действий, которые могут породить **несколько альтернативных решений** (например, перебор значений). Функция должна быть генератором (использовать `yield`).

```python
from prolog.agent import generator_action

@generator_action('range', 2, output_arg=1)
def range_action(limit, context):
    """
    range(+Limit, -Value)
    Генерирует числа от 0 до Limit-1.
    """
    for i in range(limit):
        yield i
```

- Каждое `yield` создаёт отдельное решение, в котором выходной аргумент унифицируется с очередным значением.
- Позволяет реализовать классический Prolog-поиск с возвратами.

### 3.4. Что получает функция действия?

Все декораторы передают в функцию следующие аргументы:

- Все **входные аргументы** (кроме выходного для `simple_action` и `generator_action`) в том же порядке, что и в Prolog, преобразованные в Python-объекты (числа, строки, списки).
- Параметр `context` – словарь, переданный при создании агента. В нём можно хранить любые внешние ресурсы: клиенты LLM, соединения с БД, конфигурацию и т.п.

### 3.5. Преобразование типов

Фреймворк автоматически преобразует Prolog-термы в Python-объекты и обратно с помощью функций `prolog_to_python` и `python_to_prolog`.

| Prolog | Python |
|--------|--------|
| Число (`42`, `3.14`) | `int` / `float` |
| Строка в кавычках (`"hello"`) | `str` (без кавычек) |
| Атом (`hello`) | `str` |
| Список (`[a, b, c]`) | `list` |
| Пустой список (`[]`) | `[]` |
| Составной терм (`f(1,2)`) | `str` (упрощённо) |
| Словарь (как список пар `-`) | `dict` (если передан из Python) |

## 4. Написание логики на Prolog

Prolog-программа описывает поведение агента. Рекомендуется использовать **динамические предикаты** (объявленные через `:- dynamic`), чтобы можно было изменять их с помощью встроенных предикатов `set_state/1` и `update_state/1`.

### 4.1. Встроенные предикаты для управления состоянием

- **`set_state(Term)`** – заменяет **все** факты с тем же функтором и арностью, что и `Term`, на новый факт `Term`. Требует, чтобы `Term` был конкретизирован (не содержал переменных). Идеально для хранения одиночного значения (например, текущей фазы).
- **`update_state(Term)`** – добавляет `Term` как новый факт, не удаляя старые. Подходит для накопления информации (история, инвентарь).

Эти предикаты уже зарегистрированы в каждом экземпляре `Agent`.

### 4.2. Организация цикла агента

Агент обычно работает в цикле: на каждом шаге проверяется текущее состояние, выполняется действие, состояние обновляется. Цикл реализуется рекурсивным правилом.

```prolog
:- dynamic state/2.
:- dynamic memory/1.

init :-
    set_state(state(phase, init)),
    set_state(memory([])).

step :-
    state(phase, Phase),
    handle_phase(Phase, NextPhase),
    set_state(state(phase, NextPhase)),
    continue(NextPhase).

continue(final) :- !.
continue(_) :- step.

handle_phase(init, working) :-
    print_message('Агент инициализирован. Введите действие:'),
    read_input(Action),
    set_state(memory(Action)),
    print_message('Действие запомнено.').

handle_phase(working, final) :-
    memory(LastAction),
    print_message('Последнее действие:'),
    print_message(LastAction),
    llm_echo(LastAction, Analysis),
    print_message('LLM ответила:'),
    print_message(Analysis),
    print_message('Завершаю работу.').

handle_phase(final, final) :-
    print_message('Агент уже завершил работу.').

run :- init, step.
```

Пояснения:

- `continue(final) :- !.` – при достижении фазы `final` отсечение (!) предотвращает переход к следующему правилу, и цикл останавливается.
- Все внешние взаимодействия (ввод, вывод, вызов LLM) вынесены в действия на Python.

## 5. Создание и запуск агента

### 5.1. Инициализация агента

```python
from prolog.agent import Agent

program = """ ... ваш код на Prolog ... """

context = {
    'llm_client': my_llm_client,   # любой Python-объект
    'user_name': 'Тестер'
}

agent = Agent(program, context=context)
# Альтернативно: загрузить программу из файла
# agent = Agent(pl_file='agent.pl', context=context)
```

### 5.2. Регистрация действий

```python
import sys
agent.register_action_module(sys.modules[__name__])   # регистрирует все функции с @agent_action, @simple_action и т.д. в текущем модуле
```

Можно также регистрировать действия по одному:

```python
agent.register_action(read_input)
agent.register_action(print_message)
```

### 5.3. Просмотр зарегистрированных действий

```python
for act in agent.list_actions():
    print(f"{act['name']}/{act['arity']}: {act['description']} ({act['type']})")
```

### 5.4. Выполнение запросов

```python
# Запустить основной цикл агента
result = agent.query('run.')

# Получить значение из базы знаний
mem_result = agent.query('memory(X).')
if 'solutions' in mem_result:
    for sol in mem_result['solutions']:
        print(f"X = {sol['X']}")
```

Метод `query` возвращает словарь с ключами:
- `'success'` – булево значение,
- `'solutions'` – список решений (каждое – словарь переменных),
- `'output'` – отформатированный вывод (для удобства),
- и др.

Для получения только первого решения используйте `agent.query_one('memory(X).')`.

### 5.5. Трассировка выполнения

```python
result, explanation = agent.query('run.', with_explanation=True)
print(explanation.get_proof_tree())
```

Это выведет дерево доказательства, что помогает отлаживать логику.

## 6. Полный пример агента

Ниже приведён код из файла `example_proagent.py`, который объединяет все описанные элементы.

```python
#!/usr/bin/env python3
import sys
sys.path.append('d:/BFX/Repositories/')
from prolog.agent import Agent, simple_action, effector_action

# ---------- Определение действий ----------
@simple_action('read_input', 1, output_arg=0)
def read_input(context):
    line = input(">> ").strip()
    return line

@effector_action('print_message', 1)
def print_message(msg, context):
    print(f"[Агент]: {msg}")

@simple_action('llm_echo', 2, output_arg=1)
def llm_echo(prompt, context):
    return f"Эхо: {prompt}"

# ---------- Prolog-программа ----------
AGENT_PROGRAM = """
:- dynamic state/2.
:- dynamic memory/1.

init :-
    set_state(state(phase, init)),
    set_state(memory([])).

step :-
    state(phase, Phase),
    handle_phase(Phase, NextPhase),
    set_state(state(phase, NextPhase)),
    continue(NextPhase).

continue(final) :- !.
continue(_) :- step.

handle_phase(init, working) :-
    print_message('Агент инициализирован. Введите действие:'),
    read_input(Action),
    set_state(memory(Action)),
    print_message('Действие запомнено.').

handle_phase(working, final) :-
    memory(LastAction),
    print_message('Последнее действие:'),
    print_message(LastAction),
    llm_echo(LastAction, Analysis),
    print_message('LLM ответила:'),
    print_message(Analysis),
    print_message('Завершаю работу.').

handle_phase(final, final) :-
    print_message('Агент уже завершил работу.').

run :- init, step.
"""

def main():
    agent = Agent(AGENT_PROGRAM, context={'llm': None})
    agent.register_action_module(sys.modules[__name__])

    print("Зарегистрированные действия:")
    for act in agent.list_actions():
        print(f"  {act['name']}/{act['arity']}: {act['description']}")

    print("\nЗапуск агента (запрос 'run.')")
    result = agent.query('run.')
    print("Результат:", result)

if __name__ == "__main__":
    main()
```

Запустите этот скрипт, введите несколько строк, и агент выполнит цикл, выводя сообщения и имитируя ответ LLM.

## 7. Расширенные возможности

### 7.1. Контекст

Контекст (`context`) доступен во всех действиях и может использоваться для передачи внешних ресурсов. Например:

```python
@simple_action('ask_llm', 2, output_arg=1)
def ask_llm(prompt, context):
    client = context['llm_client']
    response = client.generate(prompt)
    return response
```

### 7.2. Динамическое изменение базы знаний из Python

Вы можете добавлять и удалять факты напрямую:

```python
agent.add_fact('likes(agent, prolog).')
agent.retract_fact('likes(agent, _).')    # удалить первое подходящее
agent.retract_all('likes(_, _).')          # удалить все
```

### 7.3. Генераторы решений

Действие-генератор позволяет реализовать перебор вариантов. Например, предикат `between/3` можно реализовать так:

```python
@generator_action('between', 3, output_arg=2)
def between_gen(low, high, context):
    for i in range(low, high+1):
        yield i
```

В Prolog это будет работать с возвратами:

```prolog
?- between(1, 5, X), write(X), nl, fail.
```

### 7.4. Обработка ошибок

Если действие должно сигнализировать о неудаче, верните `None` (для `simple_action`) или ничего не возвращайте / бросьте исключение (для `effector_action`). Исключения автоматически приводят к неудаче цели.

## 8. Заключение

Фреймворк PyAgentProlog позволяет создавать мощных, гибких агентов, разделяя логику (Prolog) и действия (Python). Удобные декораторы скрывают сложность интеграции, а встроенные предикаты `set_state/1` и `update_state/1` делают управление состоянием чистым и декларативным. 