**PyAgentProlog** — это библиотека, объединяющая логическое программирование на Prolog с экосистемой LangChain. Модуль `prologchain.py` предоставляет **Runnable**-обёртки, совместимые с LangChain Expression Language (LCEL), что позволяет легко встраивать Prolog-запросы в цепочки обработки данных, включая LLM-пайплайны.

## 1. Установка и зависимости

```bash
pip install langchain-core
```

Скопируйте директорию `prolog/` в ваш проект или установите пакет стандартным способом. Структура должна включать файлы: `prologchain.py`, `interpreter.py`, `parser.py`, `predicates.py`, `solver.py`, `terms.py`, `tracer.py`.

Импортируйте необходимые классы:

```python
from prolog.prologchain import (
    PrologQueryTemplate,
    PrologChain,
    PrologOutputFormatter,
    create_prolog_chain,
    create_prolog_query_template,
    prolog_pipeline
)
```

## 2. Основные компоненты

### 2.1. `PrologQueryTemplate` — шаблон запроса

Аналог `PromptTemplate`, но для Prolog. Переменные заключаются в фигурные скобки:

```python
template = PrologQueryTemplate.from_template("родитель({person}, X).")
# или
template = PrologQueryTemplate("родитель({person}, X).")
```

Метод `invoke` возвращает словарь `{"query": отформатированная_строка}`. Поддерживает оператор `|` для объединения в цепочку.

### 2.2. `PrologChain` — выполнение Prolog-запроса

Принимает программу на Prolog и выполняет запросы. Возвращает структурированный результат.

```python
program = """
родитель(иван, алексей).
родитель(алексей, петр).
"""

prolog = PrologChain(program=program, include_explanation=False)
result = prolog.invoke({"query": "родитель(иван, X)."})
print(result["output"])   # "Решение: X = алексей"
```

Параметры:
- `program` – строка с Prolog-кодом.
- `context` – словарь для передачи внешних объектов (доступен в пользовательских предикатах через `database.context`).
- `include_explanation` – включает трассировку (поле `explanation` в результате).

### 2.3. `PrologOutputFormatter` — форматирование вывода

Преобразует результат `PrologChain` в читаемую строку для LLM:

```python
formatter = PrologOutputFormatter(include_query=True, include_stats=True)
output = formatter.invoke(result)   # возвращает str
```

### 2.4. Фабричные функции

- `create_prolog_chain(program, include_explanation=False)` – создаёт `PrologChain`.
- `create_prolog_query_template(template)` – создаёт `PrologQueryTemplate`.
- `prolog_pipeline(program, query_template=None)` – возвращает готовую цепочку (шаблон + выполнение).

## 3. Построение цепочек с LCEL

Используйте оператор `|` для объединения шаблона и выполнения:

```python
chain = PrologQueryTemplate.from_template("родитель({person}, X).") | PrologChain(program)
response = chain.invoke({"person": "иван"})
```

Можно также использовать `prolog_pipeline`:

```python
chain = prolog_pipeline(program, query_template="родитель({person}, X).")
result = chain.invoke({"person": "иван"})
```

## 4. Примеры интеграции с LLM

### 4.1. Преобразование естественного языка в Prolog-запрос

```python
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4")
prompt = ChatPromptTemplate.from_messages([
    ("human", "Преобразуй вопрос пользователя в Prolog-запрос вида 'родитель(имя, X).': {question}")
])

def extract_query(llm_output):
    return {"query": llm_output.content.strip()}

chain = (
    prompt | llm | RunnableLambda(extract_query) |
    PrologChain(program=program) |
    PrologOutputFormatter()
)

answer = chain.invoke({"question": "Кто внук Ивана?"})
print(answer)
```

### 4.2. Использование в многошаговых цепочках

```python
from langchain_core.runnables import RunnablePassthrough

# Предположим, у нас есть функция, которая выбирает программу на основе входных данных
def select_program(inputs):
    return {"program": inputs.get("domain_program"), "query": inputs.get("user_query")}

chain = (
    RunnablePassthrough.assign(program=select_program) |
    PrologChain(program=lambda x: x["program"]) |
    PrologOutputFormatter()
)
```

## 5. Пользовательские предикаты на Python

Хотя модуль `prologchain` не содержит декораторов для регистрации действий, вы можете зарегистрировать свои предикаты через `PrologChain`:

```python
from prolog.predicates import prolog_predicate

@prolog_predicate("square", 2)
def square(goal, database, subst, depth):
    # реализация...
    pass

chain = PrologChain(program=program)
chain._solver.database.register_custom_predicate("square", 2, square)
```

Более удобно использовать модуль `agent.py`, но это выходит за рамки данной инструкции.

## 6. Отладка и трассировка

- Установите `DEBUG = True` в `interpreter.py` для подробного вывода.
- При создании `PrologChain` укажите `include_explanation=True`. Тогда в результате появится поле `explanation`, которое можно отобразить через `chain._solver.get_explanation()` (если нужен доступ к объекту солвера).
- Используйте `PrologOutputFormatter` для красивого вывода.

---

PyAgentProlog в интеграции с LangChain позволяет строить гибкие пайплайны, где Prolog выступает в роли движка логического вывода, а LLM — интерпретатора естественного языка. Это даёт возможность создавать надёжные системы с объяснимыми решениями.