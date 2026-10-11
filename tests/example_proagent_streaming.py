#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тестовый агент для демонстрации фреймворка PyAgentProlog с использованием
упрощённых декораторов @simple_action, @effector_action и @generator_action,
а также стриминга результатов.
"""

import sys
sys.path.append('d:/BFX/Repositories/')

from prolog.agent import Agent, simple_action, effector_action, generator_action


# -------------------------------------------------------------------
# Определение действий агента
# -------------------------------------------------------------------

@simple_action('read_input', 1, output_arg=0)
def read_input(context):
    """
    read_input(-Text)
    Считывает строку из стандартного ввода и возвращает её.
    """
    line = input(">> ").strip()
    return line


@effector_action('print_message', 1)
def print_message(msg, context):
    """
    print_message(+Message)
    Выводит сообщение на экран.
    """
    print(f"[Агент]: {msg}")


@simple_action('llm_echo', 2, output_arg=1)
def llm_echo(prompt, context):
    """
    llm_echo(+Prompt, -Response)
    Имитация LLM: возвращает 'Эхо: ' + входной текст.
    """
    return f"Эхо: {prompt}"


# -------------------------------------------------------------------
# Действие-генератор для демонстрации стриминга
# -------------------------------------------------------------------
@generator_action('counter', 2, output_arg=1, description="Генерирует числа от 1 до N")
def counter(limit, context):
    """
    counter(+Limit, -Value)
    Генерирует все целые числа от 1 до Limit включительно.
    """
    for i in range(1, limit + 1):
        yield i


# -------------------------------------------------------------------
# Prolog-программа агента (без изменений)
# -------------------------------------------------------------------
AGENT_PROGRAM = """
% Динамические предикаты состояния
:- dynamic state/2.
:- dynamic memory/1.

% Инициализация агента
init :-
    set_state(state(phase, init)),
    set_state(memory([])).

% Основной шаг цикла
step :-
    state(phase, Phase),
    handle_phase(Phase, NextPhase),
    set_state(state(phase, NextPhase)),
    continue(NextPhase).

continue(final) :- !.
continue(_) :- step.

% Обработка фазы init
handle_phase(init, working) :-
    print_message('Агент инициализирован. Введите действие:'),
    read_input(Action),
    set_state(memory(Action)),
    print_message('Действие запомнено.').

% Обработка фазы working
handle_phase(working, final) :-
    memory(LastAction),
    print_message('Последнее действие:'),
    print_message(LastAction),
    print_message('Обращаюсь к LLM для анализа...'),
    llm_echo(LastAction, Analysis),
    print_message('LLM ответила:'),
    print_message(Analysis),
    print_message('Завершаю работу.').

handle_phase(final, final) :-
    print_message('Агент уже завершил работу.').

% Точка входа
run :- init, step.
"""


# -------------------------------------------------------------------
# Тестирование агента
# -------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ТЕСТОВЫЙ АГЕНТ PROAGENT (СТРИМИНГ)")
    print("=" * 60)

    context = {
        'llm': None,
        'user_name': 'Тестер'
    }

    agent = Agent(AGENT_PROGRAM, context=context)
    agent.register_action_module(sys.modules[__name__])

    print("\nЗарегистрированные действия:")
    for act in agent.list_actions():
        print(f"  - {act['name']}/{act['arity']}: {act['description']} ({act['type']})")
    print()

    # -----------------------------------------------------------------
    # Демонстрация стриминга на примере генератора counter/2
    # -----------------------------------------------------------------
    print("Демонстрация стриминга: запрос counter(5, X)")
    stream = agent.stream("counter(5, X).")
    for i, solution in enumerate(stream, 1):
        if isinstance(solution, dict) and 'error' in solution:
            print(f"  Ошибка: {solution['error']}")
            break
        elif isinstance(solution, dict):
            print(f"  Решение {i}: X = {solution.get('X')}")
        else:
            print(f"  Неожиданный результат: {solution}")
    print()

    # -----------------------------------------------------------------
    # Запуск основного агента с использованием стриминга
    # -----------------------------------------------------------------
    print("Запуск агента (запрос 'run.') в режиме стриминга.")
    print("Вводите действия по запросу.\n")

    run_stream = agent.stream('run.')
    try:
        for result in run_stream:
            if isinstance(result, dict) and 'error' in result:
                print(f"Ошибка: {result['error']}")
                break
            elif result is True:
                print("[Стрим] Агент успешно завершил работу.")
            elif result is False:
                print("[Стрим] Агент не выполнился (False).")
            else:
                print(f"[Стрим] Неожиданный результат: {result}")
    except Exception as e:
        print(f"Исключение при стриминге: {e}")

    print("\n" + "=" * 60)

    # -----------------------------------------------------------------
    # Проверка финального состояния через стриминг
    # -----------------------------------------------------------------
    print("\nФинальное содержимое памяти (memory/1) в режиме стриминга:")
    mem_stream = agent.stream('memory(X).')
    found = False
    for sol in mem_stream:
        if isinstance(sol, dict) and 'error' in sol:
            print(f"Ошибка: {sol['error']}")
            break
        elif isinstance(sol, dict):
            print(f"  X = {sol.get('X')}")
            found = True
        else:
            print(f"  Неожиданный результат: {sol}")
    if not found:
        print("  Нет решений.")

    print("\nТест завершён.")


if __name__ == "__main__":
    main()