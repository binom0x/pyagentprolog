#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тестовый агент для демонстрации фреймворка PyAgentProlog с использованием
упрощённых декораторов @simple_action и @effector_action.
"""

import sys
sys.path.append('d:/BFX/Repositories/')

from prolog.agent import Agent, simple_action, effector_action


# -------------------------------------------------------------------
# Определение действий агента с новыми декораторами
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
    # В реальном приложении здесь можно использовать context['llm_client']
    return f"Эхо: {prompt}"


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
    print("ТЕСТОВЫЙ АГЕНТ PROAGENT (УПРОЩЁННЫЕ ДЕЙСТВИЯ)")
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

    print("Запуск агента (запрос 'run.'). Вводите действия по запросу.\n")
    result = agent.query('run.')

    print("\n" + "=" * 60)
    print("Результат выполнения 'run.':", result)

    print("\nФинальное содержимое памяти (memory/1):")
    mem_result = agent.query('memory(X).')
    if 'solutions' in mem_result:
        for sol in mem_result['solutions']:
            print(f"  X = {sol['X']}")
    else:
        print(mem_result)

    print("\nТест завершён.")


if __name__ == "__main__":
    main()