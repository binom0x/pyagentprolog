"""
Пример агента для управления освещением (агент-ориентированный подход).
"""
import sys
sys.path.append('d:/BFX/Repositories/')
from prolog.agent import Agent, simple_action, effector_action, generator_action

# ----------------------------------------------------------------------
# Python-действия
# ----------------------------------------------------------------------
@effector_action('physical_turn_light', 2,
                 description="Имитирует физическое включение/выключение света")
def physical_turn_light(room, state, context=None):
    print(f"[PHYSICAL] Свет в комнате '{room}' переключён в состояние '{state}'")

# ----------------------------------------------------------------------
# Программа на Prolog (картина мира)
# ----------------------------------------------------------------------
PROLOG_PROGRAM = """
% Единственный факт, хранящий состояние всех комнат
lights([(kitchen, off), (living_room, on), (bedroom, off)]).

% Получить состояние конкретной комнаты
light(Room, State) :-
    lights(L),
    member((Room, State), L).

% Обновить состояние комнаты (создаём новый список и заменяем факт lights/1)
set_light(Room, NewState) :-
    lights(OldL),
    update_list(OldL, Room, NewState, NewL),
    set_state(lights(NewL)),          % атомарная замена всего состояния
    physical_turn_light(Room, NewState).

% Вспомогательный предикат: заменить в списке пару для комнаты
update_list([], _, _, []) :- fail.
update_list([(R,_)|T], R, NewState, [(R,NewState)|T]) :- !.
update_list([H|T], R, NewState, [H|NewT]) :-
    update_list(T, R, NewState, NewT).

% Удобные обёртки
turn_on(Room) :- set_light(Room, on).
turn_off(Room) :- set_light(Room, off).

toggle(Room) :-
    light(Room, State),
    (   State = on -> turn_off(Room)
    ;   State = off -> turn_on(Room)
    ).
"""

# ----------------------------------------------------------------------
# Инициализация и выполнение
# ----------------------------------------------------------------------
def main():
    agent = Agent(program=PROLOG_PROGRAM, context={})
    agent.register_action(physical_turn_light)

    print("=== Текущее состояние света ===")
    result = agent.query("light(Room, State).")
    print(result)

    print("\n=== Включаем свет на кухне ===")
    success = agent.query_one("set_light(kitchen, on).")
    print(f"  set_light выполнено: {success is not None}")

    print("\n=== Проверяем состояние кухни ===")
    state = agent.query_one("light(kitchen, State).")
    if state:
        print(f"  Кухня: {state['State']}")

    print("\n=== Переключаем свет в спальне ===")
    success = agent.query_one("toggle(bedroom).")
    print(f"  toggle выполнено: {success is not None}")

    print("\n=== Новое состояние спальни ===")
    state = agent.query_one("light(bedroom, State).")
    if state:
        print(f"  Спальня: {state['State']}")
    else:
        print("  Факт о спальне не найден! (но это странно, он должен быть)")

    print("\n=== Все комнаты после изменений ===")
    for solution in agent.stream("light(Room, State)."):
        print(f"  {solution['Room']}: {solution['State']}")

    print("\n=== Полное состояние (факт lights/1) ===")
    full = agent.query_one("lights(L).")
    if full:
        print(f"  {full['L']}")

if __name__ == "__main__":
    main()