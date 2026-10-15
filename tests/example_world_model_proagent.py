"""
Example agent for controlling lighting (agent-oriented approach).
"""
import sys
sys.path.append('d:/BFX/Repositories/')
from prolog.agent import Agent, simple_action, effector_action, generator_action

# ----------------------------------------------------------------------
# Python actions
# ----------------------------------------------------------------------
@effector_action('physical_turn_light', 2,
                 description="Simulates physically turning a light on/off")
def physical_turn_light(room, state, context=None):
    print(f"[PHYSICAL] Light in room '{room}' switched to state '{state}'")

# ----------------------------------------------------------------------
# Prolog program (world model)
# ----------------------------------------------------------------------
PROLOG_PROGRAM = """
% Single fact storing the state of all rooms
lights([(kitchen, off), (living_room, on), (bedroom, off)]).

% Get the state of a specific room
light(Room, State) :-
    lights(L),
    member((Room, State), L).

% Update the state of a room (build a new list and replace the lights/1 fact)
set_light(Room, NewState) :-
    lights(OldL),
    update_list(OldL, Room, NewState, NewL),
    set_state(lights(NewL)),          % atomic replacement of the whole state
    physical_turn_light(Room, NewState).

% Helper predicate: replace the pair for a room in the list
update_list([], _, _, []) :- fail.
update_list([(R,_)|T], R, NewState, [(R,NewState)|T]) :- !.
update_list([H|T], R, NewState, [H|NewT]) :-
    update_list(T, R, NewState, NewT).

% Convenient wrappers
turn_on(Room) :- set_light(Room, on).
turn_off(Room) :- set_light(Room, off).

toggle(Room) :-
    light(Room, State),
    (   State = on -> turn_off(Room)
    ;   State = off -> turn_on(Room)
    ).
"""

# ----------------------------------------------------------------------
# Initialization and execution
# ----------------------------------------------------------------------
def main():
    agent = Agent(program=PROLOG_PROGRAM, context={})
    agent.register_action(physical_turn_light)

    print("=== Current light state ===")
    result = agent.query("light(Room, State).")
    print(result)

    print("\n=== Turning on the kitchen light ===")
    success = agent.query_one("set_light(kitchen, on).")
    print(f"  set_light executed: {success is not None}")

    print("\n=== Checking the kitchen state ===")
    state = agent.query_one("light(kitchen, State).")
    if state:
        print(f"  Kitchen: {state['State']}")

    print("\n=== Toggling the bedroom light ===")
    success = agent.query_one("toggle(bedroom).")
    print(f"  toggle executed: {success is not None}")

    print("\n=== New bedroom state ===")
    state = agent.query_one("light(bedroom, State).")
    if state:
        print(f"  Bedroom: {state['State']}")
    else:
        print("  Fact about the bedroom not found! (which is odd, it should be there)")

    print("\n=== All rooms after changes ===")
    for solution in agent.stream("light(Room, State)."):
        print(f"  {solution['Room']}: {solution['State']}")

    print("\n=== Full state (fact lights/1) ===")
    full = agent.query_one("lights(L).")
    if full:
        print(f"  {full['L']}")

if __name__ == "__main__":
    main()