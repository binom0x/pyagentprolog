#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test agent demonstrating the PyAgentProlog framework using
the simplified decorators @simple_action, @effector_action and
@generator_action, as well as streaming of results.
"""

import sys
sys.path.append('d:/BFX/Repositories/')

from prolog.agent import Agent, simple_action, effector_action, generator_action


# -------------------------------------------------------------------
# Defining agent actions
# -------------------------------------------------------------------

@simple_action('read_input', 1, output_arg=0)
def read_input(context):
    """
    read_input(-Text)
    Reads a line from standard input and returns it.
    """
    line = input(">> ").strip()
    return line


@effector_action('print_message', 1)
def print_message(msg, context):
    """
    print_message(+Message)
    Prints a message to the screen.
    """
    print(f"[Agent]: {msg}")


@simple_action('llm_echo', 2, output_arg=1)
def llm_echo(prompt, context):
    """
    llm_echo(+Prompt, -Response)
    LLM simulation: returns 'Echo: ' + the input text.
    """
    return f"Echo: {prompt}"


# -------------------------------------------------------------------
# Generator action to demonstrate streaming
# -------------------------------------------------------------------
@generator_action('counter', 2, output_arg=1, description="Generates numbers from 1 to N")
def counter(limit, context):
    """
    counter(+Limit, -Value)
    Generates all integers from 1 to Limit inclusive.
    """
    for i in range(1, limit + 1):
        yield i


# -------------------------------------------------------------------
# Prolog agent program (unchanged)
# -------------------------------------------------------------------
AGENT_PROGRAM = """
% Dynamic state predicates
:- dynamic state/2.
:- dynamic memory/1.

% Agent initialization
init :-
    set_state(state(phase, init)),
    set_state(memory([])).

% Main loop step
step :-
    state(phase, Phase),
    handle_phase(Phase, NextPhase),
    set_state(state(phase, NextPhase)),
    continue(NextPhase).

continue(final) :- !.
continue(_) :- step.

% Handling the init phase
handle_phase(init, working) :-
    print_message('Agent initialized. Enter an action:'),
    read_input(Action),
    set_state(memory(Action)),
    print_message('Action remembered.').

% Handling the working phase
handle_phase(working, final) :-
    memory(LastAction),
    print_message('Last action:'),
    print_message(LastAction),
    print_message('Contacting the LLM for analysis...'),
    llm_echo(LastAction, Analysis),
    print_message('LLM responded:'),
    print_message(Analysis),
    print_message('Finishing up.').

handle_phase(final, final) :-
    print_message('The agent has already finished.').

% Entry point
run :- init, step.
"""


# -------------------------------------------------------------------
# Testing the agent
# -------------------------------------------------------------------
def main():
    print("=" * 60)
    print("TEST AGENT PROAGENT (STREAMING)")
    print("=" * 60)

    context = {
        'llm': None,
        'user_name': 'Tester'
    }

    agent = Agent(AGENT_PROGRAM, context=context)
    agent.register_action_module(sys.modules[__name__])

    print("\nRegistered actions:")
    for act in agent.list_actions():
        print(f"  - {act['name']}/{act['arity']}: {act['description']} ({act['type']})")
    print()

    # -----------------------------------------------------------------
    # Demonstrating streaming with the counter/2 generator
    # -----------------------------------------------------------------
    print("Streaming demo: query counter(5, X)")
    stream = agent.stream("counter(5, X).")
    for i, solution in enumerate(stream, 1):
        if isinstance(solution, dict) and 'error' in solution:
            print(f"  Error: {solution['error']}")
            break
        elif isinstance(solution, dict):
            print(f"  Solution {i}: X = {solution.get('X')}")
        else:
            print(f"  Unexpected result: {solution}")
    print()

    # -----------------------------------------------------------------
    # Running the main agent using streaming
    # -----------------------------------------------------------------
    print("Starting the agent (query 'run.') in streaming mode.")
    print("Enter actions when prompted.\n")

    run_stream = agent.stream('run.')
    try:
        for result in run_stream:
            if isinstance(result, dict) and 'error' in result:
                print(f"Error: {result['error']}")
                break
            elif result is True:
                print("[Stream] The agent finished successfully.")
            elif result is False:
                print("[Stream] The agent did not complete (False).")
            else:
                print(f"[Stream] Unexpected result: {result}")
    except Exception as e:
        print(f"Exception during streaming: {e}")

    print("\n" + "=" * 60)

    # -----------------------------------------------------------------
    # Checking the final state through streaming
    # -----------------------------------------------------------------
    print("\nFinal memory contents (memory/1) in streaming mode:")
    mem_stream = agent.stream('memory(X).')
    found = False
    for sol in mem_stream:
        if isinstance(sol, dict) and 'error' in sol:
            print(f"Error: {sol['error']}")
            break
        elif isinstance(sol, dict):
            print(f"  X = {sol.get('X')}")
            found = True
        else:
            print(f"  Unexpected result: {sol}")
    if not found:
        print("  No solutions.")

    print("\nTest completed.")


if __name__ == "__main__":
    main()