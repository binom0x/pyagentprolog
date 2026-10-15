#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test agent demonstrating the PyAgentProlog framework using
the simplified decorators @simple_action and @effector_action.
"""

import sys
sys.path.append('d:/BFX/Repositories/')

from prolog.agent import Agent, simple_action, effector_action


# -------------------------------------------------------------------
# Defining agent actions with the new decorators
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
    # In a real application, context['llm_client'] could be used here
    return f"Echo: {prompt}"


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
    print("TEST AGENT PROAGENT (SIMPLIFIED ACTIONS)")
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

    print("Starting the agent (query 'run.'). Enter actions when prompted.\n")
    result = agent.query('run.')

    print("\n" + "=" * 60)
    print("Result of 'run.':", result)

    print("\nFinal memory contents (memory/1):")
    mem_result = agent.query('memory(X).')
    if 'solutions' in mem_result:
        for sol in mem_result['solutions']:
            print(f"  X = {sol['X']}")
    else:
        print(mem_result)

    print("\nTest completed.")


if __name__ == "__main__":
    main()