import sys
sys.path.append('d:/BFX/Repositories/')
import gradio as gr
from prolog.agent import Agent, simple_action
from prolog.proagent_gradio import (
    create_session, get_session,
    gradio_print, gradio_read_input, gradio_get_ui_value,
    format_action
)

AGENT_PROGRAM = """
:- dynamic state/2.

init :- set_state(state(phase, loop)).

step :-
    state(phase, Phase),
    handle_phase(Phase, NextPhase),
    set_state(state(phase, NextPhase)),
    continue(NextPhase).

continue(final) :- !.
continue(_) :- step.

handle_phase(loop, NextPhase) :-
    gradio_print('Enter an action (or "stop" to exit):'),
    gradio_read_input(Action),
    process_action(Action, NextPhase).

process_action("stop", final) :- !.
process_action(Action, loop) :-
    gradio_get_ui_value('repeat_check', Repeat),
    gradio_get_ui_value('mode_select', Mode),
    gradio_get_ui_value('volume_slider', Volume),
    format('Mode: ~w, Volume: ~w, Repeat: ~w', [Mode, Volume, Repeat], Message),
    gradio_print(Message),
    gradio_print('Contacting the LLM for analysis...'),
    llm_echo(Action, Analysis),
    gradio_print('LLM responded:'),
    gradio_print(Analysis),
    handle_repeat(Repeat, Analysis).

handle_repeat("true", Analysis) :-
    gradio_print('(Repeat is enabled, duplicating)'),
    gradio_print(Analysis),
    !.
handle_repeat(_, _).

handle_phase(final, final) :-
    gradio_print('The agent has finished.').

run :- init, step.
"""

@simple_action('llm_echo', 2, output_arg=1)
def llm_echo(prompt, context):
    return f"Echo: {prompt}"


def create_interface():
    with gr.Blocks(title="Prolog Agent with UI configuration") as demo:
        gr.Markdown("# Agent controlled by UI elements")

        with gr.Row():
            with gr.Column(scale=2):
                log = gr.Textbox(label="Dialogue", lines=20, interactive=False, autoscroll=True)
            with gr.Column(scale=1):
                gr.Markdown("### Agent settings")
                repeat_check = gr.Checkbox(label="Repeat the response", value=False)
                mode_select = gr.Dropdown(choices=["gentle", "rude", "neutral"],
                                          label="Mode", value="neutral")
                volume_slider = gr.Slider(minimum=0, maximum=10, step=1,
                                          label="Volume", value=5)

        with gr.Row():
            start_btn = gr.Button("Start the agent", variant="primary")
            user_input = gr.Textbox(label="Input", placeholder="Type a message...", interactive=False)
            send_btn = gr.Button("Send", interactive=False, variant="secondary")

        session_id_state = gr.State()
        timer = gr.Number(value=0, visible=False, every=1)

        def start_agent(session_id, repeat, mode, volume):
            if session_id:
                session = get_session(session_id)
                if session and session.running:
                    return session_id, gr.update(), gr.update()

            session_id = create_session()
            session = get_session(session_id)
            session.set_ui_value('repeat_check', repeat)
            session.set_ui_value('mode_select', mode)
            session.set_ui_value('volume_slider', volume)
            print(f"start_agent: created session {session_id}, id={id(session)}, setting values: repeat={repeat}, mode={mode}, volume={volume}")

            agent = Agent(AGENT_PROGRAM, context={'agent_session': session})
            agent.register_action(gradio_print)
            agent.register_action(gradio_read_input)
            agent.register_action(gradio_get_ui_value)
            agent.register_action(format_action)
            agent.register_action_module(sys.modules[__name__])

            session.start(agent, query="run.")
            return session_id, gr.update(interactive=False), gr.update(interactive=False)

        def send_message(session_id, msg, repeat, mode, volume):
            if not session_id:
                return session_id, "", gr.update()
            session = get_session(session_id)
            if session:
                print(f"send_message: session {session_id}, id={id(session)}, setting values before sending: repeat={repeat}, mode={mode}, volume={volume}")
                session.set_ui_value('repeat_check', repeat)
                session.set_ui_value('mode_select', mode)
                session.set_ui_value('volume_slider', volume)
                print(f"After setting: repeat={session.get_ui_value('repeat_check')}, mode={session.get_ui_value('mode_select')}, volume={session.get_ui_value('volume_slider')}")
                if session.waiting:
                    session.send_input(msg)
                    return session_id, "", gr.update(interactive=False)
            return session_id, "", gr.update()

        def update_ui(session_id, repeat, mode, volume):
            if not session_id:
                return "", gr.update(interactive=False), gr.update(interactive=False)
            session = get_session(session_id)
            if session is None:
                return "", gr.update(interactive=False), gr.update(interactive=False)

            session.set_ui_value('repeat_check', repeat)
            session.set_ui_value('mode_select', mode)
            session.set_ui_value('volume_slider', volume)

            new_msgs = session.get_pending_outputs()
            if new_msgs:
                session.history.extend(new_msgs)

            log_text = "\n".join(session.history)
            input_interactive = session.waiting and session.running
            send_interactive = input_interactive

            return log_text, gr.update(interactive=input_interactive), gr.update(interactive=send_interactive)

        start_btn.click(
            start_agent,
            inputs=[session_id_state, repeat_check, mode_select, volume_slider],
            outputs=[session_id_state, user_input, send_btn]
        ).then(
            update_ui,
            inputs=[session_id_state, repeat_check, mode_select, volume_slider],
            outputs=[log, user_input, send_btn]
        )

        send_btn.click(
            send_message,
            inputs=[session_id_state, user_input, repeat_check, mode_select, volume_slider],
            outputs=[session_id_state, user_input, send_btn]
        ).then(
            update_ui,
            inputs=[session_id_state, repeat_check, mode_select, volume_slider],
            outputs=[log, user_input, send_btn]
        )

        timer.change(
            update_ui,
            inputs=[session_id_state, repeat_check, mode_select, volume_slider],
            outputs=[log, user_input, send_btn]
        )

        demo.load(
            lambda: (None, "", gr.update(interactive=False), gr.update(interactive=False)),
            inputs=[],
            outputs=[session_id_state, log, user_input, send_btn]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(inbrowser=True, debug=True)