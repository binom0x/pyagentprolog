""" PyAgentProlog

Gradio sessions for the agent (without Expert — Expert moved to examples/expert_app/).

(c) 2026 Andrey Vorfolomeev
version 1.0.1
"""

import threading
import queue
import uuid
import re
from typing import Optional, List, Tuple, Any, Dict

from prolog.agent import Agent, effector_action, simple_action


# ----------------------------------------------------------------------
# BASE SESSION CLASS
# ----------------------------------------------------------------------
class BaseGradioSession:
    def __init__(self):
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        with self._lock:
            return self.running

    def stop(self):
        with self._lock:
            self.running = False
        self.input_queue.put(None)

    def send_input(self, text: str):
        self.input_queue.put(text)

    def get_output(self) -> List[Any]:
        msgs = []
        while not self.output_queue.empty():
            try:
                msgs.append(self.output_queue.get_nowait())
            except queue.Empty:
                break
        return msgs

    def add_output(self, msg: Any):
        self.output_queue.put(msg)


# ----------------------------------------------------------------------
# SESSION FOR THE AGENT
# ----------------------------------------------------------------------
class AgentGradioSession(BaseGradioSession):
    def __init__(self):
        super().__init__()
        self.history = []
        self.ui_values: Dict[str, Any] = {}

    def start(self, agent: Agent, query: str = "run."):
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._run, args=(agent, query), daemon=True)
        self.thread.start()

    def _run(self, agent: Agent, query: str):
        try:
            if agent.context is None:
                agent.context = {}
            agent.context['agent_session'] = self
            result = agent.query(query)
            self.add_output(f"Agent finished execution. Result: {result}")
        except Exception as e:
            self.add_output(f"Error in agent: {e}")
        finally:
            self.running = False
            self.thread = None

    def get_input(self) -> str:
        return self.input_queue.get()

    def get_ui_value(self, name: str) -> Any:
        with self._lock:
            return self.ui_values.get(name)

    def set_ui_value(self, name: str, value: Any):
        with self._lock:
            self.ui_values[name] = value


# ----------------------------------------------------------------------
# SESSION REGISTRY (public API — the application registers its own sessions)
# ----------------------------------------------------------------------
_sessions: Dict[str, BaseGradioSession] = {}
_sessions_lock = threading.Lock()


def register_session(session_id: str, session: BaseGradioSession) -> None:
    with _sessions_lock:
        _sessions[session_id] = session


def get_session(session_id: str) -> Optional[BaseGradioSession]:
    with _sessions_lock:
        return _sessions.get(session_id)


def remove_session(session_id: str) -> None:
    with _sessions_lock:
        _sessions.pop(session_id, None)


# ----------------------------------------------------------------------
# FUNCTIONS FOR WORKING WITH AGENT SESSIONS
# ----------------------------------------------------------------------
def create_agent_session() -> str:
    session_id = str(uuid.uuid4())
    register_session(session_id, AgentGradioSession())
    return session_id


def start_agent(session_id: str, agent: Agent, query: str = "run.") -> None:
    sess = get_session(session_id)
    if sess and isinstance(sess, AgentGradioSession):
        sess.start(agent, query)


def send_input_to_agent(session_id: str, text: str) -> None:
    sess = get_session(session_id)
    if sess and isinstance(sess, AgentGradioSession):
        sess.send_input(text)


def get_agent_outputs(session_id: str) -> List[str]:
    sess = get_session(session_id)
    if sess and isinstance(sess, AgentGradioSession):
        return sess.get_output()
    return []


# ----------------------------------------------------------------------
# COMMON FUNCTIONS
# ----------------------------------------------------------------------
def is_session_running(session_id: str) -> bool:
    sess = get_session(session_id)
    return sess.is_running() if sess else False


def stop_session(session_id: str) -> None:
    sess = get_session(session_id)
    if sess:
        sess.stop()
        remove_session(session_id)


# ----------------------------------------------------------------------
# AGENT ACTIONS FOR INTERACTING WITH THE UI
# ----------------------------------------------------------------------
@effector_action('gradio_print', 1)
def gradio_print(msg, context):
    session = context.get('agent_session')
    if session and isinstance(session, AgentGradioSession):
        session.add_output(msg)
    else:
        print("gradio_print: session not found")


@simple_action('gradio_read_input', 1, output_arg=0)
def gradio_read_input(context):
    session = context.get('agent_session')
    if session and isinstance(session, AgentGradioSession):
        return session.get_input()
    return ""


@simple_action('gradio_get_ui_value', 2, output_arg=1,
               description="Retrieves the value of a UI component by name")
def gradio_get_ui_value(name, context):
    session = context.get('agent_session')
    if session and isinstance(session, AgentGradioSession):
        clean_name = name.strip("'\"") if isinstance(name, str) else name
        value = session.get_ui_value(clean_name)
        if value is None:
            return []
        if isinstance(value, bool):
            return "true" if value else "false"
        return value
    return []


@effector_action('gradio_set_ui_value', 2)
def gradio_set_ui_value(name, value, context):
    session = context.get('agent_session')
    if session and isinstance(session, AgentGradioSession):
        session.set_ui_value(name, value)


def prolog_format_to_python(fmt: str) -> str:
    fmt = fmt.replace('~~', '\0')
    fmt = re.sub(r'~w', '{}', fmt)
    fmt = re.sub(r'~d', '{:d}', fmt)
    fmt = re.sub(r'~f', '{:f}', fmt)
    fmt = re.sub(r'~s', '{:s}', fmt)
    fmt = fmt.replace('\0', '~')
    return fmt


@simple_action('format', 3, output_arg=2,
               description="format(+Format, +Args, -Result) – formats a string.")
def format_action(fmt, args, context):
    if not isinstance(args, list):
        args = [args]
    python_fmt = prolog_format_to_python(fmt)
    try:
        return python_fmt.format(*args)
    except Exception as e:
        return f"<format error: {e}>"


def register_gradio_actions(agent: Agent) -> None:
    agent.register_action(gradio_print)
    agent.register_action(gradio_read_input)
    agent.register_action(gradio_get_ui_value)
    agent.register_action(gradio_set_ui_value)
    agent.register_action(format_action)


__all__ = [
    'BaseGradioSession',
    'AgentGradioSession',
    'register_session',
    'get_session',
    'remove_session',
    'create_agent_session',
    'start_agent',
    'send_input_to_agent',
    'get_agent_outputs',
    'is_session_running',
    'stop_session',
    'gradio_print',
    'gradio_read_input',
    'gradio_get_ui_value',
    'gradio_set_ui_value',
    'format_action',
    'register_gradio_actions',
]