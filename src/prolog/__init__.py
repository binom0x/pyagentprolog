"""PyAgentProlog — Prolog-движок с агентами, трассировкой и семантической унификацией.

Prolog-движок на Python с поддержкой:
  * трассировки доказательств                     (tracer);
  * встроенных и мета-предикатов                  (predicates);
  * итеративного солвера с TCO и отсечением       (interpreter, solver);
  * агентов с Python-действиями                   (agent, actions);
  * семантической унификации на эмбеддингах       (embedding_unify);
  * интеграции с LangChain                        (prologchain);
  * Gradio-сессий для агента и эксперта           (proagent_gradio).

Порядок импортов важен: ядро
    terms → bridge → parser → tracer → predicates
          → interpreter → solver → actions → agent
загружается раньше зависимых модулей. Опциональные зависимости
(numpy, langchain_core, gradio, expert, sentence-transformers)
подключаются по мере наличия: если пакет недоступен, соответствующий
модуль просто не импортируется, а ядро продолжает работать.

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)
"""

__version__ = "5.9.1"
__author__ = "Andrey Vorfolomeev"

# --- Ядро (без внешних зависимостей) ---
from .terms import (
    Term, Variable, Number, Rule,
    set_custom_unifier, get_custom_unifier, clear_custom_unifier,
)
from .bridge import prolog_to_python, python_to_prolog, prolog_list_to_python
from .parser import Parser, parse_query_from_string
from .tracer import ProofStep, ProofTracer
from .predicates import (
    Predicate, prolog_predicate, PrologError, CutException, RepeatException,
    evaluate_expression, STANDARD_PREDICATES,
    arith_numba_status, warmup_arith_numba,
)
from .interpreter import Database, BuiltinRegistry, Frame
from .solver import (
    Solver, ParseSolution,
    format_explanation_for_human, format_dict_explanation,
)
from .actions import simple_action, effector_action, generator_action
from .agent import Agent, agent_action

# --- Опциональные модули ---
try:
    from .embedding_unify import (
        EmbeddingUnifier, semantic_on, semantic_off,
        semantic_status, install_embedding_unifier,
    )
except ImportError:
    pass

try:
    from .prologchain import PrologChain, PrologQueryTemplate
except ImportError:
    pass

try:
    from .proagent_gradio import (
        BaseGradioSession,
        AgentGradioSession,
        create_agent_session,
        start_agent,
        send_input_to_agent,
        get_agent_outputs,
        is_session_running,
        stop_session,
        register_gradio_actions,
    )
except ImportError:
    pass
