"""PyAgentProlog — Prolog engine with agents, tracing, and semantic unification.

Prolog engine in Python with support for:
  * proof tracing                          (tracer);
  * built-in and meta-predicates           (predicates);
  * iterative solver with TCO and cut      (interpreter, solver);
  * agents with Python actions             (agent, actions);
  * semantic unification on embeddings     (embedding_unify);
  * LangChain integration                  (prologchain);
  * Gradio sessions for the agent          (proagent_gradio).

Import order matters: the core
    terms → bridge → parser → tracer → predicates
          → interpreter → solver → actions → agent
is loaded before dependent modules. Optional dependencies
(numpy, langchain_core, gradio, expert, sentence-transformers)
are connected when available: if a package is unavailable,
the corresponding module is simply not imported, and the core
continues to work.

(c) 2026 Andrey Vorfolomeev
"""

__version__ = "1.0.1"
__author__ = "Andrey Vorfolomeev"

# --- Core (no external dependencies) ---
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

# --- Optional modules ---
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