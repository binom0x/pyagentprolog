""" PyAgentProlog

Solver with explanation support and application of semantic unification directives.

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""

from prolog.parser import Parser
from prolog.terms import Term, Variable, Number
from prolog.interpreter import Database
from prolog.tracer import ProofTracer, ProofStep
from prolog.predicates import PrologError
from collections import defaultdict


class Solver(object):
    def __init__(self, rules_text="", context=None):
        """Parse the rules text and initialize the database,
        and apply semantic unification directives."""
        parser = Parser(rules_text)
        rules, directives = parser.parse_rules(return_directives=True)
        self.database = Database(rules, context=context)
        self.last_tracer = None
        self._apply_directives(directives)

    # ------------------------------------------------------------------
    # Applying directives recognized by the parser
    # ------------------------------------------------------------------
    def _apply_directives(self, directives):
        for d in directives:
            try:
                self._apply_one_directive(d)
            except Exception as e:
                print(f"[Solver] Error applying directive {d}: {e}")

    def _apply_one_directive(self, d: Term):
        name = d.functor
        args = d.arguments

        if name == 'semantic_on':
            from prolog.embedding_unify import semantic_on
            if len(args) == 0:
                semantic_on()
            elif len(args) == 1:
                t = args[0]
                if isinstance(t, Number):
                    semantic_on(threshold=float(t.value))
                else:
                    raise ValueError(f"semantic_on/1 expects a number, got {t}")
            else:
                raise ValueError("semantic_on takes 0 or 1 argument")

        elif name == 'semantic_off':
            from prolog.embedding_unify import semantic_off
            semantic_off()

        elif name == 'semantic_threshold':
            if len(args) != 1 or not isinstance(args[0], Number):
                raise ValueError("semantic_threshold/1 expects a number")
            from prolog.embedding_unify import semantic_set_threshold, semantic_get_unifier
            if semantic_get_unifier() is None:
                # If the unifier is not yet enabled — do not fail, just do nothing
                return
            semantic_set_threshold(float(args[0].value))

    # ------------------------------------------------------------------
    # Searching for solutions
    # ------------------------------------------------------------------
    def find_solutions(self, query_text, with_explanation=False):
        """Parse the query text and find solutions"""
        try:
            query = Parser(query_text).parse_query()
        except Exception as e:
            print(f"Query parse error: {e}")
            return None

        try:
            if with_explanation:
                matching_query_terms, self.last_tracer = self.database.query_with_explanation(query)
            else:
                matching_query_terms = list(self.database.query(query))
        except PrologError as e:
            return {"error": str(e), "type": e.error_type, "detail": e.detail}

        query_variable_map = {}
        variables_in_query = False

        def find_variables(term):
            vars_list = []
            if isinstance(term, Variable):
                if term.name != "_":
                    vars_list.append(term)
            elif isinstance(term, Term):
                for arg in term.arguments:
                    vars_list.extend(find_variables(arg))
            return vars_list

        variables = find_variables(query)
        for var in variables:
            variables_in_query = True
            query_variable_map[var.name] = var

        if matching_query_terms:
            if query_variable_map:
                solutions_list = []
                for matching_query_term in matching_query_terms:
                    subst = {}

                    def unify_and_collect(query_term, result_term, subst):
                        if isinstance(query_term, Variable):
                            if query_term.name != "_":
                                subst[query_term] = result_term
                        elif isinstance(query_term, Term) and isinstance(result_term, Term):
                            if len(query_term.arguments) == len(result_term.arguments):
                                for q_arg, r_arg in zip(query_term.arguments, result_term.arguments):
                                    unify_and_collect(q_arg, r_arg, subst)

                    unify_and_collect(query, matching_query_term, subst)

                    sol_dict = {}
                    for var, val in subst.items():
                        if isinstance(val, Variable):
                            continue
                        sol_dict[var.name] = val
                    solutions_list.append(sol_dict)

                if with_explanation:
                    return solutions_list, self.last_tracer
                return solutions_list
            else:
                if matching_query_terms:
                    if with_explanation:
                        return True, self.last_tracer
                    return True
                else:
                    if with_explanation:
                        return False, self.last_tracer
                    return False
        else:
            if with_explanation:
                return False, self.last_tracer
            return False

    def get_explanation(self, format="text"):
        """Get the explanation of the last query"""
        if not self.last_tracer:
            return "No tracing data. Run a query with with_explanation=True"

        if format == "text":
            return self.last_tracer.get_proof_tree()
        elif format == "dict":
            return self.last_tracer.get_proof_dict()
        elif format == "tree":
            return self.last_tracer.root_steps
        else:
            return self.last_tracer.get_proof_tree()


def ParseSolution(query, solution, explanation=None):
    """Format the solution for output"""
    def term_to_string(term: Term):
        term_str = str(term)
        return term_str

    def is_unbound(var_name, value):
        return isinstance(value, Variable) and value.name == var_name

    result = {"query": query}

    if solution is True:
        result["solution"] = True
        result["output"] = "True"
    elif solution is False:
        result["solution"] = False
        result["output"] = "False"
    elif solution is None:
        result["solution"] = None
        result["output"] = "No solution"
    elif isinstance(solution, list):
        if not solution:
            result["solution"] = None
            result["output"] = "No solution"
        else:
            solutions_list = []
            outputs = []
            for sol_dict in solution:
                if isinstance(sol_dict, dict):
                    bound_dict = {}
                    out_parts = []
                    for var_name, value in sol_dict.items():
                        if not is_unbound(var_name, value):
                            val_str = term_to_string(value) if isinstance(value, Term) else str(value)
                            bound_dict[var_name] = val_str
                            out_parts.append(f"{var_name} = {val_str}")
                    if not bound_dict:
                        bound_dict = None
                        out_parts = ["true"]
                    solutions_list.append(bound_dict)
                    outputs.append(", ".join(out_parts))
                else:
                    solutions_list.append(sol_dict)
                    outputs.append(str(sol_dict))
            if all(s is None for s in solutions_list):
                result["solution"] = True
                result["output"] = "True"
            else:
                solutions_list = [s for s in solutions_list if s is not None]
                result["solution"] = solutions_list
                result["output"] = outputs if len(outputs) > 1 else outputs[0]
                result["count"] = len(solutions_list)
    elif isinstance(solution, dict):
        if not solution:
            result["solution"] = None
            result["output"] = "No solution"
        else:
            bound_dict = {}
            out_parts = []
            for var_name, val_list in solution.items():
                if val_list:
                    val = val_list[0]
                    if not is_unbound(var_name, val):
                        val_str = term_to_string(val) if isinstance(val, Term) else str(val)
                        bound_dict[var_name] = val_str
                        out_parts.append(f"{var_name} = {val_str}")
            if not bound_dict:
                result["solution"] = True
                result["output"] = "True"
            else:
                result["solution"] = bound_dict
                result["output"] = ", ".join(out_parts)
                result["count"] = 1

    if explanation:
        if isinstance(explanation, ProofTracer):
            result["explanation"] = explanation.get_proof_dict()
        elif isinstance(explanation, dict):
            result["explanation"] = explanation
        else:
            result["explanation"] = str(explanation)

    return result


def format_explanation_for_human(explanation):
    """Format an explanation for human reading"""
    if isinstance(explanation, dict):
        return format_dict_explanation(explanation)
    elif hasattr(explanation, 'get_proof_tree'):
        return explanation.get_proof_tree()
    else:
        return str(explanation)


def format_dict_explanation(explanation_dict):
    """Format a dict with an explanation"""
    result = "=" * 60 + "\n"
    result += "QUERY EXECUTION EXPLANATION\n"
    result += "=" * 60 + "\n\n"

    if 'steps' in explanation_dict:
        for step in explanation_dict['steps']:
            result += format_step(step, 0) + "\n"

    result += "\n" + "=" * 60
    return result


def format_step(step_dict, indent):
    """Format a single step"""
    indent_str = "  " * indent
    result = f"{indent_str}• {step_dict['goal']}"

    if step_dict['rule_used']:
        result += f"\n{indent_str}  Rule: {step_dict['rule_used']}"

    if step_dict['unification']:
        unif_str = ", ".join([f"{var} = {val}" for var, val in step_dict['unification']])
        result += f"\n{indent_str}  Unification: {unif_str}"

    if step_dict['builtin']:
        result += f"\n{indent_str}  [Built-in predicate]"

    result += f"\n{indent_str}  Result: {'✓ SUCCESS' if step_dict['success'] else '✗ FAILURE'}"

    for child in step_dict['children']:
        result += "\n" + format_step(child, indent + 1)

    return result


if __name__ == "__main__":
    print("Testing the solver with explanations:")
    print("=" * 50)

    program = """
    parent(ivan, alexey).
    parent(ivan, maria).
    parent(alexey, peter).
    parent(alexey, anna).

    grandfather(Grandfather, Grandson) :-
        parent(Grandfather, Parent),
        parent(Parent, Grandson).

    test_sum(A, B, C) :- C is A + B.
    """

    solver = Solver(program)

    print("Loaded program:")
    print("=" * 50)
    for rule in solver.database.rules:
        print(rule)
    print("=" * 50)
    print()

    query = "grandfather(ivan, X)."
    print(f"Query: {query}")

    solutions, explanation = solver.find_solutions(query, with_explanation=True)

    print("RAW RESULT: ", solutions)

    parsed_result = ParseSolution(query, solutions, explanation)
    print(f"Result: {parsed_result['solution']}")
    if 'count' in parsed_result:
        print(f"Number of solutions: {parsed_result['count']}")

    print("\n" + "=" * 50)
    print("EXPLANATION:")
    print(solver.get_explanation())
    print("=" * 50)

    print("\nFormatted explanation:")
    print(format_explanation_for_human(explanation))