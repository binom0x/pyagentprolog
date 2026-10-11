""" PyAgentProlog

Солвер с поддержкой объяснений и применением директив семантической унификации.

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""

from prolog.parser import Parser
from prolog.terms import Term, Variable, Number
from prolog.interpreter import Database
from prolog.tracer import ProofTracer, ProofStep
from prolog.predicates import PrologError
from collections import defaultdict


class Solver(object):
    def __init__(self, rules_text="", context=None):
        """Проанализировать текст правил и инициализировать базу данных,
        а также применить директивы семантической унификации."""
        parser = Parser(rules_text)
        rules, directives = parser.parse_rules(return_directives=True)
        self.database = Database(rules, context=context)
        self.last_tracer = None
        self._apply_directives(directives)

    # ------------------------------------------------------------------
    # Применение директив, распознанных парсером
    # ------------------------------------------------------------------
    def _apply_directives(self, directives):
        for d in directives:
            try:
                self._apply_one_directive(d)
            except Exception as e:
                print(f"[Solver] Ошибка применения директивы {d}: {e}")

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
                    raise ValueError(f"semantic_on/1 ожидает число, получено {t}")
            else:
                raise ValueError("semantic_on принимает 0 или 1 аргумент")

        elif name == 'semantic_off':
            from prolog.embedding_unify import semantic_off
            semantic_off()

        elif name == 'semantic_threshold':
            if len(args) != 1 or not isinstance(args[0], Number):
                raise ValueError("semantic_threshold/1 ожидает число")
            from prolog.embedding_unify import semantic_set_threshold, semantic_get_unifier
            if semantic_get_unifier() is None:
                # Если унификатор ещё не включён — не падаем, просто ничего не делаем
                return
            semantic_set_threshold(float(args[0].value))

    # ------------------------------------------------------------------
    # Поиск решений
    # ------------------------------------------------------------------
    def find_solutions(self, query_text, with_explanation=False):
        """Проанализировать текст запроса и найти решения"""
        try:
            query = Parser(query_text).parse_query()
        except Exception as e:
            print(f"Ошибка парсинга запроса: {e}")
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
        """Получить объяснение последнего запроса"""
        if not self.last_tracer:
            return "Нет данных трассировки. Выполните запрос с with_explanation=True"

        if format == "text":
            return self.last_tracer.get_proof_tree()
        elif format == "dict":
            return self.last_tracer.get_proof_dict()
        elif format == "tree":
            return self.last_tracer.root_steps
        else:
            return self.last_tracer.get_proof_tree()


def ParseSolution(query, solution, explanation=None):
    """Форматирование решения для вывода"""
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
        result["output"] = "Нет решения"
    elif isinstance(solution, list):
        if not solution:
            result["solution"] = None
            result["output"] = "Нет решения"
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
            result["output"] = "Нет решения"
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
    """Форматирование объяснения для человеческого чтения"""
    if isinstance(explanation, dict):
        return format_dict_explanation(explanation)
    elif hasattr(explanation, 'get_proof_tree'):
        return explanation.get_proof_tree()
    else:
        return str(explanation)


def format_dict_explanation(explanation_dict):
    """Форматирование словаря с объяснением"""
    result = "=" * 60 + "\n"
    result += "ОБЪЯСНЕНИЕ ВЫПОЛНЕНИЯ ЗАПРОСА\n"
    result += "=" * 60 + "\n\n"

    if 'steps' in explanation_dict:
        for step in explanation_dict['steps']:
            result += format_step(step, 0) + "\n"

    result += "\n" + "=" * 60
    return result


def format_step(step_dict, indent):
    """Форматирование одного шага"""
    indent_str = "  " * indent
    result = f"{indent_str}• {step_dict['goal']}"

    if step_dict['rule_used']:
        result += f"\n{indent_str}  Правило: {step_dict['rule_used']}"

    if step_dict['unification']:
        unif_str = ", ".join([f"{var} = {val}" for var, val in step_dict['unification']])
        result += f"\n{indent_str}  Унификация: {unif_str}"

    if step_dict['builtin']:
        result += f"\n{indent_str}  [Встроенный предикат]"

    result += f"\n{indent_str}  Результат: {'✓ УСПЕХ' if step_dict['success'] else '✗ НЕУДАЧА'}"

    for child in step_dict['children']:
        result += "\n" + format_step(child, indent + 1)

    return result


if __name__ == "__main__":
    print("Тестирование солвера с объяснениями:")
    print("=" * 50)

    program = """
    родитель(иван, алексей).
    родитель(иван, мария).
    родитель(алексей, петр).
    родитель(алексей, анна).

    дедушка(Дедушка, Внук) :-
        родитель(Дедушка, Родитель),
        родитель(Родитель, Внук).

    тест_суммы(A, B, C) :- C is A + B.
    """

    solver = Solver(program)

    print("Загруженная программа:")
    print("=" * 50)
    for rule in solver.database.rules:
        print(rule)
    print("=" * 50)
    print()

    query = "дедушка(иван, X)."
    print(f"Запрос: {query}")

    solutions, explanation = solver.find_solutions(query, with_explanation=True)

    print("RAW RESULT: ", solutions)

    parsed_result = ParseSolution(query, solutions, explanation)
    print(f"Результат: {parsed_result['solution']}")
    if 'count' in parsed_result:
        print(f"Количество решений: {parsed_result['count']}")

    print("\n" + "=" * 50)
    print("ОБЪЯСНЕНИЕ:")
    print(solver.get_explanation())
    print("=" * 50)

    print("\nФорматированное объяснение:")
    print(format_explanation_for_human(explanation))