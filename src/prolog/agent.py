""" PyAgentProlog

Фреймворк для создания ИИ-агентов на PyAgentProlog с действиями на Python.

Канонические реализации вынесены:
    - prolog_to_python, python_to_prolog, prolog_list_to_python
      → prolog.bridge
    - simple_action, effector_action, generator_action
      → prolog.actions

Этот модуль реэкспортирует их для обратной совместимости и
предоставляет класс Agent и его встроенные предикаты set_state/1,
update_state/1.

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1 (canonical bridge/actions)
"""

import pickle
import gzip
import functools
import sys
import ast
from typing import Dict, Any, Optional, Union, List, Tuple, Iterator

from prolog.solver import Solver
from prolog.parser import Parser
from prolog.terms import Term, Variable, Number, Rule
from prolog.predicates import prolog_predicate

# Канонические реализации — в prolog.bridge / prolog.actions.
# Реэкспорт для обратной совместимости:
#   from prolog.agent import prolog_to_python  → prolog.bridge.prolog_to_python
#   from prolog.agent import simple_action     → prolog.actions.simple_action
from prolog.bridge import (
    prolog_to_python,
    python_to_prolog,
    prolog_list_to_python,
)
from prolog.actions import (
    simple_action,
    effector_action,
    generator_action,
)

DEBUG = False


# ------------------------------------------------------------
# Legacy @agent_action decorator (обратная совместимость)
# ------------------------------------------------------------
def agent_action(name: str, arity: int, description: str = None,
                 action_type: str = "generic"):
    """Legacy-декоратор для регистрации Python-функции как
    Prolog-предиката-действия агента.

    Оставлен для обратной совместимости. Новый код должен
    использовать simple_action / effector_action / generator_action
    из prolog.actions.

    Возвращает функцию-обёртку с метаданными:
        _agent_action, _action_name, _action_arity,
        _action_description, _action_type,
        _is_prolog_predicate, _predicate_name, _predicate_arity.
    """
    def decorator(func):
        wrapped = prolog_predicate(name, arity)(func)
        wrapped._agent_action = True
        wrapped._action_name = name
        wrapped._action_arity = arity
        wrapped._action_description = description
        wrapped._action_type = action_type

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if DEBUG:
                print(f"Agent action called: {name}/{arity}")
            return func(*args, **kwargs)

        # functools.wraps скопировал __dict__ из func, куда prolog_predicate
        # и последующие присваивания положили все метаданные. Поэтому wrapper
        # тоже имеет их и будет найден register_action_module.
        return wrapper
    return decorator


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
def is_ground(term) -> bool:
    """Проверяет, не содержит ли терм свободных переменных."""
    if isinstance(term, Variable):
        return False
    elif isinstance(term, Number):
        return True
    elif isinstance(term, Term):
        return all(is_ground(arg) for arg in term.arguments)
    else:
        return True


# ------------------------------------------------------------
# Встроенные агентские предикаты (действия)
# ------------------------------------------------------------
@agent_action('set_state', 1,
              description="Replace all facts with the same functor/arity "
                          "by this ground fact")
def set_state(goal, database, subst, depth):
    """Заменяет все факты с данным функтором/арностью на новый факт."""
    if len(goal.arguments) != 1:
        return []

    term = goal.arguments[0].substitute(subst)
    if not isinstance(term, Term) or not is_ground(term):
        return []

    from prolog.terms import Variable

    # Создаём шаблон головы с переменными
    functor = term.functor
    arity = len(term.arguments)
    vars = [Variable('_') for _ in range(arity)]
    head_template = Term(functor, vars)
    template_rule = Rule(head_template, Term('true', []))

    # Удаляем все подходящие факты
    database.retract_all(template_rule)

    # Добавляем новый факт
    database.add_rule(Rule(term, Term('true', [])))

    return [subst.copy()]


@agent_action('update_state', 1,
              description="Add a fact (without deleting old ones)")
def update_state(goal, database, subst, depth):
    """Добавляет факт, не удаляя старые."""
    if len(goal.arguments) != 1:
        return []

    term = goal.arguments[0].substitute(subst)
    if not isinstance(term, Term) or not is_ground(term):
        return []

    database.add_rule(Rule(term, Term('true', [])))

    return [subst.copy()]


# ------------------------------------------------------------
# Основной класс Agent
# ------------------------------------------------------------
class Agent:
    """
    ИИ-агент, управляемый Prolog-программой с возможностью вызова
    Python-действий через пользовательские предикаты.
    """

    def __init__(self, program: str = "", pl_file: str = "",
                 context: dict = None):
        """
        Инициализирует агента.

        Args:
            program: Текст программы на Prolog.
            pl_file: Файл с программой на Prolog для загрузки.
            context: Словарь с внешними ресурсами (LLM-клиенты, API и т.д.),
                     доступный из действий через database.context.
        """
        pl_prog = program
        if not program and pl_file:
            with open(pl_file, encoding='utf-8') as f:
                pl_prog = f.read()

        self.solver = Solver(pl_prog, context=context)
        self._actions: Dict[Tuple[str, int], Dict] = {}

        # Автоматически регистрируем встроенные агентские предикаты
        self._register_builtins()

    def _register_builtins(self):
        """Регистрирует встроенные предикаты set_state/1 и update_state/1."""
        current_module = sys.modules[__name__]
        self.register_action_module(current_module)

    def register_action(self, func):
        """Регистрирует одно действие, декорированное @agent_action
        или новыми декораторами (simple_action / effector_action /
        generator_action)."""
        if not hasattr(func, '_agent_action'):
            raise ValueError("Function is not decorated with @agent_action")

        name = func._action_name
        arity = func._action_arity
        self.solver.database.register_custom_predicate(name, arity, func)
        self._actions[(name, arity)] = {
            'description': func._action_description,
            'type': func._action_type,
            'function': func
        }
        if DEBUG:
            print(f"Registered agent action: {name}/{arity}")

    def register_action_module(self, module):
        """Регистрирует все действия из указанного модуля."""
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and hasattr(attr, '_agent_action'):
                self.register_action(attr)

    def list_actions(self) -> List[Dict]:
        """Возвращает список зарегистрированных действий с метаданными."""
        return [
            {
                'name': name,
                'arity': arity,
                **meta
            }
            for (name, arity), meta in self._actions.items()
        ]

    def query(self, query_text: str, with_explanation: bool = False):
        """
        Выполняет Prolog-запрос.

        Args:
            query_text: Текст запроса (должен оканчиваться точкой).
            with_explanation: Если True, возвращает также объект объяснения.

        Returns:
            Если with_explanation=False: словарь с результатами.
            Если with_explanation=True: кортеж (словарь с результатами,
                объект объяснения).
        """
        result = self.solver.find_solutions(
            query_text, with_explanation=with_explanation
        )
        if with_explanation:
            solutions, explanation = result
            parsed = self._parse_solutions(query_text, solutions)
            return parsed, explanation
        else:
            return self._parse_solutions(query_text, result)

    def query_one(self, query_text: str) -> Optional[Dict[str, Any]]:
        """
        Возвращает первое решение запроса или None.

        Отличает «запрос провалился» от «запрос успешен, но без свободных
        переменных». В первом случае возвращает None, во втором — словарь
        с маркером '__success__'.
        """
        result = self.query(query_text)
        if isinstance(result, dict):
            # Успех без свободных переменных: query() вернул True.
            if result.get('solution') is True:
                return {'__success__': True}
            if ('solutions' in result
                    and isinstance(result['solutions'], list)
                    and result['solutions']):
                return result['solutions'][0]
            if 'solution' in result:
                sol = result['solution']
                if isinstance(sol, list) and sol:
                    return sol[0]
                elif isinstance(sol, dict):
                    return sol
        return None

    def _parse_solutions(self, query_text, raw_solutions):
        """Переопределяем парсинг решений, чтобы возвращать Python-объекты."""
        from prolog.solver import ParseSolution
        # Стандартный парсинг (словарь с ключами 'solution', 'output' и т.д.)
        parsed = ParseSolution(query_text, raw_solutions, explanation=None)
        # Преобразуем строковые значения в Python-объекты
        if 'solution' in parsed:
            sol = parsed['solution']
            if isinstance(sol, dict):
                parsed['solution'] = self._convert_solution(sol)
            elif isinstance(sol, list):
                parsed['solution'] = [
                    self._convert_solution(s) for s in sol if s is not None
                ]
        return parsed

    def _convert_solution(self, sol_dict):
        """
        Рекурсивно преобразует строковые значения в Python-объекты
        (числа, списки). Использует ast.literal_eval для безопасного
        парсинга литералов.
        """
        if not isinstance(sol_dict, dict):
            return sol_dict
        converted = {}
        for var, val in sol_dict.items():
            if isinstance(val, str):
                try:
                    # ast.literal_eval безопасно парсит числа, строки,
                    # списки, кортежи, словари и т.д.
                    parsed = ast.literal_eval(val)
                    val = parsed
                except (ValueError, SyntaxError):
                    # Если не удалось, оставляем как строку
                    pass
            converted[var] = val
        return converted

    def add_fact(self, fact_text: str):
        """Добавляет один факт в базу знаний (совместимость).

        Делегирует в add_facts, чтобы не дублировать логику парсинга.
        """
        self.add_facts(fact_text)

    def add_facts(self, facts_text: str):
        """Добавляет пачку фактов/правил за ОДИН вызов парсера.

        В отличие от add_fact, рассчитан на многострочный текст: каждое
        правило отделено точкой. Это даёт ~3–5× ускорение при загрузке
        5+ фактов, потому что Parser создаётся и делает regex-токенизацию
        один раз на весь пакет, а не на каждый факт.

        Пример:
            agent.add_facts('''
                my_pos(5.0, 5.0).
                my_mode(search).
                observed(bravo, search, 10.0, 5.0, 17).
            ''')
        """
        text = facts_text.strip()
        if not text:
            return
        if not text.endswith('.'):
            text = text + '.'
        rules = Parser(text).parse_rules()
        for rule in rules:
            self.solver.database.add_rule(rule)
            if DEBUG:
                print(f"Added fact: {rule}")

    def retract_fact(self, fact_template: str) -> Optional[Dict]:
        """Удаляет первое правило, унифицирующееся с шаблоном."""
        if not fact_template.strip().endswith('.'):
            fact_template = fact_template.strip() + '.'
        rules = Parser(fact_template).parse_rules()
        if not rules:
            return None
        template_rule = rules[0]
        return self.solver.database.retract_first(template_rule)

    def retract_all(self, fact_template: str) -> int:
        """Удаляет все правила, унифицирующиеся с шаблоном."""
        if not fact_template.strip().endswith('.'):
            fact_template = fact_template.strip() + '.'
        rules = Parser(fact_template).parse_rules()
        if not rules:
            return 0
        template_rule = rules[0]
        return self.solver.database.retract_all(template_rule)

    @property
    def context(self) -> dict:
        return self.solver.database.context

    @context.setter
    def context(self, value: dict):
        self.solver.database.context = value

    def stream(self, query_text: str, with_explanation: bool = False):
        """
        Выполняет Prolog-запрос и возвращает генератор, выдающий решения
        по мере их нахождения. Полезно для длительных запросов или когда
        нужно обрабатывать результаты в реальном времени.

        Args:
            query_text: Текст запроса (должен оканчиваться точкой).
            with_explanation: Зарезервировано (пока не реализовано).

        Yields:
            - Если запрос содержит переменные: словарь
              {имя_переменной: значение} для каждого решения.
            - Если запрос не содержит переменных: True при успехе,
              False при неудаче.
            - В случае ошибки: словарь с ключом 'error'.
        """
        from prolog.parser import Parser
        from prolog.terms import Variable

        try:
            query = Parser(query_text).parse_query()
        except Exception as e:
            yield {'error': f'Ошибка парсинга запроса: {e}'}
            return

        # Находим все переменные в запросе
        variables = {}

        def find_vars(term):
            if isinstance(term, Variable) and term.name != '_':
                variables[term.name] = term
            elif isinstance(term, Term):
                for arg in term.arguments:
                    find_vars(arg)

        find_vars(query)

        has_vars = bool(variables)
        found = False

        try:
            # Используем внутренний генератор решений базы знаний
            for subst in self.solver.database._solve(
                query, self.solver.database.rule_index, {}, 0
            ):
                found = True
                if has_vars:
                    solution = {}
                    for var_name, var in variables.items():
                        val = var.substitute(subst)
                        solution[var_name] = prolog_to_python(val)
                    yield solution
                else:
                    yield True
        except Exception as e:
            yield {'error': f'Ошибка выполнения запроса: {e}'}
        finally:
            if not has_vars and not found:
                yield False

    def __repr__(self):
        return f"Agent(actions={len(self._actions)})"

    def save_kb(self, filename: str, format: str = 'prolog') -> None:
        """
        Сохраняет текущую базу знаний в файл указанного формата.

        Args:
            filename: Имя файла для сохранения.
            format:   'prolog' – текстовый формат Prolog (каждое правило
                      с точкой);
                      'binary' – бинарный формат (pickle) для быстрой
                      загрузки.
        """
        if format == 'prolog':
            with open(filename, 'w', encoding='utf-8') as f:
                for rule in self.solver.database.rules:
                    f.write(str(rule) + '\n')
            if DEBUG:
                print(f"База знаний сохранена в текстовом формате: {filename}")

        elif format == 'binary':
            # Сериализуем список правил с версией для будущих изменений
            data = {'version': 1, 'rules': self.solver.database.rules}
            with open(filename, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            if DEBUG:
                print(f"База знаний сохранена в бинарном формате: {filename}")

        else:
            raise ValueError("format должен быть 'prolog' или 'binary'")

    def load_kb(self, filename: str, format: str = 'prolog',
                clear: bool = False) -> None:
        """
        Загружает базу знаний из файла указанного формата.

        Args:
            filename: Имя файла для загрузки.
            format:   'prolog' – текстовый формат Prolog (парсится);
                      'binary' – бинарный формат (pickle, без парсинга).
            clear:    Если True, текущая база полностью очищается
                      перед загрузкой.
        """
        if clear:
            self.clear_kb()

        if format == 'prolog':
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.read()
            from prolog.parser import Parser
            try:
                rules = Parser(content).parse_rules()
            except Exception as e:
                raise ValueError(
                    f"Ошибка парсинга файла {filename}: {e}"
                )

            for rule in rules:
                self.solver.database.add_rule(rule)

            if DEBUG:
                print(f"Загружено {len(rules)} правил из текстового файла "
                      f"{filename}")

        elif format == 'binary':
            try:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
            except Exception as e:
                raise ValueError(
                    f"Ошибка загрузки бинарного файла {filename}: {e}"
                )

            if not isinstance(data, dict) or data.get('version') != 1:
                raise ValueError(
                    "Неверный формат бинарного файла или версия "
                    "не поддерживается"
                )

            rules = data['rules']
            for rule in rules:
                self.solver.database.add_rule(rule)

            if DEBUG:
                print(f"Загружено {len(rules)} правил из бинарного файла "
                      f"{filename}")

        else:
            raise ValueError("format должен быть 'prolog' или 'binary'")

    def clear_kb(self) -> None:
        """Полностью очищает базу знаний."""
        self.solver.database.rules.clear()
        self.solver.database.build_rule_index()
        if DEBUG:
            print("База знаний очищена")


# ------------------------------------------------------------
# Экспорт
# ------------------------------------------------------------
# Сохранён оригинальный __all__ (обратная совместимость для
# `from prolog.agent import *`). Реэкспортированные bridge-функции
# доступны через явный импорт:
#     from prolog.agent import prolog_to_python, python_to_prolog
__all__ = [
    'Agent',
    'agent_action',
    'simple_action',
    'effector_action',
    'generator_action',
]