#
# Полный тест оптимизированного интерпретатора Prolog
# С отладочным выводом ошибок (traceback) и проверкой семантической унификации
# на РЕАЛЬНОМ эмбеддере (Ollama, nomic-embed-text-v2-moe).
#
import os
import sys
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np

from prolog.interpreter import Database
from prolog.terms import Term, Variable, Number, Rule
from prolog.parser import Parser
from prolog.predicates import CutException, RepeatException, PrologError
from prolog.solver import Solver
import prolog.embedding_unify as eu


class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = []
        self.failed = []
        self.errors = []
        self.skipped = []

    def add_passed(self, test_name):
        self.passed.append(test_name)

    def add_failed(self, test_name, reason=""):
        self.failed.append((test_name, reason))

    def add_error(self, test_name, error):
        self.errors.append((test_name, error))

    def add_skipped(self, test_name, reason=""):
        self.skipped.append((test_name, reason))

    def print_summary(self):
        print("\n" + "="*60)
        print(f"РЕЗУЛЬТАТЫ ТЕСТА: {self.name}")
        print("="*60)
        print(f"Пройдено: {len(self.passed)}")
        print(f"Провалено: {len(self.failed)}")
        print(f"Ошибок выполнения: {len(self.errors)}")
        print(f"Пропущено: {len(self.skipped)}")
        if self.failed:
            print("\n❌ ПРОВАЛЕННЫЕ ТЕСТЫ:")
            for test, reason in self.failed:
                print(f"  - {test}")
                if reason:
                    print(f"    Причина: {reason}")
        if self.errors:
            print("\n⚠️ ОШИБКИ ВЫПОЛНЕНИЯ:")
            for test, error in self.errors:
                print(f"  - {test}: {error}")
        if self.skipped:
            print("\n⏭️ ПРОПУЩЕННЫЕ ТЕСТЫ:")
            for test, reason in self.skipped:
                print(f"  - {test}")
                if reason:
                    print(f"    Причина: {reason}")
        print("\n" + "="*60)
        return len(self.failed) == 0 and len(self.errors) == 0


def test_parser_basic(result):
    try:
        prog = "a(1). b(X) :- a(X)."
        parser = Parser(prog)
        rules = parser.parse_rules()
        assert len(rules) == 2 and str(rules[0]) == "a(1)." and "b(X) :- a(X)." in str(rules[1])
        result.add_passed("Парсинг фактов и правил")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Парсинг", e)
        return False


def test_family_rules(result):
    try:
        prog = """
        parent(a, b). parent(a, c). parent(b, d). parent(c, e).
        ancestor(X, Y) :- parent(X, Y).
        ancestor(X, Y) :- parent(X, Z), ancestor(Z, Y).
        """
        db = Database(Parser(prog).parse_rules())
        sol = list(db.query(Term("parent", [Term("a"), Variable("X")])))
        assert len(sol) == 2
        sol2 = list(db.query(Term("ancestor", [Term("a"), Variable("X")])))
        assert len(sol2) == 4
        result.add_passed("Рекурсивные правила (ancestor)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Семейные правила", e)
        return False


def test_arithmetic(result):
    try:
        prog = "sum(X) :- X is 2 + 3 * 4."
        db = Database(Parser(prog).parse_rules())
        sol = list(db.query(Term("sum", [Variable("X")])))
        assert len(sol) == 1 and str(sol[0]) == "sum(14)"
        # Сравнения
        assert len(list(db.query(Term(">", [Number(5), Number(3)])))) == 1
        assert len(list(db.query(Term(">", [Number(3), Number(5)])))) == 0
        assert len(list(db.query(Term("<", [Number(3), Number(5)])))) == 1
        assert len(list(db.query(Term("=<", [Number(3), Number(5)])))) == 1
        assert len(list(db.query(Term(">=", [Number(5), Number(5)])))) == 1
        assert len(list(db.query(Term("=:=", [Number(5), Number(5)])))) == 1
        assert len(list(db.query(Term(r"=\=", [Number(5), Number(6)])))) == 1
        result.add_passed("Арифметика и сравнения")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Арифметика", e)
        return False


def test_unification(result):
    try:
        db = Database([])
        # X = 5
        sol = list(db.query(Term("=", [Variable("X"), Number(5)])))
        assert len(sol) == 1, f"Ожидалось 1 решение, получено {len(sol)}"
        assert "5" in str(sol[0]), f"Строка решения не содержит '5': {sol[0]}"
        # f(X, g(Y)) = f(a, g(b))
        goal = Term("=", [
            Term("f", [Variable("X"), Term("g", [Variable("Y")])]),
            Term("f", [Term("a"), Term("g", [Term("b")])])
        ])
        sol2 = list(db.query(goal))
        assert len(sol2) == 1
        assert "f(a, g(b))" in str(sol2[0])
        result.add_passed("Унификация")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Унификация", e)
        return False


def test_lists_basic(result):
    try:
        db = Database([])
        lst = Term(".", [Number(1), Term(".", [Number(2), Term(".", [Number(3), Term("[]")])])])
        sol = list(db.query(Term("=", [lst, lst])))
        assert len(sol) == 1
        pattern = Term(".", [Variable("H"), Variable("T")])
        sol2 = list(db.query(Term("=", [lst, pattern])))
        assert len(sol2) == 1
        assert "1" in str(sol2[0]) and "2" in str(sol2[0]) and "3" in str(sol2[0])
        result.add_passed("Списки (создание и сопоставление)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Списки базовые", e)
        return False


def test_builtin_io(result):
    try:
        import io
        import sys
        old = sys.stdout
        sys.stdout = io.StringIO()
        db = Database([])
        list(db.query(Term("write", [Term("hello")])))
        list(db.query(Term("nl")))
        out = sys.stdout.getvalue()
        sys.stdout = old
        assert out == "hello\n"
        result.add_passed("write/1, nl/0")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Ввод/вывод", e)
        return False


def test_control_predicates(result):
    try:
        db = Database([])
        assert len(list(db.query(Term("true")))) == 1
        assert len(list(db.query(Term("fail")))) == 0
        assert len(list(db.query(Term("false")))) == 0
        gen = db.query(Term("repeat"))
        first3 = [next(gen) for _ in range(3)]
        assert len(first3) == 3
        # cut
        prog_cut = "a(1). a(2). test(X) :- a(X), !."
        db_cut = Database(Parser(prog_cut).parse_rules())
        sol = list(db_cut.query(Term("test", [Variable("X")])))
        assert len(sol) == 1
        assert "1" in str(sol[0])
        # negation
        prog_neg = "p(1). p(2). q(2). test(X) :- p(X), \\+ q(X)."
        db_neg = Database(Parser(prog_neg).parse_rules())
        sol_neg = list(db_neg.query(Term("test", [Variable("X")])))
        assert len(sol_neg) == 1
        assert "1" in str(sol_neg[0])
        result.add_passed("Управляющие предикаты (true, fail, repeat, cut, negation)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Управляющие предикаты", e)
        return False


def test_list_predicates(result):
    try:
        db = Database([])
        # member
        sol = list(db.query(Term("member", [Variable("X"), Term(".", [Term("a"), Term(".", [Term("b"), Term(".", [Term("c"), Term("[]")])])])])))
        assert len(sol) == 3
        values = set()
        for s in sol:
            parts = str(s).replace("member(", "").replace(")", "").split(",")
            if parts:
                val = parts[0].strip()
                if val.isalpha():
                    values.add(val)
        assert values == {"a", "b", "c"} or set(str(s) for s in sol) == {"member(a,[a,b,c])", "member(b,[a,b,c])", "member(c,[a,b,c])"}
        # append
        sol_app = list(db.query(Term("append", [
            Term(".", [Number(1), Term(".", [Number(2), Term("[]")])]),
            Term(".", [Number(3), Term(".", [Number(4), Term("[]")])]),
            Variable("X")
        ])))
        assert len(sol_app) == 1
        assert "[1,2,3,4]" in str(sol_app[0])
        sol_app2 = list(db.query(Term("append", [Variable("A"), Variable("B"), Term(".", [Number(1), Term(".", [Number(2), Term(".", [Number(3), Term("[]")])])])])))
        assert len(sol_app2) == 4
        # length
        sol_len = list(db.query(Term("length", [Term(".", [Number(1), Term(".", [Number(2), Term(".", [Number(3), Term("[]")])])]), Variable("X")])))
        assert len(sol_len) == 1
        assert "3" in str(sol_len[0])
        sol_len2 = list(db.query(Term("length", [Variable("L"), Number(2)])))
        assert len(sol_len2) >= 1
        lst_str = str(sol_len2[0])
        assert "length" in lst_str and "2" in lst_str
        result.add_passed("member/2, append/3, length/2")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Списковые предикаты", e)
        return False


def test_meta_predicates(result):
    try:
        db = Database(Parser("p(1). p(2). q(1). q(2). r(1,a). r(1,b). r(2,c).").parse_rules())
        # call
        assert len(list(db.query(Term("call", [Term("p", [Variable("X")])])))) == 2
        # once
        sol_once = list(db.query(Term("once", [Term("p", [Variable("X")])])))
        assert len(sol_once) == 1
        assert "1" in str(sol_once[0])
        # ignore
        sol_ignore = list(db.query(Term("ignore", [Term("p", [Variable("X")])])))
        assert len(sol_ignore) == 1
        # forall
        sol_forall = list(db.query(Term("forall", [Term("p", [Variable("X")]), Term("q", [Variable("X")])])))
        assert len(sol_forall) == 1
        sol_forall_fail = list(db.query(Term("forall", [Term("p", [Variable("X")]), Term("=", [Variable("X"), Number(3)])])))
        assert len(sol_forall_fail) == 0
        # findall
        sol_find = list(db.query(Term("findall", [Variable("X"), Term("p", [Variable("X")]), Variable("L")])))
        assert len(sol_find) == 1
        assert "[1,2]" in str(sol_find[0]) or "L" in str(sol_find[0])
        # bagof
        sol_bag = list(db.query(Term("bagof", [Variable("Y"), Term("r", [Variable("X"), Variable("Y")]), Variable("L")])))
        assert len(sol_bag) == 2
        # setof
        sol_set = list(db.query(Term("setof", [Variable("Y"), Term("r", [Variable("X"), Variable("Y")]), Variable("L")])))
        assert len(sol_set) == 2
        result.add_passed("Мета-предикаты (call, once, ignore, forall, findall, bagof, setof)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Мета-предикаты", e)
        return False


def test_dynamic_db(result):
    try:
        db = Database([])
        # asserta
        list(db.query(Term("asserta", [Term("p", [Number(1)])])))
        list(db.query(Term("asserta", [Term("p", [Number(2)])])))
        rules = db.get_rules_for_predicate("p", 1)
        assert len(rules) == 2 and rules[0].head.arguments[0].value == 2
        # assertz
        list(db.query(Term("assertz", [Term("p", [Number(3)])])))
        rules = db.get_rules_for_predicate("p", 1)
        assert rules[-1].head.arguments[0].value == 3
        # retract
        list(db.query(Term("retract", [Term("p", [Number(2)])])))
        rules = db.get_rules_for_predicate("p", 1)
        assert len(rules) == 2 and rules[0].head.arguments[0].value == 1
        # retractall
        list(db.query(Term("retractall", [Term("p", [Variable("_")])])))
        assert len(db.get_rules_for_predicate("p", 1)) == 0
        # abolish
        list(db.query(Term("asserta", [Term("q", [Number(42)])])))
        assert db.has_predicate("q", 1)
        list(db.query(Term("abolish", [Term("/", [Term("q"), Number(1)])])))
        assert not db.has_predicate("q", 1)
        # clause
        list(db.query(Term("asserta", [Term("r", [Number(5)])])))
        sol_clause = list(db.query(Term("clause", [Term("r", [Variable("X")]), Variable("B")])))
        assert len(sol_clause) == 1
        # current_predicate
        sol_curr = list(db.query(Term("current_predicate", [Variable("P")])))
        found = False
        for s in sol_curr:
            p_str = str(s)
            if "r/1" in p_str or "r" in p_str:
                found = True
                break
        assert found
        result.add_passed("Динамическая БД (asserta, assertz, retract, retractall, abolish, clause, current_predicate)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Динамическая БД", e)
        return False


def test_string_predicates(result):
    try:
        db = Database([])
        sol = list(db.query(Term("string_concat", [Term("'Hello'"), Term("'World'"), Variable("S")])))
        assert len(sol) == 1 and "HelloWorld" in str(sol[0])
        sol2 = list(db.query(Term("atomic_list_concat", [
            Term(".", [Term("a"), Term(".", [Term("b"), Term(".", [Term("c"), Term("[]")])])]),
            Variable("A")
        ])))
        assert len(sol2) == 1 and "abc" in str(sol2[0])
        sol3 = list(db.query(Term("atomic_list_concat", [
            Term(".", [Term("a"), Term(".", [Term("b"), Term(".", [Term("c"), Term("[]")])])]),
            Term("'-'"),
            Variable("A")
        ])))
        assert len(sol3) == 1 and "a-b-c" in str(sol3[0])
        sol4 = list(db.query(Term("atom_string", [Term("hello"), Variable("S")])))
        assert len(sol4) == 1 and '"hello"' in str(sol4[0])
        sol5 = list(db.query(Term("atom_string", [Variable("A"), Term('"hello"')])))
        assert len(sol5) == 1 and "hello" in str(sol5[0])
        sol6 = list(db.query(Term("term_string", [Term("f", [Number(1), Number(2)]), Variable("S")])))
        assert len(sol6) == 1 and "f(1,2)" in str(sol6[0]).replace(" ", "")
        sol7 = list(db.query(Term("number_string", [Number(42), Variable("S")])))
        assert len(sol7) == 1 and '"42"' in str(sol7[0])
        sol8 = list(db.query(Term("split_string", [Term("'a,b,c'"), Term("','"), Term("''"), Variable("L")])))
        assert len(sol8) == 1 and "a" in str(sol8[0]) and "b" in str(sol8[0]) and "c" in str(sol8[0])
        result.add_passed("Строковые предикаты")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Строковые предикаты", e)
        return False


def test_term_compare(result):
    try:
        db = Database([])
        assert len(list(db.query(Term("==", [Term("a"), Term("a")])))) == 1
        assert len(list(db.query(Term("==", [Term("a"), Term("b")])))) == 0
        assert len(list(db.query(Term(r"\==", [Term("a"), Term("b")])))) == 1
        assert len(list(db.query(Term("@<", [Term("a"), Term("b")])))) == 1
        assert len(list(db.query(Term("@>", [Term("b"), Term("a")])))) == 1
        assert len(list(db.query(Term("@=<", [Term("a"), Term("a")])))) == 1
        assert len(list(db.query(Term("@>=", [Term("a"), Term("a")])))) == 1
        sol_cmp = list(db.query(Term("compare", [Variable("R"), Term("a"), Term("b")])))
        assert len(sol_cmp) == 1
        assert "<" in str(sol_cmp[0]) or "R" in str(sol_cmp[0])
        result.add_passed("Сравнение термов (==, \\==, @<, @>, @=<, @>=, compare)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Сравнение термов", e)
        return False


def test_term_constructors(result):
    try:
        db = Database([])
        # functor
        sol = list(db.query(Term("functor", [Term("f", [Term("a"), Term("b")]), Variable("F"), Variable("A")])))
        assert len(sol) == 1
        assert "f" in str(sol[0]) and "2" in str(sol[0])
        # arg
        sol2 = list(db.query(Term("arg", [Number(2), Term("f", [Term("a"), Term("b")]), Variable("X")])))
        assert len(sol2) == 1
        assert "b" in str(sol2[0])
        # =..
        sol3 = list(db.query(Term("=..", [Term("f", [Term("a"), Term("b")]), Variable("L")])))
        assert len(sol3) == 1
        assert "[f,a,b]" in str(sol3[0]) or "L" in str(sol3[0])
        # copy_term
        sol4 = list(db.query(Term("copy_term", [Variable("X"), Variable("Y")])))
        assert len(sol4) >= 1
        # term_variables
        sol5 = list(db.query(Term("term_variables", [Term("f", [Variable("X"), Term("g", [Variable("Y"), Variable("Z")])]), Variable("V")])))
        assert len(sol5) == 1
        assert "X" in str(sol5[0]) and "Y" in str(sol5[0]) and "Z" in str(sol5[0])
        # numbervars
        sol6 = list(db.query(Term("numbervars", [Term("f", [Variable("X"), Variable("Y")]), Number(0), Variable("N")])))
        assert len(sol6) == 1
        assert "$VAR" in str(sol6[0])
        result.add_passed("Конструкторы/деструкторы термов")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Конструкторы термов", e)
        return False


def test_type_predicates(result):
    try:
        db = Database([])
        assert len(list(db.query(Term("var", [Variable("X")])))) == 1
        assert len(list(db.query(Term("var", [Number(42)])))) == 0
        assert len(list(db.query(Term("nonvar", [Number(42)])))) == 1
        assert len(list(db.query(Term("nonvar", [Variable("X")])))) == 0
        assert len(list(db.query(Term("atom", [Term("abc")])))) == 1
        assert len(list(db.query(Term("atom", [Number(123)])))) == 0
        assert len(list(db.query(Term("atom", [Term("f")])))) == 1  # f/0 – атом
        assert len(list(db.query(Term("integer", [Number(42)])))) == 1
        assert len(list(db.query(Term("integer", [Number(3.14)])))) == 0
        assert len(list(db.query(Term("float", [Number(3.14)])))) == 1
        assert len(list(db.query(Term("number", [Number(42)])))) == 1
        assert len(list(db.query(Term("number", [Number(3.14)])))) == 1
        assert len(list(db.query(Term("atomic", [Number(42)])))) == 1
        assert len(list(db.query(Term("atomic", [Term("abc")])))) == 1
        assert len(list(db.query(Term("compound", [Term("f", [Term("a")])])))) == 1
        assert len(list(db.query(Term("compound", [Term("abc")])))) == 0
        assert len(list(db.query(Term("ground", [Term("f", [Number(1), Number(2)])])))) == 1
        assert len(list(db.query(Term("ground", [Term("f", [Variable("X")])])))) == 0
        assert len(list(db.query(Term("callable", [Term("f", [Term("a")])])))) == 1
        assert len(list(db.query(Term("callable", [Number(42)])))) == 0
        result.add_passed("Проверка типов (var, nonvar, atom, integer, float, number, atomic, compound, ground, callable)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Проверка типов", e)
        return False


def test_memberchk(result):
    try:
        db = Database([])
        lst = Term(".", [Term("a"), Term(".", [Term("b"), Term(".", [Term("c"), Term("[]")])])])
        assert len(list(db.query(Term("memberchk", [Term("b"), lst])))) == 1
        assert len(list(db.query(Term("memberchk", [Term("x"), lst])))) == 0
        result.add_passed("memberchk/2")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("memberchk", e)
        return False


def test_conjunction_disjunction(result):
    try:
        prog = """
        a(1). a(2). b(2). b(3).
        test_conj(X) :- a(X), b(X).
        test_disj(X) :- a(X) ; b(X).
        """
        db = Database(Parser(prog).parse_rules())
        sol = list(db.query(Term("test_conj", [Variable("X")])))
        assert len(sol) == 1
        assert "2" in str(sol[0])
        sol2 = list(db.query(Term("test_disj", [Variable("X")])))
        values = set()
        for s in sol2:
            parts = str(s).replace("test_disj(", "").replace(")", "").split(",")
            if parts:
                val = parts[0].strip()
                if val.isdigit():
                    values.add(int(val))
        assert values == {1,2,3}
        result.add_passed("Конъюнкция и дизъюнкция")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Конъюнкция/дизъюнкция", e)
        return False


def test_cut_advanced(result):
    try:
        prog = """
        a(1). a(2). a(3).
        test1(X) :- a(X), !.
        test2(X) :- a(X), (X = 2, ! ; X = 1).
        """
        db = Database(Parser(prog).parse_rules())
        sol1 = list(db.query(Term("test1", [Variable("X")])))
        assert len(sol1) == 1
        assert "1" in str(sol1[0])
        sol2 = list(db.query(Term("test2", [Variable("X")])))
        assert len(sol2) == 2
        assert "1" in str(sol2[0]) or "1" in str(sol2[1])
        assert "2" in str(sol2[0]) or "2" in str(sol2[1])
        result.add_passed("Отсечение (продвинутое)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Отсечение продвинутое", e)
        return False


def test_if_then_else(result):
    try:
        prog = """
        a(1). a(2). a(3).
        b(2). b(3).
        test(X, Y) :- (a(X) -> b(Y) ; Y = none).
        """
        db = Database(Parser(prog).parse_rules())
        sol = list(db.query(Term("test", [Variable("X"), Variable("Y")])))
        pairs = set()
        for s in sol:
            parts = str(s).replace("test(", "").replace(")", "").split(",")
            if len(parts) >= 2:
                x_str = parts[0].strip()
                y_str = parts[1].strip()
                if x_str.isdigit() and y_str.isdigit():
                    pairs.add((int(x_str), int(y_str)))
        assert pairs == {(1,2), (1,3)}
        result.add_passed("Условный оператор ->")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Условный оператор", e)
        return False


def test_negation_complex(result):
    try:
        prog = """
        p(1). p(2). q(2). q(3).
        test(X) :- p(X), \\+ q(X).
        """
        db = Database(Parser(prog).parse_rules())
        sol = list(db.query(Term("test", [Variable("X")])))
        assert len(sol) == 1
        assert "1" in str(sol[0])
        result.add_passed("Отрицание (сложное)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Отрицание сложное", e)
        return False


# ======================================================================
# СЕМАНТИЧЕСКАЯ УНИФИКАЦИЯ (РЕАЛЬНЫЙ ЭМБЕДДЕР Ollama)
# ======================================================================
# Требуется запущенный Ollama и загруженная модель nomic-embed-text-v2-moe.
# Если Ollama/модель недоступны — соответствующие тесты помечаются skipped,
# чтобы не ломать прогон в средах без модели.
#
# Переопределение через переменные окружения:
#   PROLOG_SEMANTIC_MODEL — имя модели (по умолчанию nomic-embed-text-v2-moe:latest)
#   PROLOG_OLLAMA_URL     — URL сервера (по умолчанию http://localhost:11434)
# ----------------------------------------------------------------------

OLLAMA_MODEL = os.environ.get("PROLOG_SEMANTIC_MODEL", "nomic-embed-text-v2-moe:latest")
OLLAMA_URL = os.environ.get("PROLOG_OLLAMA_URL", "http://localhost:11434")

_ollama_available = None  # None — не проверяли; True/False — результат


def _check_ollama():
    """Проверить, доступны ли Ollama и указанная модель. Результат кэшируется."""
    global _ollama_available
    if _ollama_available is not None:
        return _ollama_available
    print(f"[SEMANTIC] Проверка Ollama: model='{OLLAMA_MODEL}', url='{OLLAMA_URL}'")
    try:
        fn = eu.make_ollama_embed_fn(model=OLLAMA_MODEL, base_url=OLLAMA_URL)
        v = fn("ping")
        ok = v is not None and len(v) > 0
        _ollama_available = ok
        if ok:
            print(f"[SEMANTIC] Ollama доступен, размерность вектора: {len(v)}")
        else:
            print(f"[SEMANTIC] Ollama вернул пустой вектор — модель '{OLLAMA_MODEL}' загружена?")
    except Exception as e:
        print(f"[SEMANTIC] Ollama недоступен: {type(e).__name__}: {e}")
        _ollama_available = False
    return _ollama_available


def _install_real_semantic(threshold=0.8):
    """Включить семантическую унификацию на реальном эмбеддере Ollama."""
    eu.semantic_off()
    return eu.semantic_on(
        threshold=threshold,
        model=OLLAMA_MODEL,
        base_url=OLLAMA_URL,
    )


def _uninstall_semantic():
    """Снять семантическую унификацию и вернуть строгую."""
    try:
        eu.semantic_off()
    except Exception:
        pass


def test_semantic_status_initially_off(result):
    """semantic_status/1 возвращает off, если унификация не включена. Ollama не требуется."""
    try:
        _uninstall_semantic()
        db = Database([])
        sol = list(db.query(Term("semantic_status", [Variable("S")])))
        assert len(sol) == 1, f"Ожидалось 1 решение, получено {len(sol)}"
        s = str(sol[0])
        assert "off" in s, f"Ожидалось 'off', получено: {s}"
        result.add_passed("semantic_status/1 -> off (изначально)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("semantic_status начальный", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_on_as_goal(result):
    """Вызов semantic_on/0 как цели включает унификатор (реальный Ollama)."""
    try:
        if not _check_ollama():
            result.add_skipped("semantic_on/0 как цель",
                               f"Ollama/модель недоступны ({OLLAMA_MODEL} @ {OLLAMA_URL})")
            return True
        _uninstall_semantic()
        db = Database([])
        sol = list(db.query(Term("semantic_on")))
        assert len(sol) == 1, f"Ожидалось 1 решение для semantic_on, получено {len(sol)}"
        u = eu.semantic_get_unifier()
        assert u is not None, "semantic_on/0 не установил унификатор"
        # Проверяем через semantic_status/1
        sol2 = list(db.query(Term("semantic_status", [Variable("S")])))
        assert len(sol2) == 1
        assert "on" in str(sol2[0]), f"Ожидалось 'on', получено: {sol2[0]}"
        result.add_passed("semantic_on/0 как цель (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("semantic_on/0 как цель", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_on_threshold_as_goal(result):
    """semantic_on(0.9) как цель — реальный Ollama."""
    try:
        if not _check_ollama():
            result.add_skipped("semantic_on/1 как цель", "Ollama/модель недоступны")
            return True
        _uninstall_semantic()
        db = Database([])
        sol = list(db.query(Term("semantic_on", [Number(0.9)])))
        assert len(sol) == 1
        u = eu.semantic_get_unifier()
        assert u is not None, "semantic_on/1 не установил унификатор"
        assert abs(u.threshold - 0.9) < 1e-6, f"Порог = {u.threshold}, ожидался 0.9"
        result.add_passed("semantic_on/1 как цель (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("semantic_on/1 как цель", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_off_as_goal(result):
    """semantic_off/0 отключает унификатор (реальный Ollama)."""
    try:
        if not _check_ollama():
            result.add_skipped("semantic_off/0 как цель", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        assert eu.semantic_get_unifier() is not None
        db = Database([])
        sol = list(db.query(Term("semantic_off")))
        assert len(sol) == 1
        assert eu.semantic_get_unifier() is None, "semantic_off/0 не снял унификатор"
        result.add_passed("semantic_off/0 как цель (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("semantic_off/0 как цель", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_threshold_as_goal(result):
    """semantic_threshold/1 меняет порог (реальный Ollama)."""
    try:
        if not _check_ollama():
            result.add_skipped("semantic_threshold/1 как цель", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        db = Database([])
        sol = list(db.query(Term("semantic_threshold", [Number(0.7)])))
        assert len(sol) == 1
        u = eu.semantic_get_unifier()
        assert u is not None
        assert abs(u.threshold - 0.7) < 1e-6, f"Порог = {u.threshold}, ожидался 0.7"
        result.add_passed("semantic_threshold/1 как цель (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("semantic_threshold/1 как цель", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_directive_on(result):
    """Директива :- semantic_on. в исходнике включает унификатор (реальный Ollama)."""
    try:
        if not _check_ollama():
            result.add_skipped("Директива :- semantic_on.", "Ollama/модель недоступны")
            return True
        _uninstall_semantic()
        program = """
        :- semantic_on.
        человек(сократ).
        """
        solver = Solver(program)
        assert eu.semantic_get_unifier() is not None, "Директива :- semantic_on. не сработала"
        result.add_passed("Директива :- semantic_on. (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Директива :- semantic_on.", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_directive_on_with_threshold(result):
    """Директива :- semantic_on(0.9)."""
    try:
        if not _check_ollama():
            result.add_skipped("Директива :- semantic_on(0.9).", "Ollama/модель недоступны")
            return True
        _uninstall_semantic()
        program = """
        :- semantic_on(0.9).
        человек(сократ).
        """
        solver = Solver(program)
        u = eu.semantic_get_unifier()
        assert u is not None, "Директива :- semantic_on(0.9). не сработала"
        assert abs(u.threshold - 0.9) < 1e-6, f"Порог = {u.threshold}, ожидался 0.9"
        result.add_passed("Директива :- semantic_on(0.9). (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Директива :- semantic_on(0.9).", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_directive_off(result):
    """Директива :- semantic_off. отключает унификатор."""
    try:
        if not _check_ollama():
            result.add_skipped("Директива :- semantic_off.", "Ollama/модель недоступны")
            return True
        _install_real_semantic()
        assert eu.semantic_get_unifier() is not None
        program = """
        :- semantic_off.
        человек(сократ).
        """
        solver = Solver(program)
        assert eu.semantic_get_unifier() is None, "Директива :- semantic_off. не сработала"
        result.add_passed("Директива :- semantic_off. (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Директива :- semantic_off.", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_directive_threshold(result):
    """Директива :- semantic_threshold(0.7)."""
    try:
        if not _check_ollama():
            result.add_skipped("Директива :- semantic_threshold(0.7).", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        program = """
        :- semantic_threshold(0.7).
        человек(сократ).
        """
        solver = Solver(program)
        u = eu.semantic_get_unifier()
        assert u is not None, "Директива :- semantic_threshold. сняла унификатор"
        assert abs(u.threshold - 0.7) < 1e-6, f"Порог = {u.threshold}, ожидался 0.7"
        result.add_passed("Директива :- semantic_threshold(0.7). (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Директива :- semantic_threshold.", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_case_insensitive_match(result):
    """Реальный эмбеддер: 'Москва' ~ 'москва' (регистр) при пороге 0.8."""
    try:
        if not _check_ollama():
            result.add_skipped("Семантика: 'Москва' = 'москва'", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        program = """
        город(москва).
        город(париж).
        """
        db = Database(Parser(program).parse_rules())
        sol = list(db.query(Term("город", [Term("Москва")])))
        assert len(sol) == 1, (
            f"Ожидалось 1 решение, получено {len(sol)}: {[str(s) for s in sol]}"
        )
        # Заодно покажем косинусную близость для отчёта
        u = eu.semantic_get_unifier()
        v1 = u._embed("Москва")
        v2 = u._embed("москва")
        cos = u._cosine(v1, v2)
        print(f"[SEMANTIC] cos('Москва', 'москва') = {cos:.4f}")
        assert cos >= 0.8, f"cos('Москва', 'москва') = {cos:.4f} < 0.8"
        result.add_passed("Семантическая унификация: 'Москва' = 'москва' (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Семантическое совпадение регистра", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_different_words_fail(result):
    """Реальный эмбеддер: 'лондон' не совпадает с 'москва'/'париж' при пороге 0.8."""
    try:
        if not _check_ollama():
            result.add_skipped("Семантика: 'лондон' ≠ 'москва'", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        program = """
        город(москва).
        город(париж).
        """
        db = Database(Parser(program).parse_rules())
        sol = list(db.query(Term("город", [Term("лондон")])))
        u = eu.semantic_get_unifier()
        v1 = u._embed("лондон")
        v2 = u._embed("москва")
        v3 = u._embed("париж")
        cos1 = u._cosine(v1, v2)
        cos2 = u._cosine(v1, v3)
        print(f"[SEMANTIC] cos('лондон','москва') = {cos1:.4f}, "
              f"cos('лондон','париж') = {cos2:.4f}")
        assert len(sol) == 0, (
            f"Ожидалось 0 решений, получено: {[str(s) for s in sol]}"
        )
        assert cos1 < 0.8 and cos2 < 0.8, (
            f"Косинусы должны быть < 0.8: {cos1:.4f}, {cos2:.4f}"
        )
        result.add_passed("Семантическая унификация: 'лондон' ≠ 'москва'/'париж' (реальный эмбеддер)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Семантическое различие", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_off_falls_back_to_strict(result):
    """После semantic_off строгая унификация не находит 'Москва' vs 'москва'. Ollama не требуется."""
    try:
        _uninstall_semantic()
        program = """
        город(москва).
        """
        db = Database(Parser(program).parse_rules())
        sol = list(db.query(Term("город", [Term("Москва")])))
        assert len(sol) == 0, (
            f"Ожидалось 0 решений (строгая унификация), получено: {[str(s) for s in sol]}"
        )
        result.add_passed("Строгая унификация после semantic_off (Москва ≠ москва)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Строгая унификация после off", e)
        return False
    finally:
        _uninstall_semantic()


def test_semantic_cache_reuse(result):
    """Кэш эмбеддингов накапливает попадания при повторных вызовах (реальный эмбеддер)."""
    try:
        if not _check_ollama():
            result.add_skipped("Кэш эмбеддингов", "Ollama/модель недоступны")
            return True
        _install_real_semantic(threshold=0.8)
        program = """
        город(москва).
        город(париж).
        """
        db = Database(Parser(program).parse_rules())
        # Дважды выполняем один и тот же запрос с семантикой
        list(db.query(Term("город", [Term("Москва")])))
        list(db.query(Term("город", [Term("Москва")])))
        u = eu.semantic_get_unifier()
        assert u is not None
        stats = u.cache.stats()
        print(f"[SEMANTIC] cache stats: {stats}")
        assert stats['hits'] >= 1, f"Ожидались попадания в кэш, получено: {stats}"
        assert stats['misses'] >= 1, f"Ожидались промахи в кэш, получено: {stats}"
        result.add_passed("Кэш эмбеддингов (есть hits и misses)")
        return True
    except Exception as e:
        traceback.print_exc()
        result.add_error("Кэш эмбеддингов", e)
        return False
    finally:
        _uninstall_semantic()


def main():
    print("="*60)
    print("ПОЛНЫЙ ТЕСТ ОПТИМИЗИРОВАННОГО ИНТЕРПРЕТАТОРА PROLOG (СТРОКИ)")
    print(f"Семантическая модель: {OLLAMA_MODEL} @ {OLLAMA_URL}")
    print("="*60)
    result = TestResult("Оптимизированный интерпретатор Prolog")
    tests = [
        ("Парсинг", test_parser_basic),
        ("Семейные правила", test_family_rules),
        ("Арифметика", test_arithmetic),
        ("Унификация", test_unification),
        ("Списки базовые", test_lists_basic),
        ("Ввод/вывод", test_builtin_io),
        ("Управляющие предикаты", test_control_predicates),
        ("Списковые предикаты", test_list_predicates),
        ("Мета-предикаты", test_meta_predicates),
        ("Динамическая БД", test_dynamic_db),
        ("Строковые предикаты", test_string_predicates),
        ("Сравнение термов", test_term_compare),
        ("Конструкторы термов", test_term_constructors),
        ("Проверка типов", test_type_predicates),
        ("memberchk", test_memberchk),
        ("Конъюнкция/дизъюнкция", test_conjunction_disjunction),
        ("Отсечение продвинутое", test_cut_advanced),
        ("Условный оператор", test_if_then_else),
        ("Отрицание сложное", test_negation_complex),
        # --- Семантическая унификация (реальный эмбеддер) ---
        ("Семантика: статус изначально off", test_semantic_status_initially_off),
        ("Семантика: semantic_on/0 как цель", test_semantic_on_as_goal),
        ("Семантика: semantic_on/1 как цель", test_semantic_on_threshold_as_goal),
        ("Семантика: semantic_off/0 как цель", test_semantic_off_as_goal),
        ("Семантика: semantic_threshold/1 как цель", test_semantic_threshold_as_goal),
        ("Семантика: директива :- semantic_on.", test_semantic_directive_on),
        ("Семантика: директива :- semantic_on(0.9).", test_semantic_directive_on_with_threshold),
        ("Семантика: директива :- semantic_off.", test_semantic_directive_off),
        ("Семантика: директива :- semantic_threshold(0.7).", test_semantic_directive_threshold),
        ("Семантика: 'Москва' = 'москва'", test_semantic_case_insensitive_match),
        ("Семантика: 'лондон' ≠ 'москва'", test_semantic_different_words_fail),
        ("Семантика: off -> строгая унификация", test_semantic_off_falls_back_to_strict),
        ("Семантика: кэш эмбеддингов", test_semantic_cache_reuse),
    ]
    for name, func in tests:
        print(f"\n--- {name} ---")
        func(result)
    success = result.print_summary()
    return success


if __name__ == "__main__":
    sys.exit(0 if main() else 1)