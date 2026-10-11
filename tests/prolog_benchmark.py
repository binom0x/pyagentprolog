"""prolog_benchmark.py

Бенчмарк для выявления компонентов PyAgentProlog, которые имеет смысл
переносить на GPU (NumPy/CuPy/TensorFlow).

Классификация кандидатов:
    GPU_GOOD      — регулярные батчевые операции, большой объём данных
    GPU_MAYBE     — векторизуемо, но объём может быть мал
    GPU_BAD       — нерегулярные / последовательные / мелкие операции
    CPU_ONLY      — символьные вычисления без SIMD-потенциала

Патч A (бенчмарк):
  * Тест [5] "tail_recursion" переименован в "non-tail" и ограничен
    глубинами (50, 100, 200). Программа `count(N, R)` содержит
    накопление `R is R1 + 1` после рекурсивного вызова, поэтому TCO
    не срабатывает и subst растёт линейно с глубиной. Наша реализация
    subst через dict.copy() даёт O(N²) — при N=1000 это ~18 секунд,
    при N=5000 — минуты. Тест теперь честно помечен и быстро проходит.
  * Добавлен тест [5b] "tail_acc" — настоящая хвостовая рекурсия с
    аккумулятором. После рекурсивного вызова нет других целей, TCO
    срабатывает, subst не растёт. Ожидается O(N) вместо O(N²).

(c) 2026 Andrey Vorfolomeev
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
sys.path.append('d:/BFX/Repositories/')

# Поднимаем лимит рекурсии ДО импорта prolog — некоторые встроенные
# предикаты (member/length/append) реализованы рекурсивными генераторами.
sys.setrecursionlimit(100_000)

# ---------------------------------------------------------------------
# Импорты вашего Prolog
# ---------------------------------------------------------------------
from prolog.parser import Parser
from prolog.terms import Term, Variable, Number, Rule  # noqa: F401
from prolog.interpreter import Database
from prolog.solver import Solver
from prolog.predicates import evaluate_expression


# =====================================================================
# Замер времени
# =====================================================================
@dataclass
class BenchmarkResult:
    name: str
    category: str
    iterations: int = 0
    total_time: float = 0.0
    per_call_us: float = 0.0
    gpu_verdict: str = "CPU_ONLY"
    notes: str = ""
    extra: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"{self.name:<32} {self.category:<12} "
                f"{self.per_call_us:>12.2f} µs   {self.gpu_verdict:<10}  {self.notes}")


def bench(name: str, category: str, fn: Callable, *,
          repeat: int = 1, warmup: int = 1,
          gpu_verdict: str = "CPU_ONLY", notes: str = "",
          extra: Optional[Dict] = None) -> BenchmarkResult:
    for _ in range(warmup):
        fn()
    gc.disable()
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    t1 = time.perf_counter()
    gc.enable()
    total = t1 - t0
    return BenchmarkResult(
        name=name, category=category, iterations=repeat,
        total_time=total, per_call_us=(total / repeat) * 1e6,
        gpu_verdict=gpu_verdict, notes=notes, extra=extra or {}
    )


def bench_best_of(name: str, category: str, fn: Callable, *,
                  repeat: int = 5, gpu_verdict: str = "CPU_ONLY",
                  notes: str = "", extra: Optional[Dict] = None) -> BenchmarkResult:
    """Несколько замеров, берём лучший (минимум времени)."""
    times = []
    for _ in range(repeat):
        fn()  # warmup перед каждым замером
        gc.disable()
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        gc.enable()
        times.append(t1 - t0)
    best = min(times)
    return BenchmarkResult(
        name=name, category=category, iterations=repeat,
        total_time=best, per_call_us=best * 1e6,
        gpu_verdict=gpu_verdict, notes=notes, extra=extra or {}
    )


def take(gen, limit: int = 10_000):
    """Забирает не более `limit` элементов из генератора."""
    out = []
    for i, x in enumerate(gen):
        out.append(x)
        if i + 1 >= limit:
            break
    return out


# =====================================================================
# Генераторы данных
# =====================================================================
def make_facts(n: int, arity: int = 2) -> str:
    """Генерирует n фактов вида factI(vI, vI+1, ...)."""
    lines = []
    for i in range(n):
        if arity == 1:
            lines.append(f"fact{i}(v{i}).")
        else:
            args = ", ".join([f"v{i}", f"v{i+1}"] +
                             [f"v{i+j}" for j in range(2, arity)])
            lines.append(f"fact{i}({args}).")
    return "\n".join(lines)


def make_numbered_facts(n: int, functor: str = "num") -> str:
    return "\n".join(f"{functor}({i})." for i in range(n))


def make_list_term(n: int) -> Term:
    """Строит Prolog-список [0,1,...,n-1] итеративно (без рекурсии)."""
    lst = Term("[]")
    for i in reversed(range(n)):
        lst = Term(".", [Number(i), lst])
    return lst


# =====================================================================
# Бенчмарки
# =====================================================================
class PrologBenchmark:

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.results: List[BenchmarkResult] = []

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _add(self, r: BenchmarkResult) -> None:
        self.results.append(r)
        self._log("  " + str(r))

    # -----------------------------------------------------------------
    # 1. ПАРСИНГ
    # -----------------------------------------------------------------
    def test_parsing(self) -> None:
        self._log("\n[1] Парсинг")

        small = "parent(ivan, alex). parent(alex, petr)."
        medium = make_facts(200, arity=2)
        large = make_facts(2000, arity=2)

        for label, src, n_iter in [("parse_small", small, 200),
                                    ("parse_medium", medium, 20),
                                    ("parse_large", large, 3)]:
            r = bench(label, "parsing",
                      lambda s=src: Parser(s).parse_rules(),
                      repeat=n_iter,
                      gpu_verdict="CPU_ONLY",
                      notes="regex/tokenizing, нерегулярно")
            self._add(r)

    # -----------------------------------------------------------------
    # 2. СТРОГАЯ УНИФИКАЦИЯ АТОМОВ
    # -----------------------------------------------------------------
    def test_unification_atoms(self) -> None:
        self._log("\n[2] Унификация атомов")
        n_pairs = 5000
        pairs = [(Term(f"a{i}"), Term(f"a{i}")) for i in range(n_pairs)]
        pairs += [(Term(f"a{i}"), Term(f"b{i}")) for i in range(n_pairs)]

        def run():
            for x, y in pairs:
                x.unify_strict(y, {})

        r = bench_best_of("unify_atoms_pairs", "unify", run,
                          repeat=5,
                          gpu_verdict="GPU_MAYBE",
                          notes="батчевое сравнение атомов тривиально векторизуемо")
        r.extra["pairs"] = len(pairs)
        self._add(r)

    # -----------------------------------------------------------------
    # 3. УНИФИКАЦИЯ СТРУКТУРНЫХ ТЕРМОВ
    # -----------------------------------------------------------------
    def test_unification_structures(self) -> None:
        self._log("\n[3] Унификация структурных термов")

        def make_tree(depth: int):
            if depth == 0:
                return Variable("X")
            return Term("f", [make_tree(depth - 1), Number(depth)])

        t1 = make_tree(6)
        t2 = make_tree(6)

        def run_one():
            for _ in range(500):
                t1.unify_strict(t2, {})

        r = bench_best_of("unify_struct_deep", "unify", run_one,
                          repeat=5,
                          gpu_verdict="GPU_BAD",
                          notes="глубокая рекурсия, нерегулярно")
        self._add(r)

        batch = [Term("pair", [Number(i), Number(i + 1)]) for i in range(2000)]
        batch2 = [Term("pair", [Number(i), Number(i + 1)]) for i in range(2000)]

        def run_batch():
            for x, y in zip(batch, batch2):
                x.unify_strict(y, {})

        r = bench_best_of("unify_struct_batch", "unify", run_batch,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="одинаковая форма — идеально для SIMD")
        r.extra["batch"] = len(batch)
        self._add(r)

    # -----------------------------------------------------------------
    # 4. ИНДЕКСАЦИЯ ПРАВИЛ
    # -----------------------------------------------------------------
    def test_rule_lookup(self) -> None:
        self._log("\n[4] Индексация правил / поиск кандидатов")

        for n_rules in (100, 1000, 5000):
            src = "\n".join(f"p({i}, v{i})." for i in range(n_rules))
            db = Database(Parser(src).parse_rules())
            goal = Term("p", [Number(n_rules // 2), Variable("X")])

            def run(db=db, goal=goal):
                for _ in range(200):
                    db.get_candidate_rules(goal)

            r = bench_best_of(f"lookup_{n_rules}_rules", "lookup", run,
                              repeat=5,
                              gpu_verdict="GPU_GOOD" if n_rules >= 1000 else "GPU_MAYBE",
                              notes="хэш-поиск; при больших базах — GPU hash/scatter")
            r.extra["rules"] = n_rules
            self._add(r)

    # -----------------------------------------------------------------
    # 5. NON-TAIL РЕКУРСИЯ (накопление rest_goals)
    # -----------------------------------------------------------------
    def test_recursive_tail(self) -> None:
        """НЕ хвостовая рекурсия — так исторически назван тест.

        Программа `count(N, R)` содержит накопление `R is R1 + 1` после
        рекурсивного вызова, поэтому TCO не срабатывает: `rest_goals`
        растёт с каждым уровнем, и subst тоже. Наша реализация subst
        через dict.copy() даёт O(N²) на таких программах: count(100)
        ~200 мс, count(1000) ~18 с, count(5000) — минуты.

        Держим глубины (50, 100, 200), чтобы тест завершался за
        разумное время. Настоящая хвостовая рекурсия измеряется
        отдельно в test_recursive_tail_true.
        """
        self._log("\n[5] Non-tail рекурсия (rest_goals накапливается)")
        for depth in (50, 100, 200):
            src = """
            count(0, 0).
            count(N, R) :- N > 0, N1 is N - 1, count(N1, R1), R is R1 + 1.
            """
            solver = Solver(src)
            q = f"count({depth}, R)."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"nontail_{depth}", "recursion", run,
                              repeat=3,
                              gpu_verdict="GPU_BAD",
                              notes="не TCO: после вызова стоит R is R1+1 — O(N²)")
            r.extra["depth"] = depth
            self._add(r)

    # -----------------------------------------------------------------
    # 5b. TAIL РЕКУРСИЯ (аккумулятор)
    # -----------------------------------------------------------------
    def test_recursive_tail_true(self) -> None:
        """Настоящая хвостовая рекурсия через аккумулятор.

        Рекурсивный вызов `count_acc(N1, A1, R)` — последняя цель в теле.
        TCO срабатывает (new_depth = frame.depth), стек не растёт, а
        патч 5.8 (прунинг subst) обрезает subst до переменных, реально
        используемых в новой цели. Итого O(N) по времени.
        """
        self._log("\n[5b] Tail рекурсия (аккумулятор, TCO)")
        for depth in (100, 1000, 5000):
            src = """
            count(N, R) :- count_acc(N, 0, R).
            count_acc(0, A, A).
            count_acc(N, A, R) :- N > 0, N1 is N - 1, A1 is A + 1,
                                  count_acc(N1, A1, R).
            """
            solver = Solver(src)
            q = f"count({depth}, R)."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"tail_acc_{depth}", "recursion", run,
                              repeat=3,
                              gpu_verdict="GPU_BAD",
                              notes="аккумулятор — TCO срабатывает, subst не растёт")
            r.extra["depth"] = depth
            self._add(r)

    # -----------------------------------------------------------------
    # 6. НЕХВОСТОВАЯ РЕКУРСИЯ (батч-ветвление)
    # -----------------------------------------------------------------
    def test_recursive_nontail(self) -> None:
        self._log("\n[6] Нехвостовая рекурсия (батч-ветвление)")

        src = """
        fib(0, 0).
        fib(1, 1).
        fib(N, R) :- N > 1, N1 is N - 1, N2 is N - 2,
                     fib(N1, R1), fib(N2, R2), R is R1 + R2.
        """
        solver = Solver(src)
        for n in (10, 15, 18):
            q = f"fib({n}, R)."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"fib_{n}", "recursion", run,
                              repeat=3,
                              gpu_verdict="GPU_MAYBE",
                              notes="дерево ветвлений — потенциально батчево")
            r.extra["n"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # 7. member/2
    # -----------------------------------------------------------------
    def test_builtin_member(self) -> None:
        self._log("\n[7] member/2")

        for size in (100, 500, 1000):
            lst = make_list_term(size)
            elem = Number(size - 1)  # худший случай — в конце
            goal = Term("member", [elem, lst])
            db = Database([])

            def run(db=db, goal=goal, size=size):
                take(db.builtins.execute(goal, db, {}, 0), limit=size + 10)

            try:
                r = bench_best_of(f"member_{size}", "builtin", run,
                                  repeat=3,
                                  gpu_verdict="GPU_GOOD",
                                  notes="линейный поиск — идеально для GPU")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  member_{size:<6} ПРОПУЩЕН: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 8. append/3
    # -----------------------------------------------------------------
    def test_builtin_append(self) -> None:
        self._log("\n[8] append/3")

        # append(A, B, C) с ground A, B и свободным C даёт |A|+1 решений.
        # Ограничиваем размер списков — реализация O(N²) и тяжело масштабируется.
        for size, sol_cap in ((10, 100), (30, 200), (50, 200)):
            a = make_list_term(size)
            b = make_list_term(size)
            c = Variable("C")
            goal = Term("append", [a, b, c])
            db = Database([])

            def run(db=db, goal=goal, cap=sol_cap):
                take(db.builtins.execute(goal, db, {}, 0), limit=cap)

            try:
                r = bench_best_of(f"append_{size}+{size}", "builtin", run,
                                  repeat=3,
                                  gpu_verdict="GPU_MAYBE",
                                  notes=f"перебор до {sol_cap} решений")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  append_{size}+{size:<6} ПРОПУЩЕН: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 9. length/2
    # -----------------------------------------------------------------
    def test_builtin_length(self) -> None:
        self._log("\n[9] length/2")

        # Реализация length рекурсивна и O(N²) из-за subst.copy/substitute
        # на каждом уровне — держим размеры скромными.
        for size in (100, 200, 300):
            lst = make_list_term(size)
            n = Variable("N")
            goal = Term("length", [lst, n])
            db = Database([])

            def run(db=db, goal=goal):
                take(db.builtins.execute(goal, db, {}, 0), limit=1)

            try:
                r = bench_best_of(f"length_{size}", "builtin", run,
                                  repeat=3,
                                  gpu_verdict="GPU_MAYBE",
                                  notes="O(N²) из-за substitute — сначала алгоритм, потом GPU")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  length_{size:<6} ПРОПУЩЕН: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 10. АРИФМЕТИКА
    # -----------------------------------------------------------------
    def test_arithmetic(self) -> None:
        self._log("\n[10] Арифметика")

        def make_expr(n_ops: int) -> Term:
            expr = Number(1)
            for i in range(n_ops):
                op = ['+', '-', '*'][i % 3]
                expr = Term(op, [expr, Number(i + 2)])
            return expr

        for n_ops in (10, 100, 1000):
            expr = make_expr(n_ops)

            def run(expr=expr):
                for _ in range(50):
                    evaluate_expression(expr, {})

            r = bench_best_of(f"arith_{n_ops}ops", "arith", run,
                              repeat=5,
                              gpu_verdict="GPU_GOOD",
                              notes="регулярное дерево — компилируется в tf.Graph")
            r.extra["ops"] = n_ops
            self._add(r)

        exprs = [Term('+', [Number(i), Number(i * 2)]) for i in range(5000)]

        def run_batch():
            for e in exprs:
                evaluate_expression(e, {})

        r = bench_best_of("arith_batch_5000", "arith", run_batch,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="5000 независимых выражений — батч")
        self._add(r)

    # -----------------------------------------------------------------
    # 11. findall/3
    # -----------------------------------------------------------------
    def test_findall(self) -> None:
        self._log("\n[11] findall/3")

        for n in (50, 200, 500):
            src = make_numbered_facts(n, "num")
            solver = Solver(src)
            q = "findall(X, num(X), L)."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"findall_{n}", "builtin", run,
                              repeat=3,
                              gpu_verdict="GPU_GOOD",
                              notes="сбор всех решений — батчевая агрегация")
            r.extra["n"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # 12. СЕМАНТИЧЕСКАЯ УНИФИКАЦИЯ
    # -----------------------------------------------------------------
    def test_semantic_unification(self) -> None:
        self._log("\n[12] Семантическая унификация (без реальной модели)")

        import numpy as np

        def fake_embed(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            return rng.standard_normal(384).astype(np.float32)

        try:
            from prolog.embedding_unify import EmbeddingUnifier
        except Exception as e:
            self._log(f"  пропуск: {e}")
            return

        unifier = EmbeddingUnifier(fake_embed, threshold=0.8)

        pairs = [(Term(f"word{i}"), Term(f"word{i + 100000}"))
                 for i in range(2000)]

        def run_unify():
            for a, b in pairs:
                unifier._semantic_match(a, b)

        r = bench_best_of("semantic_match_2000pairs", "semantic", run_unify,
                          repeat=3,
                          gpu_verdict="GPU_GOOD",
                          notes="косинус 2000×384 — прямая матричная операция")
        r.extra["pairs"] = len(pairs)
        self._add(r)

        V = np.random.randn(2000, 384).astype(np.float32)

        def run_cos():
            V @ V.T

        r = bench_best_of("cosine_matrix_2000x384", "semantic", run_cos,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="чистый BLAS — на GPU 10-50× быстрее")
        self._add(r)

    # -----------------------------------------------------------------
    # 13. БОЛЬШАЯ БАЗА ФАКТОВ
    # -----------------------------------------------------------------
    def test_large_fact_base(self) -> None:
        self._log("\n[13] Большая база фактов")

        for n in (500, 2000, 8000):
            src = make_numbered_facts(n, "num")
            solver = Solver(src)
            q = f"num({n // 2})."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"query_fact_{n}", "lookup", run,
                              repeat=3,
                              gpu_verdict="GPU_GOOD" if n >= 2000 else "GPU_MAYBE",
                              notes="поиск по индексу; GPU hash/scatter")
            r.extra["facts"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # Запуск всего
    # -----------------------------------------------------------------
    def run_all(self) -> List[BenchmarkResult]:
        tests = [
            self.test_parsing,
            self.test_unification_atoms,
            self.test_unification_structures,
            self.test_rule_lookup,
            self.test_recursive_tail,        # non-tail (переименован)
            self.test_recursive_tail_true,   # настоящая TCO
            self.test_recursive_nontail,
            self.test_builtin_member,
            self.test_builtin_append,
            self.test_builtin_length,
            self.test_arithmetic,
            self.test_findall,
            self.test_semantic_unification,
            self.test_large_fact_base,
        ]
        for t in tests:
            try:
                t()
            except Exception as e:
                self._log(f"\n!! Тест {t.__name__} упал: "
                          f"{type(e).__name__}: {e}")
        return self.results

    # -----------------------------------------------------------------
    # Отчёт
    # -----------------------------------------------------------------
    def report(self) -> None:
        print("\n" + "=" * 100)
        print("ОТЧЁТ: кандидаты на GPU-ускорение")
        print("=" * 100)

        by_verdict: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_verdict.setdefault(r.gpu_verdict, []).append(r)

        order = ["GPU_GOOD", "GPU_MAYBE", "GPU_BAD", "CPU_ONLY"]
        titles = {
            "GPU_GOOD":  "СИЛЬНЫЕ КАНДИДАТЫ (ожидается 5-50x)",
            "GPU_MAYBE": "ПОТЕНЦИАЛЬНЫЕ (зависит от объёма данных)",
            "GPU_BAD":   "СЛАБЫЕ (нерегулярные, но возможны батчи)",
            "CPU_ONLY":  "НЕ КАНДИДАТЫ (символьная/последовательная логика)",
        }

        for v in order:
            items = by_verdict.get(v, [])
            if not items:
                continue
            print(f"\n{titles[v]}")
            print("-" * 100)
            for r in sorted(items, key=lambda x: -x.per_call_us):
                print(f"  {r}")

        print("\n" + "=" * 100)
        print("ТОП-10 самых «горячих» операций (по убыванию времени на вызов):")
        print("=" * 100)
        for r in sorted(self.results, key=lambda x: -x.per_call_us)[:10]:
            print(f"  {r.name:<32} {r.per_call_us:>12.2f} µs   {r.gpu_verdict}")

        total_good = sum(r.per_call_us for r in self.results
                         if r.gpu_verdict == "GPU_GOOD")
        total_all = sum(r.per_call_us for r in self.results)
        if total_all > 0:
            print(f"\nДоля времени, приходящаяся на GPU_GOOD-операции: "
                  f"{100 * total_good / total_all:.1f}%")


# =====================================================================
# main
# =====================================================================
def main():
    print("=" * 100)
    print("PyAgentProlog GPU-candidate benchmark")
    print("=" * 100)

    bm = PrologBenchmark(verbose=True)
    bm.run_all()
    bm.report()


if __name__ == "__main__":
    main()