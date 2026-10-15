"""prolog_benchmark.py

Benchmark to identify components of PyAgentProlog that are worth
moving to the GPU (NumPy/CuPy/TensorFlow).

Classification of candidates:
    GPU_GOOD      — regular batched operations with large data volumes
    GPU_MAYBE     — vectorizable, but the volume may be small
    GPU_BAD       — irregular / sequential / tiny operations
    CPU_ONLY      — symbolic computation without SIMD potential

Patch A (benchmark):
  * Test [5] "tail_recursion" was renamed to "non-tail" and limited
    to depths (50, 100, 200). The program `count(N, R)` contains
    accumulation `R is R1 + 1` after the recursive call, so TCO
    does not fire and subst grows linearly with depth. Our subst
    via dict.copy() gives O(N²) — at N=1000 that is ~18 seconds,
    at N=5000 — minutes. The test is now honestly labeled and runs fast.
  * Added test [5b] "tail_acc" — real tail recursion with an
    accumulator. After the recursive call there are no other goals,
    TCO fires, subst does not grow. Expected O(N) instead of O(N²).

(c) 2026 Andrey Vorfolomeev
"""

from __future__ import annotations

import gc
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
sys.path.append('d:/BFX/Repositories/')

# Raise the recursion limit BEFORE importing prolog — some built-in
# predicates (member/length/append) are implemented as recursive generators.
sys.setrecursionlimit(100_000)

# ---------------------------------------------------------------------
# Imports of your Prolog
# ---------------------------------------------------------------------
from prolog.parser import Parser
from prolog.terms import Term, Variable, Number, Rule  # noqa: F401
from prolog.interpreter import Database
from prolog.solver import Solver
from prolog.predicates import evaluate_expression


# =====================================================================
# Timing
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
    """Several runs, take the best (minimum time)."""
    times = []
    for _ in range(repeat):
        fn()  # warmup before each run
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
    """Consume at most `limit` elements from the generator."""
    out = []
    for i, x in enumerate(gen):
        out.append(x)
        if i + 1 >= limit:
            break
    return out


# =====================================================================
# Data generators
# =====================================================================
def make_facts(n: int, arity: int = 2) -> str:
    """Generates n facts of the form factI(vI, vI+1, ...)."""
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
    """Builds a Prolog list [0,1,...,n-1] iteratively (no recursion)."""
    lst = Term("[]")
    for i in reversed(range(n)):
        lst = Term(".", [Number(i), lst])
    return lst


# =====================================================================
# Benchmarks
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
    # 1. PARSING
    # -----------------------------------------------------------------
    def test_parsing(self) -> None:
        self._log("\n[1] Parsing")

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
                      notes="regex/tokenizing, irregular")
            self._add(r)

    # -----------------------------------------------------------------
    # 2. STRICT ATOM UNIFICATION
    # -----------------------------------------------------------------
    def test_unification_atoms(self) -> None:
        self._log("\n[2] Atom unification")
        n_pairs = 5000
        pairs = [(Term(f"a{i}"), Term(f"a{i}")) for i in range(n_pairs)]
        pairs += [(Term(f"a{i}"), Term(f"b{i}")) for i in range(n_pairs)]

        def run():
            for x, y in pairs:
                x.unify_strict(y, {})

        r = bench_best_of("unify_atoms_pairs", "unify", run,
                          repeat=5,
                          gpu_verdict="GPU_MAYBE",
                          notes="batched atom comparison is trivially vectorizable")
        r.extra["pairs"] = len(pairs)
        self._add(r)

    # -----------------------------------------------------------------
    # 3. UNIFICATION OF STRUCTURAL TERMS
    # -----------------------------------------------------------------
    def test_unification_structures(self) -> None:
        self._log("\n[3] Unification of structural terms")

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
                          notes="deep recursion, irregular")
        self._add(r)

        batch = [Term("pair", [Number(i), Number(i + 1)]) for i in range(2000)]
        batch2 = [Term("pair", [Number(i), Number(i + 1)]) for i in range(2000)]

        def run_batch():
            for x, y in zip(batch, batch2):
                x.unify_strict(y, {})

        r = bench_best_of("unify_struct_batch", "unify", run_batch,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="uniform shape — ideal for SIMD")
        r.extra["batch"] = len(batch)
        self._add(r)

    # -----------------------------------------------------------------
    # 4. RULE INDEXING
    # -----------------------------------------------------------------
    def test_rule_lookup(self) -> None:
        self._log("\n[4] Rule indexing / candidate lookup")

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
                              notes="hash lookup; for large bases — GPU hash/scatter")
            r.extra["rules"] = n_rules
            self._add(r)

    # -----------------------------------------------------------------
    # 5. NON-TAIL RECURSION (rest_goals accumulates)
    # -----------------------------------------------------------------
    def test_recursive_tail(self) -> None:
        """NOT tail recursion — that's how the test was historically named.

        The program `count(N, R)` contains accumulation `R is R1 + 1` after
        the recursive call, so TCO does not fire: `rest_goals` grows with
        each level, and subst does too. Our subst implementation via
        dict.copy() gives O(N²) on such programs: count(100)
        ~200 ms, count(1000) ~18 s, count(5000) — minutes.

        We keep depths (50, 100, 200) so the test finishes within
        reasonable time. True tail recursion is measured separately
        in test_recursive_tail_true.
        """
        self._log("\n[5] Non-tail recursion (rest_goals accumulates)")
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
                              notes="not TCO: R is R1+1 comes after the call — O(N²)")
            r.extra["depth"] = depth
            self._add(r)

    # -----------------------------------------------------------------
    # 5b. TAIL RECURSION (accumulator)
    # -----------------------------------------------------------------
    def test_recursive_tail_true(self) -> None:
        """Real tail recursion via an accumulator.

        The recursive call `count_acc(N1, A1, R)` is the last goal in the
        body. TCO fires (new_depth = frame.depth), the stack does not grow,
        and patch 5.8 (subst pruning) trims subst to the variables actually
        used by the new goal. Total is O(N) in time.
        """
        self._log("\n[5b] Tail recursion (accumulator, TCO)")
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
                              notes="accumulator — TCO fires, subst does not grow")
            r.extra["depth"] = depth
            self._add(r)

    # -----------------------------------------------------------------
    # 6. NON-TAIL RECURSION (batch branching)
    # -----------------------------------------------------------------
    def test_recursive_nontail(self) -> None:
        self._log("\n[6] Non-tail recursion (batch branching)")

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
                              notes="branching tree — potentially batchable")
            r.extra["n"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # 7. member/2
    # -----------------------------------------------------------------
    def test_builtin_member(self) -> None:
        self._log("\n[7] member/2")

        for size in (100, 500, 1000):
            lst = make_list_term(size)
            elem = Number(size - 1)  # worst case — at the end
            goal = Term("member", [elem, lst])
            db = Database([])

            def run(db=db, goal=goal, size=size):
                take(db.builtins.execute(goal, db, {}, 0), limit=size + 10)

            try:
                r = bench_best_of(f"member_{size}", "builtin", run,
                                  repeat=3,
                                  gpu_verdict="GPU_GOOD",
                                  notes="linear search — ideal for GPU")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  member_{size:<6} SKIPPED: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 8. append/3
    # -----------------------------------------------------------------
    def test_builtin_append(self) -> None:
        self._log("\n[8] append/3")

        # append(A, B, C) with ground A, B and a free C yields |A|+1 solutions.
        # Keep the list sizes modest — the implementation is O(N²) and scales poorly.
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
                                  notes=f"enumeration up to {sol_cap} solutions")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  append_{size}+{size:<6} SKIPPED: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 9. length/2
    # -----------------------------------------------------------------
    def test_builtin_length(self) -> None:
        self._log("\n[9] length/2")

        # The length implementation is recursive and O(N²) due to
        # subst.copy/substitute at each level — keep sizes modest.
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
                                  notes="O(N²) due to substitute — algorithm first, then GPU")
                r.extra["size"] = size
                self._add(r)
            except RecursionError as e:
                self._log(f"  length_{size:<6} SKIPPED: RecursionError: {e}")

    # -----------------------------------------------------------------
    # 10. ARITHMETIC
    # -----------------------------------------------------------------
    def test_arithmetic(self) -> None:
        self._log("\n[10] Arithmetic")

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
                              notes="regular tree — compiles into tf.Graph")
            r.extra["ops"] = n_ops
            self._add(r)

        exprs = [Term('+', [Number(i), Number(i * 2)]) for i in range(5000)]

        def run_batch():
            for e in exprs:
                evaluate_expression(e, {})

        r = bench_best_of("arith_batch_5000", "arith", run_batch,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="5000 independent expressions — a batch")
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
                              notes="collecting all solutions — batch aggregation")
            r.extra["n"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # 12. SEMANTIC UNIFICATION
    # -----------------------------------------------------------------
    def test_semantic_unification(self) -> None:
        self._log("\n[12] Semantic unification (without a real model)")

        import numpy as np

        def fake_embed(text: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
            return rng.standard_normal(384).astype(np.float32)

        try:
            from prolog.embedding_unify import EmbeddingUnifier
        except Exception as e:
            self._log(f"  skipping: {e}")
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
                          notes="cosine 2000×384 — a direct matrix operation")
        r.extra["pairs"] = len(pairs)
        self._add(r)

        V = np.random.randn(2000, 384).astype(np.float32)

        def run_cos():
            V @ V.T

        r = bench_best_of("cosine_matrix_2000x384", "semantic", run_cos,
                          repeat=5,
                          gpu_verdict="GPU_GOOD",
                          notes="pure BLAS — 10-50× faster on GPU")
        self._add(r)

    # -----------------------------------------------------------------
    # 13. LARGE FACT BASE
    # -----------------------------------------------------------------
    def test_large_fact_base(self) -> None:
        self._log("\n[13] Large fact base")

        for n in (500, 2000, 8000):
            src = make_numbered_facts(n, "num")
            solver = Solver(src)
            q = f"num({n // 2})."

            def run(s=solver, q=q):
                s.find_solutions(q)

            r = bench_best_of(f"query_fact_{n}", "lookup", run,
                              repeat=3,
                              gpu_verdict="GPU_GOOD" if n >= 2000 else "GPU_MAYBE",
                              notes="index lookup; GPU hash/scatter")
            r.extra["facts"] = n
            self._add(r)

    # -----------------------------------------------------------------
    # Run everything
    # -----------------------------------------------------------------
    def run_all(self) -> List[BenchmarkResult]:
        tests = [
            self.test_parsing,
            self.test_unification_atoms,
            self.test_unification_structures,
            self.test_rule_lookup,
            self.test_recursive_tail,        # non-tail (renamed)
            self.test_recursive_tail_true,   # real TCO
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
                self._log(f"\n!! Test {t.__name__} crashed: "
                          f"{type(e).__name__}: {e}")
        return self.results

    # -----------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------
    def report(self) -> None:
        print("\n" + "=" * 100)
        print("REPORT: GPU-acceleration candidates")
        print("=" * 100)

        by_verdict: Dict[str, List[BenchmarkResult]] = {}
        for r in self.results:
            by_verdict.setdefault(r.gpu_verdict, []).append(r)

        order = ["GPU_GOOD", "GPU_MAYBE", "GPU_BAD", "CPU_ONLY"]
        titles = {
            "GPU_GOOD":  "STRONG CANDIDATES (5-50x expected)",
            "GPU_MAYBE": "POTENTIAL (depends on data volume)",
            "GPU_BAD":   "WEAK (irregular, but batches may be possible)",
            "CPU_ONLY":  "NOT CANDIDATES (symbolic/sequential logic)",
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
        print("TOP-10 hottest operations (sorted by decreasing per-call time):")
        print("=" * 100)
        for r in sorted(self.results, key=lambda x: -x.per_call_us)[:10]:
            print(f"  {r.name:<32} {r.per_call_us:>12.2f} µs   {r.gpu_verdict}")

        total_good = sum(r.per_call_us for r in self.results
                         if r.gpu_verdict == "GPU_GOOD")
        total_all = sum(r.per_call_us for r in self.results)
        if total_all > 0:
            print(f"\nShare of time spent in GPU_GOOD operations: "
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