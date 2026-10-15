""" PyAgentProlog

Interpreter with tracing, cut, and tail-call optimization support.

Patch 5.4: if-then-else via _IfThenBarrier without nested _solve.
Patch 5.5: removed _rename_variables cache (broke TCO).
Patch 5.6: fixed the constant-index bug (Variable as a key).
Patch 5.7: removed redundant subst.copy() in the hot path.
Patch 5.8 (this file): subst pruning on every rule application.
  Before creating body_frame, _prune_subst_to_goals is called:
  only bindings for variables reachable from the new goals
  and from the "anchor" variables (query vars + keys of initial_subst)
  are kept. Chains v -> w -> x -> 42 are collapsed into v -> 42. This makes
  subst bounded in size regardless of recursion depth, so
  subst.copy() inside unify becomes O(1) instead of O(N). Total
  time of tail and non-tail recursion drops from O(N²) to O(N).
  alt_frame uses the original frame.subst — pruning does not touch it.

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""

from typing import Generator, Dict, List, Optional, Tuple, Any, Iterator
from prolog.tracer import ProofTracer
from prolog.predicates import STANDARD_PREDICATES, Predicate, CutException, PrologError, RepeatException
from prolog.terms import Term, Variable, Number, Rule

DEBUG = False


def debug_print(*args, **kwargs):
    if DEBUG:
        print("[DEBUG]", *args, **kwargs)


class _IfThenBarrier:
    """State of the `(Cond -> Then ; Else)` construct (or `Cond -> Then`)."""

    __slots__ = ('c_id', 'then_goal', 'rest_goals', 'parent_cut_id',
                 'parent_depth', 'parent_ifthen_meta', 'else_frame')

    def __init__(self, c_id, then_goal, rest_goals, parent_cut_id,
                 parent_depth, parent_ifthen_meta, else_frame):
        self.c_id = c_id
        self.then_goal = then_goal
        self.rest_goals = rest_goals
        self.parent_cut_id = parent_cut_id
        self.parent_depth = parent_depth
        self.parent_ifthen_meta = parent_ifthen_meta
        self.else_frame = else_frame


class Frame:
    """Stack frame for iterative search."""
    __slots__ = ('goals', 'subst', 'depth', 'rule_index', 'total_rules',
                 'tail', 'trace_step', 'cut_id', 'disj_step', 'ifthen_meta')

    def __init__(self, goals, subst, depth, rule_index, total_rules, tail,
                 trace_step, cut_id=None, disj_step=None, ifthen_meta=None):
        self.goals = goals
        self.subst = subst
        self.depth = depth
        self.rule_index = rule_index
        self.total_rules = total_rules
        self.tail = tail
        self.trace_step = trace_step
        self.cut_id = cut_id
        self.disj_step = disj_step
        self.ifthen_meta = ifthen_meta


class BuiltinRegistry:
    """Registry of built-in predicates with arity support."""

    __slots__ = ('builtins', 'custom_predicates')

    def __init__(self):
        self.builtins = {}
        self.custom_predicates = {}
        self._register_standard_predicates()

    def _register_standard_predicates(self):
        for name, arity_dict in STANDARD_PREDICATES.items():
            for arity, func in arity_dict.items():
                self.register(Predicate(name, arity, func))

    def register(self, predicate: Predicate):
        name = predicate.name
        arity = predicate.arity
        if name not in self.builtins:
            self.builtins[name] = {}
        self.builtins[name][arity] = predicate

    def register_custom(self, name: str, arity: int, func):
        predicate = Predicate(name, arity, func)
        self.custom_predicates[name] = predicate
        self.register(predicate)

    def register_module(self, module):
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, staticmethod):
                func = attr.__func__
            else:
                func = attr
            if callable(func) and hasattr(func, '_is_prolog_predicate'):
                name = getattr(func, '_predicate_name', attr_name)
                arity = getattr(func, '_predicate_arity', 0)
                self.register_custom(name, arity, func)

    def is_builtin(self, name):
        return name in self.builtins

    def get_predicate(self, name, arity):
        by_arity = self.builtins.get(name)
        if by_arity:
            return by_arity.get(arity)
        return None

    def execute(self, goal, database, subst, depth):
        predicate = self.get_predicate(goal.functor, len(goal.arguments))
        if predicate:
            return predicate.execute(goal, database, subst, depth)
        if goal.functor in self.builtins:
            print(f"[WARNING] Wrong arity for predicate {goal.functor}: "
                  f"expected one of {list(self.builtins[goal.functor].keys())}, "
                  f"got {len(goal.arguments)}")
        return []


class Database:
    """Prolog knowledge base with tracing, rule indexing, and tail-call optimization support."""

    __slots__ = ('rules', 'builtins', 'tracer', 'context', 'rule_index',
                 '_cut_counter', '_renamed_cache', '_rename_nonce')

    def __init__(self, rules, tracer: Optional[ProofTracer] = None, context=None):
        self.rules = rules
        self.builtins = BuiltinRegistry()
        self.tracer = tracer or ProofTracer()
        self.context = context
        self.rule_index = None
        self._cut_counter = 0
        self._renamed_cache = {}
        self._rename_nonce = 0
        self.build_rule_index()

    # ------------------------------------------------------------------
    # Constant index of rules by the first argument.
    # Patch 5.6: Variable is a subclass of Term but is NOT a constant.
    # ------------------------------------------------------------------
    @staticmethod
    def _is_const_first_arg(term):
        if isinstance(term, Variable):
            return False
        if isinstance(term, Number):
            return True
        if isinstance(term, Term) and not term.arguments:
            return True
        return False

    def build_rule_index(self):
        index = {}
        for rule in self.rules:
            head = rule.head
            key = (head.functor, len(head.arguments))
            if key not in index:
                index[key] = {'all': [], 'const': {}}
            entry = index[key]
            entry['all'].append(rule)
            if head.arguments:
                first_arg = head.arguments[0]
                if self._is_const_first_arg(first_arg):
                    const_val = (first_arg.value if isinstance(first_arg, Number)
                                 else first_arg.functor)
                    if const_val not in entry['const']:
                        entry['const'][const_val] = []
                    entry['const'][const_val].append(rule)
        self.rule_index = index

    def get_candidate_rules(self, goal):
        key = (goal.functor, len(goal.arguments))
        entry = self.rule_index.get(key)
        if not entry:
            return []
        if goal.arguments:
            first_arg = goal.arguments[0]
            if self._is_const_first_arg(first_arg):
                const_val = (first_arg.value if isinstance(first_arg, Number)
                             else first_arg.functor)
                bucket = entry['const'].get(const_val)
                if bucket is not None:
                    return list(bucket)
        return list(entry['all'])

    def register_custom_predicate(self, name: str, arity: int, func):
        self.builtins.register_custom(name, arity, func)

    def register_predicate_module(self, module):
        self.builtins.register_module(module)

    def query_with_explanation(self, goal) -> Tuple[List, ProofTracer]:
        self.tracer.clear()
        solutions = []
        for subst in self._solve_with_trace(goal, self.rule_index, {}, 0):
            solutions.append(goal.substitute(subst))
        return solutions, self.tracer

    def query(self, goal):
        for subst in self._solve(goal, self.rule_index, {}, 0):
            yield goal.substitute(subst)

    def _solve(self, goal, rule_index, subst, depth):
        yield from self._solve_iterative(goal, subst, depth, with_trace=False)

    def _solve_with_trace(self, goal, rule_index, subst, depth):
        yield from self._solve_iterative(goal, subst, depth, with_trace=True)

    def _close_successful_steps(self, stack, with_trace):
        if not with_trace:
            return
        for frame in stack:
            if frame.trace_step and not frame.trace_step.success:
                frame.trace_step.success = True
                self.tracer.end_goal(True)
            if frame.disj_step and not frame.disj_step.success:
                frame.disj_step.success = True
                self.tracer.end_goal(True)

    # ------------------------------------------------------------------
    # PATCH 5.8: pruning subst to relevant variables.
    # ------------------------------------------------------------------
    def _prune_subst_to_goals(self, subst, goals, keep_vars):
        """Returns a new dict with bindings for variables reachable
        from `goals` or from `keep_vars`, with chains resolved.

        A chain `v -> w -> x -> 42` in the original subst, if v is reachable,
        becomes `v -> 42` in the result. Intermediate w, x are not
        kept unless they are reachable from the seed themselves.

        Guarantee: any variable occurring in `goals`, as well as
        every variable from `keep_vars`, gets its (resolved)
        binding if it was in the original subst.
        """
        if not subst:
            return {}

        # 1) Seed = all Variables from goals + keep_vars.
        seed = set()
        stack = list(goals)
        while stack:
            t = stack.pop()
            if isinstance(t, Variable):
                if t.name != "_":
                    seed.add(t)
            elif isinstance(t, Term):
                stack.extend(t.arguments)
        for v in keep_vars:
            if isinstance(v, Variable) and v.name != "_":
                seed.add(v)

        # 2) Closure with chain resolution and cycle protection.
        pruned = {}
        work = list(seed)
        while work:
            v = work.pop()
            if v in pruned:
                continue
            if v not in subst:
                continue
            val = subst[v]
            # Resolve the chain v -> w -> ... -> final.
            chain_seen = {v}
            while isinstance(val, Variable) and val in subst:
                if val in chain_seen:
                    # Cycle v -> ... -> v: break to avoid looping.
                    break
                chain_seen.add(val)
                val = subst[val]
            pruned[v] = val
            # Process variables inside val (if val is a Term or
            # remains a Variable without a binding).
            inner = [val]
            while inner:
                t = inner.pop()
                if isinstance(t, Variable):
                    if t.name != "_" and t not in pruned:
                        work.append(t)
                elif isinstance(t, Term):
                    inner.extend(t.arguments)
        return pruned

    # ------------------------------------------------------------------
    # Main iterative solver
    # ------------------------------------------------------------------
    def _solve_iterative(self, initial_goal: Term, initial_subst: Dict, depth: int, with_trace: bool):
        debug_print(f"\n=== Beginning search for goal: {initial_goal} "
                    f"with substitution {initial_subst}, depth {depth} ===")

        # --- Patch 5.8: anchor variables for pruning ---
        # These are variables that must be preserved in subst at any
        # moment, because the caller expects them in the yields.
        # = all Variables from initial_goal + all keys of initial_subst.
        _qv_seed = set()
        _qv_stack = [initial_goal]
        while _qv_stack:
            _t = _qv_stack.pop()
            if isinstance(_t, Variable):
                if _t.name != "_":
                    _qv_seed.add(_t)
            elif isinstance(_t, Term):
                _qv_stack.extend(_t.arguments)
        for _k in initial_subst:
            if isinstance(_k, Variable) and _k.name != "_":
                _qv_seed.add(_k)
        query_vars = frozenset(_qv_seed)

        root_step = None
        if with_trace:
            root_step = self.tracer.start_goal(
                str(initial_goal.substitute(initial_subst)), depth)
            root_step.rule_used = "query"

        stack = [Frame([initial_goal], initial_subst, depth, 0, 0, False,
                       root_step, None)]
        if stack[0].cut_id is None:
            stack[0].cut_id = self._cut_counter
            self._cut_counter += 1

        pending_disj_steps = []
        found_solution = False

        while stack:
            frame = stack[-1]

            debug_print(f"\n--- Current frame (depth {frame.depth}) ---")
            debug_print(f"  goals: {[str(g) for g in frame.goals]}")
            debug_print(f"  subst: {frame.subst}")
            debug_print(f"  rule_index: {frame.rule_index}, total_rules: {frame.total_rules}")

            # ------------------------------------------------------------------
            # All goals solved
            # ------------------------------------------------------------------
            if not frame.goals:
                if (frame.ifthen_meta is not None
                        and frame.cut_id == frame.ifthen_meta.c_id):
                    barrier = frame.ifthen_meta
                    debug_print("  if-then-else: Cond succeeded, commit to Then")

                    if with_trace and frame.trace_step and not frame.trace_step.success:
                        frame.trace_step.success = True
                        self.tracer.end_goal(True)

                    else_frame = barrier.else_frame
                    while stack and stack[-1] is not else_frame:
                        f = stack.pop()
                        if with_trace and f.trace_step and not f.trace_step.success:
                            f.trace_step.success = False
                            self.tracer.end_goal(False)
                    if stack and stack[-1] is else_frame:
                        f = stack.pop()
                        if with_trace and f.trace_step and not f.trace_step.success:
                            f.trace_step.success = False
                            self.tracer.end_goal(False)

                    new_frame = Frame(
                        [barrier.then_goal] + barrier.rest_goals,
                        frame.subst,
                        barrier.parent_depth,
                        0, 0,
                        frame.tail,
                        None,
                        barrier.parent_cut_id,
                        None,
                        barrier.parent_ifthen_meta,
                    )
                    stack.append(new_frame)
                    continue

                debug_print("  All goals solved, yielding a solution")
                self._close_successful_steps(stack, with_trace)
                stack.pop()
                if with_trace and frame.trace_step:
                    if not frame.trace_step.success:
                        frame.trace_step.success = True
                        self.tracer.end_goal(True)
                if with_trace and frame.disj_step and not frame.disj_step.success:
                    frame.disj_step.success = True
                    self.tracer.end_goal(True)
                    if frame.disj_step in pending_disj_steps:
                        pending_disj_steps.remove(frame.disj_step)
                found_solution = True
                yield frame.subst
                continue

            current_goal = frame.goals[0]
            rest_goals = frame.goals[1:]

            debug_print(f"  Current goal: {current_goal}")

            # --- Built-in predicates ---
            if self.builtins.is_builtin(current_goal.functor):
                debug_print(f"  Built-in predicate: {current_goal.functor}")
                if with_trace:
                    builtin_step = self.tracer.start_goal(
                        str(current_goal.substitute(frame.subst)), frame.depth)
                    builtin_step.builtin = True
                else:
                    builtin_step = None

                try:
                    result = self.builtins.execute(current_goal, self,
                                                   frame.subst, frame.depth)
                except CutException as e:
                    debug_print("  Cut (!) fired")
                    if with_trace and builtin_step:
                        builtin_step.success = True
                        self.tracer.end_goal(True)
                    self._close_successful_steps(stack, with_trace)
                    if len(stack) >= 2:
                        parent = stack[-2]
                        parent.rule_index = parent.total_rules
                    current_cut_id = frame.cut_id
                    if current_cut_id is not None:
                        new_stack = []
                        for i, f in enumerate(stack):
                            if i == len(stack) - 1 or f.cut_id != current_cut_id:
                                new_stack.append(f)
                        stack[:] = new_stack
                        frame = stack[-1]
                    frame.goals = rest_goals
                    frame.subst = e.subst
                    continue
                except RepeatException as e:
                    debug_print("  RepeatException: adding alternative frame")
                    stack.pop()
                    continue_frame = Frame(rest_goals, e.subst, frame.depth,
                                           0, 0, frame.tail, None,
                                           frame.cut_id, frame.disj_step,
                                           frame.ifthen_meta)
                    repeat_frame = Frame([current_goal] + rest_goals, e.subst,
                                         frame.depth, 0, 0, frame.tail, None,
                                         frame.cut_id, frame.disj_step,
                                         frame.ifthen_meta)
                    stack.append(repeat_frame)
                    stack.append(continue_frame)
                    if with_trace and builtin_step:
                        builtin_step.success = True
                        self.tracer.end_goal(True)
                    continue
                except PrologError:
                    debug_print("  Execution error in built-in predicate")
                    if with_trace and builtin_step:
                        builtin_step.success = False
                        self.tracer.end_goal(False)
                    stack.pop()
                    continue

                if isinstance(result, list):
                    if result:
                        first_subst = result[0]
                        extras = result[1:]
                        stack.pop()
                        for extra_subst in reversed(extras):
                            alt_frame = Frame(rest_goals, extra_subst,
                                              frame.depth, 0, 0, frame.tail,
                                              None, frame.cut_id,
                                              frame.disj_step, frame.ifthen_meta)
                            stack.append(alt_frame)
                        frame.goals = rest_goals
                        frame.subst = first_subst
                        stack.append(frame)
                        if with_trace and builtin_step:
                            builtin_step.success = True
                            self.tracer.end_goal(True)
                        continue
                    else:
                        debug_print("  No solutions for built-in predicate, backtracking")
                        if with_trace and builtin_step:
                            builtin_step.success = False
                            self.tracer.end_goal(False)
                        stack.pop()
                        continue
                else:
                    try:
                        it = iter(result)
                        first_subst = next(it)
                    except StopIteration:
                        debug_print("  No solutions for built-in predicate (empty generator), backtracking")
                        if with_trace and builtin_step:
                            builtin_step.success = False
                            self.tracer.end_goal(False)
                        stack.pop()
                        continue
                    except CutException as e:
                        debug_print("  Cut (!) fired inside the generator")
                        if with_trace and builtin_step:
                            builtin_step.success = True
                            self.tracer.end_goal(True)
                        self._close_successful_steps(stack, with_trace)
                        if len(stack) >= 2:
                            parent = stack[-2]
                            parent.rule_index = parent.total_rules
                        current_cut_id = frame.cut_id
                        if current_cut_id is not None:
                            new_stack = []
                            for i, f in enumerate(stack):
                                if i == len(stack) - 1 or f.cut_id != current_cut_id:
                                    new_stack.append(f)
                            stack[:] = new_stack
                            frame = stack[-1]
                        frame.goals = rest_goals
                        frame.subst = e.subst
                        continue
                    except RepeatException as e:
                        debug_print("  RepeatException inside generator")
                        stack.pop()
                        continue_frame = Frame(rest_goals, e.subst, frame.depth,
                                               0, 0, frame.tail, None,
                                               frame.cut_id, frame.disj_step,
                                               frame.ifthen_meta)
                        repeat_frame = Frame([current_goal] + rest_goals, e.subst,
                                             frame.depth, 0, 0, frame.tail,
                                             None, frame.cut_id, frame.disj_step,
                                             frame.ifthen_meta)
                        stack.append(repeat_frame)
                        stack.append(continue_frame)
                        if with_trace and builtin_step:
                            builtin_step.success = True
                            self.tracer.end_goal(True)
                        continue
                    except PrologError:
                        debug_print("  PrologError while iterating generator -> fail branch")
                        if with_trace and builtin_step:
                            builtin_step.success = False
                            self.tracer.end_goal(False)
                        stack.pop()
                        continue
                    except Exception as e:
                        import traceback
                        debug_print(
                            f"  Unexpected failure while iterating generator: "
                            f"{type(e).__name__}: {e}"
                        )
                        print(f"[_solve_iterative] unexpected {type(e).__name__} "
                              f"in Goal={current_goal}: {e}")
                        traceback.print_exc()
                        if with_trace and builtin_step:
                            builtin_step.success = False
                            self.tracer.end_goal(False)
                        raise

                    frame.goals = rest_goals
                    frame.subst = first_subst
                    extras = list(it)
                    stack.pop()
                    for extra_subst in reversed(extras):
                        alt_frame = Frame(rest_goals, extra_subst, frame.depth,
                                          0, 0, frame.tail, None,
                                          frame.cut_id, frame.disj_step,
                                          frame.ifthen_meta)
                        stack.append(alt_frame)
                    stack.append(frame)
                    if with_trace and builtin_step:
                        builtin_step.success = True
                        self.tracer.end_goal(True)
                    continue

            # --- Conjunction ---
            if (isinstance(current_goal, Term) and current_goal.functor == ','
                    and len(current_goal.arguments) == 2):
                debug_print("  Handling conjunction ','")
                flat_goals = self._flatten_conjunction(current_goal)
                debug_print(f"    Flattened conjunction: {[str(g) for g in flat_goals]}")
                frame.goals = flat_goals + rest_goals
                continue

            # --- Disjunction ---
            if (isinstance(current_goal, Term) and current_goal.functor == ';'
                    and len(current_goal.arguments) == 2):
                left, right = current_goal.arguments

                # ---- if-then-else ----
                if (isinstance(left, Term) and left.functor == '->'
                        and len(left.arguments) == 2):
                    debug_print("  Handling (Cond -> Then ; Else)")
                    cond, then = left.arguments

                    cond_cut_id = self._cut_counter
                    self._cut_counter += 1

                    else_frame = Frame(
                        [right] + rest_goals,
                        frame.subst,
                        frame.depth,
                        0, 0,
                        frame.tail,
                        None,
                        frame.cut_id,
                        frame.disj_step,
                        frame.ifthen_meta,
                    )

                    barrier = _IfThenBarrier(
                        c_id=cond_cut_id,
                        then_goal=then,
                        rest_goals=rest_goals,
                        parent_cut_id=frame.cut_id,
                        parent_depth=frame.depth,
                        parent_ifthen_meta=frame.ifthen_meta,
                        else_frame=else_frame,
                    )

                    cond_frame = Frame(
                        [cond],
                        frame.subst,
                        frame.depth,
                        0, 0,
                        frame.tail,
                        None,
                        cond_cut_id,
                        frame.disj_step,
                        barrier,
                    )

                    stack.pop()
                    stack.append(else_frame)
                    stack.append(cond_frame)
                    continue

                # ---- Ordinary disjunction ----
                debug_print("  Handling disjunction ';'")
                disj_step = None
                if with_trace:
                    disj_step = self.tracer.start_goal(
                        str(current_goal.substitute(frame.subst)), frame.depth)
                    disj_step.builtin = True
                    disj_step.rule_used = ";"
                    pending_disj_steps.append(disj_step)

                left_frame = Frame([left] + rest_goals, frame.subst,
                                   frame.depth, 0, 0, frame.tail, None,
                                   frame.cut_id, disj_step, frame.ifthen_meta)
                right_frame = Frame([right] + rest_goals, frame.subst,
                                    frame.depth, 0, 0, frame.tail, None,
                                    frame.cut_id, disj_step, frame.ifthen_meta)

                stack.pop()
                stack.append(right_frame)
                stack.append(left_frame)
                debug_print("    Added two alternative frames")
                continue

            # --- Standalone `Cond -> Then` (without `;`) ---
            if (isinstance(current_goal, Term) and current_goal.functor == '->'
                    and len(current_goal.arguments) == 2):
                debug_print("  Handling conditional operator '->' (without ;)")
                cond, then = current_goal.arguments

                cond_cut_id = self._cut_counter
                self._cut_counter += 1

                else_frame = Frame(
                    [Term('fail', [])],
                    frame.subst,
                    frame.depth,
                    0, 0,
                    frame.tail,
                    None,
                    frame.cut_id,
                    frame.disj_step,
                    frame.ifthen_meta,
                )

                barrier = _IfThenBarrier(
                    c_id=cond_cut_id,
                    then_goal=then,
                    rest_goals=rest_goals,
                    parent_cut_id=frame.cut_id,
                    parent_depth=frame.depth,
                    parent_ifthen_meta=frame.ifthen_meta,
                    else_frame=else_frame,
                )

                cond_frame = Frame(
                    [cond],
                    frame.subst,
                    frame.depth,
                    0, 0,
                    frame.tail,
                    None,
                    cond_cut_id,
                    frame.disj_step,
                    barrier,
                )

                stack.pop()
                stack.append(else_frame)
                stack.append(cond_frame)
                continue

            # --- User-defined predicate ---
            rules = self.get_candidate_rules(current_goal)
            debug_print(f"  Rules found for {current_goal.functor}: {len(rules)}")
            if frame.rule_index >= len(rules):
                debug_print(f"  All rules tried, failure")
                if with_trace and frame.trace_step and not frame.trace_step.success:
                    frame.trace_step.success = False
                    self.tracer.end_goal(False)
                stack.pop()
                continue

            if frame.cut_id is None:
                frame.cut_id = self._cut_counter
                self._cut_counter += 1
                debug_print(f"  New cut_id created: {frame.cut_id}")

            rule = rules[frame.rule_index]
            debug_print(f"  Trying rule #{frame.rule_index+1}: {rule}")

            renamed_rule = self._rename_variables(rule, frame.depth)
            debug_print(f"    Renamed rule: {renamed_rule}")

            new_subst = current_goal.unify(renamed_rule.head,
                                           frame.subst,
                                           depth=frame.depth)
            if new_subst is None:
                debug_print("    Head unification failed")
                frame.rule_index += 1
                continue

            debug_print(f"    Unification succeeded")

            trace_step = None
            if with_trace:
                goal_str = str(current_goal.substitute(frame.subst))
                trace_step = self.tracer.start_goal(goal_str, frame.depth)
                for var, val in self._extract_unifications(current_goal,
                                                            renamed_rule.head,
                                                            new_subst):
                    trace_step.unification.append((str(var), str(val)))
                trace_step.rule_used = str(renamed_rule)

            body = renamed_rule.body
            if isinstance(body, Term) and body.functor == 'true':
                body_goals = []
            else:
                body_goals = (self._flatten_conjunction(body)
                              if isinstance(body, Term) and body.functor == ','
                              else [body])

            new_goals = body_goals + rest_goals
            debug_print(f"    New goals: {[str(g) for g in new_goals]}")

            if len(rest_goals) == 0:
                new_depth = frame.depth
                debug_print("    Tail call, depth not increased")
            else:
                new_depth = frame.depth + 1
                debug_print("    Non-tail call, depth increased")

            # --- Patch 5.8: subst pruning ---
            # new_subst contains all accumulated bindings, but for the new
            # frame only those reachable from new_goals plus anchor
            # query_vars are needed. Everything else is dead weight from
            # previous recursion iterations. Without this pruning, subst grows
            # linearly with depth, and unify copies it at each step —
            # yielding O(N²).
            #
            # alt_frame (if there are alternative rules) will receive
            # the original frame.subst, not the pruned one — that is fine,
            # because alt_frame executes a different set of rules and
            # requires the full context of its level.
            
            #pruned_subst = self._prune_subst_to_goals(new_subst, new_goals, query_vars)

            # Patch 5.9: if we are inside a Cond if-then-else, the variables
            # bound in Cond may be used in the Then/Else branches.
            # These branches are not visible from the Cond body, so they
            # must be explicitly added to the seed for pruning. Otherwise
            # ObsX/ObsY (and similar) lose their binding, and the Then branch
            # cannot dereference them — which led to "decide_move -> None"
            # in the presence of an obstacle.
            extra_goals = []
            fm = frame.ifthen_meta
            if fm is not None and frame.cut_id == fm.c_id:
                extra_goals.append(fm.then_goal)
                extra_goals.extend(fm.else_frame.goals)
                # In case of nested if-then-else — walk up the chain.
                while fm.parent_ifthen_meta is not None:
                    fm = fm.parent_ifthen_meta
                    extra_goals.append(fm.then_goal)
                    extra_goals.extend(fm.else_frame.goals)

            pruned_subst = self._prune_subst_to_goals(
                new_subst,
                extra_goals + [current_goal] + new_goals,
                query_vars,
            )
            
            debug_print(f"    subst pruned: {len(new_subst)} -> {len(pruned_subst)} entries")

            body_frame = Frame(new_goals, pruned_subst, new_depth, 0, 0,
                               frame.tail, trace_step, frame.cut_id,
                               frame.disj_step, frame.ifthen_meta)

            stack.pop()

            if frame.rule_index + 1 < len(rules):
                debug_print(f"    There are alternative rules")
                alt_frame = Frame(frame.goals, frame.subst, frame.depth,
                                  frame.rule_index + 1, len(rules),
                                  frame.tail, None, frame.cut_id,
                                  frame.disj_step, frame.ifthen_meta)
                stack.append(alt_frame)

            stack.append(body_frame)

        # End of while — stack is empty
        if with_trace:
            for disj_step in pending_disj_steps:
                if not disj_step.success:
                    disj_step.success = False
                    self.tracer.end_goal(False)
            if root_step:
                root_step.success = found_solution
                if root_step in self.tracer.step_stack:
                    while (self.tracer.step_stack
                           and self.tracer.step_stack[-1] is root_step):
                        self.tracer.end_goal(found_solution)

        debug_print("=== Search completed ===")

    # ==================== HELPER METHODS ====================

    def _invalidate_cache_for_rule(self, rule):
        if not self._renamed_cache:
            return
        rid = id(rule)
        dead = [k for k in self._renamed_cache if k[0] == rid]
        for k in dead:
            del self._renamed_cache[k]

    def _index_add_rule(self, rule, at_beginning=False):
        head = rule.head
        key = (head.functor, len(head.arguments))
        entry = self.rule_index.get(key)
        if entry is None:
            entry = {'all': [], 'const': {}}
            self.rule_index[key] = entry

        if at_beginning:
            entry['all'].insert(0, rule)
        else:
            entry['all'].append(rule)

        if head.arguments:
            first_arg = head.arguments[0]
            if self._is_const_first_arg(first_arg):
                const_val = (first_arg.value if isinstance(first_arg, Number)
                             else first_arg.functor)
                bucket = entry['const'].get(const_val)
                if bucket is None:
                    bucket = []
                    entry['const'][const_val] = bucket
                if at_beginning:
                    bucket.insert(0, rule)
                else:
                    bucket.append(rule)

    def _index_remove_rule(self, rule):
        head = rule.head
        key = (head.functor, len(head.arguments))
        entry = self.rule_index.get(key)
        if entry is None:
            return
        try:
            entry['all'].remove(rule)
        except ValueError:
            return

        if head.arguments:
            first_arg = head.arguments[0]
            if self._is_const_first_arg(first_arg):
                const_val = (first_arg.value if isinstance(first_arg, Number)
                             else first_arg.functor)
                bucket = entry['const'].get(const_val)
                if bucket is not None:
                    try:
                        bucket.remove(rule)
                    except ValueError:
                        pass
                    if not bucket:
                        del entry['const'][const_val]

        if not entry['all']:
            del self.rule_index[key]

    @staticmethod
    def _is_all_anon_args(term):
        if not isinstance(term, Term):
            return False
        for arg in term.arguments:
            if not (isinstance(arg, Variable) and arg.name == "_"):
                return False
        return True

    def _flatten_conjunction(self, term):
        if (isinstance(term, Term) and term.functor == ','
                and len(term.arguments) == 2):
            left = self._flatten_conjunction(term.arguments[0])
            right = self._flatten_conjunction(term.arguments[1])
            return left + right
        return [term]

    def _rename_variables(self, rule, depth):
        self._rename_nonce += 1
        nonce = self._rename_nonce
        var_map = {}

        def rename(term):
            if isinstance(term, Variable):
                if term.name == "_":
                    return term
                if term not in var_map:
                    var_map[term] = Variable(f"{term.name}_{nonce}")
                return var_map[term]
            if isinstance(term, Number):          # <-- BEFORE Term
                return term                        # Number is unchanged
            if isinstance(term, Term):
                new_args = [rename(arg) for arg in term.arguments]
                return Term(term.functor, new_args)
            return term

        new_head = rename(rule.head)
        new_body = rename(rule.body)
        return Rule(new_head, new_body)

    def _extract_unifications(self, goal, head, subst):
        unifications = []

        def extract_from_term(term1, term2, sub):
            if isinstance(term1, Variable) and term1.name != "_":
                if term1 in sub:
                    unifications.append((term1, sub[term1]))
            elif isinstance(term1, Term) and isinstance(term2, Term):
                for arg1, arg2 in zip(term1.arguments, term2.arguments):
                    extract_from_term(arg1, arg2, sub)

        extract_from_term(goal, head, subst)
        return unifications

    # ==================== KNOWLEDGE BASE OPERATIONS ====================

    def add_rule(self, rule: Rule, at_beginning: bool = False):
        if at_beginning:
            self.rules.insert(0, rule)
        else:
            self.rules.append(rule)
        self._index_add_rule(rule, at_beginning=at_beginning)

    def retract_first(self, template_rule: Rule) -> Optional[Dict]:
        template_head = template_rule.head

        if self._is_all_anon_args(template_head):
            key = (template_head.functor, len(template_head.arguments))
            entry = self.rule_index.get(key)
            if not entry or not entry['all']:
                return None
            rule = entry['all'][0]
            self._invalidate_cache_for_rule(rule)
            try:
                self.rules.remove(rule)
            except ValueError:
                pass
            self._index_remove_rule(rule)
            return {}

        for i, rule in enumerate(self.rules):
            renamed = self._rename_variables(rule, 9999)
            subst = renamed.head.unify(template_head, {})
            if subst is not None:
                self._invalidate_cache_for_rule(rule)
                del self.rules[i]
                self._index_remove_rule(rule)
                return subst
        return None

    def retract_all(self, template_rule: Rule) -> int:
        template_head = template_rule.head

        if self._is_all_anon_args(template_head):
            key = (template_head.functor, len(template_head.arguments))
            entry = self.rule_index.get(key)
            if not entry or not entry['all']:
                return 0
            matched = list(entry['all'])
            matched_ids = {id(r) for r in matched}
            for r in matched:
                self._invalidate_cache_for_rule(r)
            self.rules = [r for r in self.rules if id(r) not in matched_ids]
            del self.rule_index[key]
            return len(matched)

        kept: List[Rule] = []
        removed_rules: List[Rule] = []
        for rule in self.rules:
            renamed = self._rename_variables(rule, 9999)
            if renamed.head.unify(template_head, {}) is not None:
                removed_rules.append(rule)
            else:
                kept.append(rule)
        if removed_rules:
            self.rules = kept
            for r in removed_rules:
                self._invalidate_cache_for_rule(r)
                self._index_remove_rule(r)
        return len(removed_rules)

    def get_rules_for_predicate(self, name: str, arity: int) -> List[Rule]:
        key = (name, arity)
        entry = self.rule_index.get(key)
        if entry:
            return entry['all'][:]
        return []

    def abolish(self, name: str, arity: int) -> None:
        key = (name, arity)
        entry = self.rule_index.get(key)
        if not entry or not entry['all']:
            return
        matched = list(entry['all'])
        matched_ids = {id(r) for r in matched}
        for r in matched:
            self._invalidate_cache_for_rule(r)
        self.rules = [r for r in self.rules if id(r) not in matched_ids]
        del self.rule_index[key]

    def has_predicate(self, name: str, arity: int) -> bool:
        if self.builtins.get_predicate(name, arity):
            return True
        key = (name, arity)
        return key in self.rule_index