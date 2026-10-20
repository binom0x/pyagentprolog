""" PyAgentProlog

Interpreter with tracing, cut, and tail-call optimization support.

Patch 5.4: if-then-else via _IfThenBarrier without nested _solve.
Patch 5.5: removed _rename_variables cache (broke TCO).
Patch 5.6: fixed the constant-index bug (Variable as a key).
Patch 5.7: removed redundant subst.copy() in the hot path.
Patch 5.8: subst pruning on every rule application.
Patch 5.9: anchor variables in if-then-else.
Patch 5.10: proper ISO cut.
  Each call to a user-defined predicate receives a FRESH cut_id; the
  caller's cut_id is recorded in Frame.cut_barrier. When CutException
  is caught, everything in the current call's zone is removed
  (cut_id >= barrier). Frames of calling predicates (cut_id < barrier)
  are preserved.
Patch 5.11 (this file): if-then-else commit restoration.
  PATCH 5.10 broke the if-then-else commit detection, because the
  "Cond succeeded" check is `frame.cut_id == frame.ifthen_meta.c_id`,
  and a fresh cut_id for the cond's subgoal made those two ids differ.
  Fix: if the frame being expanded IS the direct cond frame
  (ifthen_meta is not None and cut_id == ifthen_meta.c_id), the
  resulting body_frame keeps cut_id = frame.cut_id (the cond_cut_id)
  instead of a fresh one. Semantically correct: inside (Cond -> Then ; Else),
  a `!` in Cond behaves exactly like the `->` — both commit the cond
  and go to Then. So the cond's cut zone IS cond_cut_id, and any
  subcalls inside Cond belong to it.

  KNOWN LIMITATION (unchanged by 5.11):
    p :- q, !, r.
    p :- fallback.
  `!` here is syntactically in p, but executed inside q's body_frame
  (body_goals + rest_goals are merged into a single goals list).
  cut_id belongs to q, so p's alternative clause is NOT removed.
  Workaround: ( q -> true ; fail ), or if-then-else.
  Full fix requires separating body_goals from rest_goals in Frame.

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
    """Stack frame for iterative search.

    cut_id
        Identity of the "call zone". Each call to a user-defined predicate
        creates a fresh cut_id (except the direct cond frame of an
        if-then-else — see PATCH 5.11). `!` pops all frames whose
        cut_id >= the current zone.

    cut_barrier
        The caller's cut_id at the moment of this frame's creation.
        Diagnostic only; not consulted by the cut handler.
    """
    __slots__ = ('goals', 'subst', 'depth', 'rule_index', 'total_rules',
                 'tail', 'trace_step', 'cut_id', 'disj_step', 'ifthen_meta',
                 'cut_barrier')

    def __init__(self, goals, subst, depth, rule_index, total_rules, tail,
                 trace_step, cut_id=None, disj_step=None, ifthen_meta=None,
                 cut_barrier=None):
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
        self.cut_barrier = cut_barrier


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

    def _prune_subst_to_goals(self, subst, goals, keep_vars):
        if not subst:
            return {}

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

        pruned = {}
        work = list(seed)
        while work:
            v = work.pop()
            if v in pruned:
                continue
            if v not in subst:
                continue
            val = subst[v]
            chain_seen = {v}
            while isinstance(val, Variable) and val in subst:
                if val in chain_seen:
                    break
                chain_seen.add(val)
                val = subst[val]
            pruned[v] = val
            inner = [val]
            while inner:
                t = inner.pop()
                if isinstance(t, Variable):
                    if t.name != "_" and t not in pruned:
                        work.append(t)
                elif isinstance(t, Term):
                    inner.extend(t.arguments)
        return pruned

    def _solve_iterative(self, initial_goal: Term, initial_subst: Dict, depth: int, with_trace: bool):
        debug_print(f"\n=== Beginning search for goal: {initial_goal} "
                    f"with substitution {initial_subst}, depth {depth} ===")

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
                        cut_barrier=barrier.parent_cut_id,
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

                    # ★ PATCH 5.10: proper ISO cut.
                    barrier = frame.cut_id
                    if barrier is not None:
                        new_stack = [
                            f for f in stack
                            if f is frame or f.cut_id is None or f.cut_id < barrier
                        ]
                        stack[:] = new_stack

                    frame.goals = rest_goals
                    frame.subst = e.subst
                    continue
                except RepeatException as e:
                    debug_print("  RepeatException: adding alternative frame")
                    stack.pop()
                    continue_frame = Frame(rest_goals, e.subst, frame.depth,
                                           0, 0, frame.tail, None,
                                           frame.cut_id, frame.disj_step,
                                           frame.ifthen_meta, frame.cut_barrier)
                    repeat_frame = Frame([current_goal] + rest_goals, e.subst,
                                         frame.depth, 0, 0, frame.tail, None,
                                         frame.cut_id, frame.disj_step,
                                         frame.ifthen_meta, frame.cut_barrier)
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
                                              frame.disj_step, frame.ifthen_meta,
                                              frame.cut_barrier)
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

                        barrier = frame.cut_id
                        if barrier is not None:
                            new_stack = [
                                f for f in stack
                                if f is frame or f.cut_id is None or f.cut_id < barrier
                            ]
                            stack[:] = new_stack

                        frame.goals = rest_goals
                        frame.subst = e.subst
                        continue
                    except RepeatException as e:
                        debug_print("  RepeatException inside generator")
                        stack.pop()
                        continue_frame = Frame(rest_goals, e.subst, frame.depth,
                                               0, 0, frame.tail, None,
                                               frame.cut_id, frame.disj_step,
                                               frame.ifthen_meta, frame.cut_barrier)
                        repeat_frame = Frame([current_goal] + rest_goals, e.subst,
                                             frame.depth, 0, 0, frame.tail,
                                             None, frame.cut_id, frame.disj_step,
                                             frame.ifthen_meta, frame.cut_barrier)
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
                                          frame.ifthen_meta, frame.cut_barrier)
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
                        cut_barrier=frame.cut_barrier,
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
                        cut_barrier=frame.cut_id,
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
                                   frame.cut_id, disj_step, frame.ifthen_meta,
                                   frame.cut_barrier)
                right_frame = Frame([right] + rest_goals, frame.subst,
                                    frame.depth, 0, 0, frame.tail, None,
                                    frame.cut_id, disj_step, frame.ifthen_meta,
                                    frame.cut_barrier)

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
                    cut_barrier=frame.cut_barrier,
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
                    cut_barrier=frame.cut_id,
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
                self.tracer.set_substitutions(frame.subst, new_subst) # patch fo IDE subst_before/after
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

            extra_goals = []
            fm = frame.ifthen_meta
            if fm is not None and frame.cut_id == fm.c_id:
                extra_goals.append(fm.then_goal)
                extra_goals.extend(fm.else_frame.goals)
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

            # ★ PATCH 5.10: fresh cut_id per call, so `!` from inside
            # the predicate does not truncate choice points of callers
            # (this fixed `format_answer`'s cut inside findall in main/1).
            #
            # ★ PATCH 5.11: if the current frame is the DIRECT cond frame
            # of an if-then-else, keep its cut_id. Otherwise the "Cond
            # succeeded → commit to Then" check `frame.cut_id ==
            # frame.ifthen_meta.c_id` fails, and the interpreter falls
            # through to "yield solution" instead of committing.
            # Semantically: inside (Cond -> Then ; Else), a `!` in Cond
            # behaves exactly like `->` — both commit the cond. So the
            # cond's cut zone IS cond_cut_id, and subcalls of Cond are
            # legitimately in it.
            if (frame.ifthen_meta is not None
                    and frame.cut_id == frame.ifthen_meta.c_id):
                caller_cut_id = frame.cut_barrier
                fresh_cut_id = frame.cut_id
            else:
                caller_cut_id = frame.cut_id
                fresh_cut_id = self._cut_counter
                self._cut_counter += 1

            body_frame = Frame(new_goals, pruned_subst, new_depth, 0, 0,
                               frame.tail, trace_step, fresh_cut_id,
                               frame.disj_step, frame.ifthen_meta,
                               cut_barrier=caller_cut_id)

            stack.pop()

            if frame.rule_index + 1 < len(rules):
                debug_print(f"    There are alternative rules")
                alt_frame = Frame(frame.goals, frame.subst, frame.depth,
                                  frame.rule_index + 1, len(rules),
                                  frame.tail, None, fresh_cut_id,
                                  frame.disj_step, frame.ifthen_meta,
                                  cut_barrier=caller_cut_id)
                stack.append(alt_frame)

            stack.append(body_frame)

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
            if isinstance(term, Number):
                return term
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