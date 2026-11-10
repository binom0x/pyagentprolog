# Introduction to Prolog Programming

*For users of the PyAgentProlog IDE*

---
**Prolog** (from *PROgramming in LOGic*) is a **logic programming** language created in 1972 by Alain Colmerauer and Philippe Roussel. Unlike familiar languages (Python, C, Java), where a program is a **sequence of instructions** to be executed, a Prolog program is a **set of facts and rules** describing a domain and **questions** posed to that knowledge base. The executor (the Prolog engine) finds the answer itself by enumerating possible variable substitutions and applying inference rules. Prolog is based on **first-order predicate calculus** (formal logic) and on the mechanism of **unification** — matching terms — with automatic backtracking. Prolog is widely used in expert systems, natural language processing, automated theorem proving, planning, and — in recent years — in neuro-symbolic systems where logical inference is combined with neural network models. PyAgentProlog is a modern implementation of this paradigm in Python, extended with agents, tracing, and semantic unification.

---
## Part I. Declarative Programming

### 1. Problem Statement: A Family Knowledge Base

Before discussing declarative programming, let us define the **problem** and the **data**.

We have a knowledge base about a family — who is whose parent. Let it be stored as a list of pairs `(parent, child)`:

```python
family = [
    ("ivan",   "alexey"),
    ("ivan",   "maria"),
    ("alexey", "petr"),
    ("alexey", "anna"),
    ("maria",  "sergey"),
]
```

What this data structure means:

- Each pair `(a, b)` is **one fact**: `a` is the parent of `b`.
- `("ivan", "alexey")` reads: *"Ivan is the parent of Alexey."*
- `("alexey", "petr")` — *"Alexey is the parent of Petr."*
- The first element of a pair is the **parent**, the second is the **child**.
- The order of pairs in the list does not matter; only their content matters.

From this data we can answer questions:
- Who are Ivan's children? → `alexey`, `maria`.
- Who are Petr's parents? → `alexey`.
- Who are Ivan's **grandchildren**? → the children of his children → `petr`, `anna`, `sergey`.

The last question is the **goal of our task**: find all of Ivan's grandchildren.

Visually, this is the family tree:

```
       ivan
      /    \
   alexey  maria
   /   \      \
 petr  anna  sergey
```

Ivan's grandchildren are `petr`, `anna`, `sergey` — all nodes at the **second** level below Ivan.

### 2. Imperative Solution: How to Do It in Python

Let us try to solve the problem the familiar way — with loops.

**Algorithm idea:**

1. Iterate over all pairs and find Ivan's children (direct descendants).
2. For each child — iterate over all pairs again and find **their** children.
3. Collect the results in a list.

**Code:**

```python
def find_grandchildren(parents, person):
    result = []
    # Step 1: find all children of person
    children = []
    for (p, c) in parents:
        if p == person:
            children.append(c)
    # Step 2: for each child, find their children
    for child in children:
        for (p, c) in parents:
            if p == child:
                result.append(c)
    return result

print(find_grandchildren(family, "ivan"))
# ['petr', 'anna', 'sergey']
```

**What is important to understand here:**

- We **explicitly describe the steps**: "first find the children, then for each — the grandchildren."
- The outer loop iterates over `children` (the list of direct children).
- The inner loop iterates over the **entire** `parents` list for each child.
- Complexity is **O(N × M)**, where N is the number of direct children and M is the total size of the base. With 1000 facts and 3 children — 3000 iterations.

**Weaknesses of this solution:**

1. **Fragility.** If the data comes in a different format — for example, `{"parent": "ivan", "child": "alexey"}` (dictionaries instead of pairs) — **the entire function must be rewritten**.
2. **Repetition.** The logic "find the children of X" is duplicated twice — in the outer and inner loops. If a third condition appears (e.g., "only children over 18"), the code grows again.
3. **Scaling.** Want great-grandchildren? You need a **third** nested loop:
   ```python
   for child in children:
       for grandchild in get_children(parents, child):
           for great_grandchild in get_children(parents, grandchild):
               result.append(great_grandchild)
   ```
   Each new level — a new loop. Want "all descendants at any depth"? Now you need a recursive function that does the same work at every step.
4. **Indexes.** To find the children of `child`, you traverse the **entire** `family` list every time. If there are a million facts — that is expensive.

All of this is the price of describing **how** to search rather than **what** to search for.

### 3. Declarative Solution: What We Are Looking For

Now let us describe the same task declaratively. In Prolog, **data** is facts:

```prolog
parent(ivan, alexey).
parent(ivan, maria).
parent(alexey, petr).
parent(alexey, anna).
parent(maria, sergey).
```

Each fact `parent(a, b)` is exactly the same as the pair `(a, b)` in the Python list: *"a is the parent of b."*

And now — the rule for grandchildren:

```prolog
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

It reads as a **definition**:

> *"X is the grandfather/grandmother of Z **IF** X is the parent of Y **AND** Y is the parent of Z."*

Compare with the imperative solution:
- **No loops.** We do not say "traverse the list, find, repeat."
- **No indexes, list lengths, or exit conditions.**
- **No `get_children` functions.** The logic "find the children of X" is just `parent(X, Y)`, built into unification.
- **One rule describes all cases** — for one fact, for a thousand, for an empty base.

**Query:**

```prolog
?- grandparent(ivan, X).
```

Prolog finds all solutions itself and returns:

```
X = petr ;
X = anna ;
X = sergey.
```

### 4. What the Prolog Engine Does

When you pose `grandparent(ivan, X)`, the engine **searches for a proof**:

1. **Matching against the rule.** It finds the rule `grandparent(X, Z) :- parent(X, Y), parent(Y, Z)`. It unifies the rule head with the query: the rule's variable `X` becomes `ivan`, the rule's variable `Z` becomes the query's variable `X` (both still unbound).

2. **First goal:** `parent(ivan, Y)`. It searches for facts with `parent(ivan, ...)`:
   - `parent(ivan, alexey)` → `Y = alexey`.
   - It saves a choice point: "if this fails, take `maria`."

3. **Second goal:** `parent(alexey, Z)`. It searches for facts with `parent(alexey, ...)`:
   - `parent(alexey, petr)` → `Z = petr`. **Success.**
   - It saves a choice point: "if more are requested — take `anna`."

4. **Returns the first solution:** `X = petr`.

5. **The user asks for "more"** (`▶ Run` finds all, `⏩ First` — only the first):
   - The engine returns to the last choice point in `parent(alexey, Z)` and takes `anna` → `X = anna`.
   - Again → `Z = petr`, `anna` are exhausted, it returns to `parent(ivan, Y)` → takes `maria` → `parent(maria, sergey)` → `X = sergey`.
   - Again → everything is exhausted, the answer is `false` (or "no solutions").

**Key point:** the engine does everything that in Python was loops, lists, and checks. You described **what** it means to be a grandchild; the engine found **all** matching values.

### 5. Comparison on a Single Task

| Aspect | Imperative (Python) | Declarative (Prolog) |
|---|---|---|
| Data | `family = [(a,b), ...]` | `parent(a, b).` |
| Logic | 2 nested loops | `grandparent(X,Z) :- parent(X,Y), parent(Y,Z).` |
| Find children of X | Separate function or loop | Simply the goal `parent(X, Y)` |
| Scaling (great-grandchildren) | +1 loop | +1 rule |
| Scaling (all descendants) | Recursive function + manual stack | `descendant(X,Z) :- parent(X,Z).` <br> `descendant(X,Z) :- parent(X,Y), descendant(Y,Z).` |
| Fragility to data format | High | The format is the syntax of the facts itself |
| Performance on 1000 facts | O(N) traversals | Index on the first argument — O(1) for `parent(ivan, Y)` |

The rule for "all descendants" in Prolog:

```prolog
descendant(X, Y) :- parent(X, Y).
descendant(X, Y) :- parent(X, Z), descendant(Z, Y).
```

It finds **any** descendant at **any** depth — without a single loop. Try writing the same in Python without recursion — you will get 10–15 lines with a manual stack and cycle checking (otherwise it will run forever on a graph with a loop).

### 6. Why This Is Possible: Unification Instead of Assignment

The root of the difference is that in Prolog **there is no assignment**. There is only **unification** — the operation "make two terms identical by finding a substitution of variables."

Consider imperative assignment:

```python
y = alexey           # y is now alexey
y = maria            # y is now maria (the old value is lost)
```

In Prolog:

```prolog
Y = alexey,          % Y is bound to alexey
Y = maria.           % attempt to bind Y to maria → failure
```

`Y` is **not a memory cell** but a logic variable. It can be bound **once** — and remains bound until the end of the proof. You cannot "overwrite" it — you can only **backtrack** to try another value.

This is exactly why `parent(X, Y), parent(Y, Z)` works as a "natural" description: `Y` is the **shared** child of the two goals, and unification guarantees that both `parent` calls agree.

In imperative code the same thing is done **manually**: in an intermediate variable, in checks, in loops. In Prolog this agreement is **built into the language**.

### 7. What This Gives in Practice

Three things you get from thinking declaratively:

**Less code.** The `grandparent` rule is 3 lines. The Python equivalent is 8 lines, and it grows with each new relation.

**Fewer errors.** No loops — no off-by-one, no "forgot to check for an empty list," no "swapped the argument order." Errors are possible in the **logic** itself — but they are visible directly in the text of the rule.

**Natural relationality.** The same rule works in **different modes**:

```prolog
?- grandparent(ivan, X).      % who are Ivan's grandchildren?
?- grandparent(X, petr).      % who are Petr's grandparents?
?- grandparent(ivan, petr).   % is Ivan Petr's grandparent? (true/false)
?- findall(X-Z, grandparent(X, Z), L).  % all grandparent-grandchild pairs
```

The imperative function `find_grandchildren(parents, person)` works **only** in the first mode. To get "who are Petr's grandparents," you need **another** function. In Prolog — the very same one.

### 8. The Limits of Declarativeness

Not everything is so rosy. The flip side:

**Goal order matters.** `parent(X, Y), parent(Y, Z)` and `parent(Y, Z), parent(X, Y)` will give different results in efficiency, and sometimes in correctness (if one of the goals runs forever). Prolog executes goals **left to right** — you must think about order.

**Infinite recursion is a real problem.** `loop :- loop.` is a program that never terminates. In imperative code such a bug would be caught as a "hang," here too — but without an explicit call stack.

**Performance requires understanding.** `assert`/`retract` rebuild the index. `findall` collects **all** solutions into memory. Rule order affects solution order. The index on the first argument works only for **constants**.

**The cut `!` is not declarative.** It is an imperative "stop," and it must be applied consciously.

But these limitations are **details** that come with experience. The main thing is the **shift in thinking**: you describe **what** is true, not **how** to compute it. Prolog takes care of enumeration, unification, and backtracking.

This is why the declarative approach fits well with tasks where **description** matters (expert systems, ontologies, natural language processing, agent world models, planning) rather than **algorithm** (number crunching, stream processing).

In the following sections we will cover Prolog syntax, recursion, cut, and typical patterns — but all of it will be built on one principle: **describe the world, not the procedure**.

---
## Part II. Syntax Basics

### 7. Terms: Four Kinds

Everything in Prolog is a **term**. There are four kinds.

**Atoms** — simple names, starting with a lowercase letter:

```prolog
ivan
parent
hello_world
moscow
```

They can be quoted if spaces or uppercase letters are needed:
```prolog
'Hello, World'
'Ivan Ivanovich'
```

**Numbers** — integers and floats:

```prolog
42
-7
3.14
1e-3
```

**Variables** — start with an uppercase letter or `_`:

```prolog
X
Y
MyVar
_
```

`_` is the **anonymous** variable: each occurrence is a new variable. Useful when the value is not needed:

```prolog
has_child(X) :- parent(X, _).
```
*"X has a child if X is the parent of someone (of whom — doesn't matter)."*

**Compound terms** — functor + arguments in parentheses:

```prolog
parent(ivan, alexey)
f(X, Y, Z)
point(3, 4)
list_of(a, b, c)
```

A functor is just a name. `parent(ivan, alexey)` does **not** "call the function parent" — it is a fact with two arguments.

### 8. Unification

Unification is the heart of Prolog. It is the **matching** of two terms with the search for a substitution that makes them identical.

Unification rules:
- An atom unifies with itself: `ivan = ivan`.
- A variable unifies with **anything**: `X = 42`, `X = parent(ivan, Y)`.
- If a variable is **already bound**, it is replaced by its value: `X = 42, X = 42` — success; `X = 42, X = 43` — failure.
- Compound terms unify if the functors, arities, and all arguments unify.

```prolog
?- f(X, 2) = f(1, Y).
X = 1, Y = 2.
```

```prolog
?- f(X, X) = f(1, 2).
false.
```
*It is impossible for X to be both 1 and 2.*

```prolog
?- parent(X, Y) = parent(ivan, alexey).
X = ivan, Y = alexey.
```

**Key idea:** `=` is not "assignment" but "unification." `X = 42` does not "put 42 into X" but "says that X and 42 are the same thing."

### 9. Queries

In the IDE, queries are written without `?-`, but with a trailing period:

```prolog
parent(ivan, X).
```

The engine will answer:
```
X = alexey
── N solutions ──
```

If there are several solutions (e.g., `parent(ivan, X)` yields `alexey` and `maria`), `▶ Run` shows all, and `⏩ First` — only the first.

### 10. Facts and Rules

A **fact** is a clause without a body:

```prolog
parent(ivan, alexey).
```

It reads: *"ivan is the parent of alexey"* (always true).

A **rule** has a body:

```prolog
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).
```

It reads: *"X is the grandfather/grandmother of Z if X is the parent of Y and Y is the parent of Z."*

### 11. Execution Order

Prolog proves goals **left to right**. On failure it **goes back** (backtracking) and tries the next alternative.

```prolog
?- parent(ivan, X), parent(X, petr).
```

1. Prove `parent(ivan, X)` → `X = alexey` (the first fact).
2. With `X = alexey`, prove `parent(alexey, petr)` → success (such a fact exists).
3. Answer: `X = alexey`.

If `parent(alexey, petr)` had not been found, the engine would return to step 1 and take the next `parent(ivan, X)` — `maria`. And it would again check `parent(maria, petr)`.

### 12. Backtracking

Backtracking is automatic **enumeration**. When Prolog cannot prove a goal, it rolls back to the last choice point and tries another alternative.

```prolog
parent(ivan, alexey).
parent(ivan, maria).

?- parent(ivan, X).
X = alexey ;   ← the user presses "more"
X = maria.
```

The `;` point means "continue the search." In the IDE, `▶ Run` finds all solutions, `⏩ First` — only the first.

### 13. Arithmetic

Arithmetic is a **separate** area in Prolog. The term `2 + 3` is **not** the number `5` but a structure with the functor `+`:

```prolog
?- X = 2 + 3.
X = 2+3.        ← not 5!
```

To **evaluate**, you need the `is` operator:

```prolog
?- X is 2 + 3.
X = 5.
```

`is` is a predicate: `Result is Expression`. It evaluates `Expression` and unifies it with `Result`.

**Number comparison:**

```prolog
X > Y       % greater
X < Y       % less
X >= Y      % greater or equal
X =< Y      % less or equal (note =<, not <=)
X =:= Y     % numeric equality
X =\= Y     % numeric inequality
```

Do not confuse with `=`:

```prolog
?- 2 = 2.       true
?- 2 = 2.0.     false    ← different terms
?- 2 =:= 2.0.   true     ← identical as numbers
```

### 14. Lists

A list is either the empty `[]` or a cons cell `[Head | Tail]`:

```prolog
[]
[1, 2, 3]
[a, b | Rest]
[H | T]
```

In fact `[1, 2, 3]` is syntactic sugar for `'.'(1, '.'(2, '.'(3, [])))`. But you don't need to think that way: `[H | T]` is the **head** `H` and the **tail** `T`.

Example — list length:

```prolog
my_length([], 0).
my_length([_ | T], N) :-
    my_length(T, N1),
    N is N1 + 1.
```

It reads:
- *"the length of the empty list = 0"*.
- *"the length of a list with a head and tail T = N if the length of T = N1 and N = N1 + 1"*.

The first fact is the base case. The second rule is the recursion step. This is the classic Prolog pattern.

### 15. Recursion

Recursion in Prolog is the norm, not the exception. Any structure (list, tree, number) is traversed recursively:

```prolog
factorial(0, 1).
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),
    F is N * F1.
```

It reads: *"the factorial of 0 is 1; the factorial of N is F if N > 0, N1 = N − 1, the factorial of N1 is F1, F = N × F1."*

**A good practice rule:** always put the base case **first**, and the check `N > 0` — before the recursive call. Otherwise `factorial(-1, F)` will go into infinite recursion.

### 16. Tail Recursion (TCO)

If the recursive call is the **last goal** in the body of the rule, Prolog can execute it without stack growth (Tail Call Optimization):

```prolog
% Non-tail: the recursive call is in the middle
factorial(N, F) :-
    N > 0,
    N1 is N - 1,
    factorial(N1, F1),       ← not last
    F is N * F1.

% Tail: the recursive call is the last goal
factorial(N, F) :- factorial_acc(N, 1, F).
factorial_acc(0, Acc, Acc).
factorial_acc(N, Acc, F) :-
    N > 0,
    Acc1 is Acc * N,
    N1 is N - 1,
    factorial_acc(N1, Acc1, F).    ← last goal
```

The tail version uses an **accumulator** (`Acc`) and does not grow the stack. For `factorial(10000, F)` this is critical.

### 17. The Cut `!`

The cut is a control construct. `!` **cancels** all choice points created since entering the current predicate:

```prolog
max(X, Y, X) :- X >= Y, !.
max(_, Y, Y).
```

Without `!`, the query `max(3, 5, M)`:
1. Tries the first rule: `3 >= 5` — failure.
2. Tries the second: `M = 5`. Success.

But what about `max(5, 3, M)`?
1. First rule: `5 >= 3` — success, `M = 5`.
2. **Without `!`** the engine remembers that the second rule can be tried. The user can ask for "more" and get `M = 3` — **wrong**!

With `!` after `X >= Y`:
1. First rule: `5 >= 3` — success, `!` cuts off the second rule, `M = 5`.
2. "More" — no solutions. Correct.

**Rule:** place `!` **immediately after** the check that makes the remaining rules unnecessary.

**Caution:** abusing `!` hides alternatives and makes the program non-relational. Use it only when you are sure.

### 18. Negation `\+`

`\+ Goal` — "Goal is not provable":

```prolog
not_parent(X) :- \+ parent(X, _).
```

**Important:** `\+` works via **failure of proof**, not via "falsehood in the logical sense." If the variables in `Goal` are unbound, `\+` may give an unexpected result. Always make sure the variables are bound before calling `\+`.

### 19. If-then-else

```prolog
classify(X, positive) :- X > 0, !.
classify(X, negative) :- X < 0, !.
classify(_, zero).
```

Or via `->` (if-then-else):

```prolog
sign(X, S) :-
    ( X > 0 -> S = positive
    ; X < 0 -> S = negative
    ; S = zero
    ).
```

`Cond -> Then ; Else` — if `Cond` is provable, only `Then` is executed, otherwise only `Else`. Alternatives inside `Cond` are cut off automatically (as if there were a `!` there).

### 20. Putting It All Together

The "family" program:

```prolog
parent(ivan, alexey).
parent(ivan, maria).
parent(alexey, petr).
parent(alexey, anna).
parent(maria, sergey).

grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

sibling(X, Y) :-
    parent(P, X),
    parent(P, Y),
    X \= Y.

cousin(X, Y) :-
    parent(P1, X),
    parent(P2, Y),
    sibling(P1, P2).
```

Queries:

```prolog
?- grandparent(ivan, X).
X = petr ;
X = anna ;
X = sergey.
```

```prolog
?- sibling(petr, X).
X = anna.
```

```prolog
?- cousin(petr, X).
X = sergey.
```

Each rule reads like an English sentence. Not a single loop, not a single `if`. Only a description of the world.

---
## Part III. Custom Predicates in Python

So far we have described the world **only with Prolog facts and rules**. But there are tasks where pure logic is powerless: call an LLM, query a database, compute a complex formula, send an HTTP request. For this, PyAgentProlog has **Python actions** — ordinary Python functions connected to Prolog as predicates.

### 23. Why This Is Needed

Logic and computation are different things. Prolog describes **relations** well (`X is the parent of Y`, `X is the grandparent of Z`). But it has no access to the outside world:

- how to ask ChatGPT "What is the capital of France?";
    
- how to find the distance between two points;
    
- how to read a file;
    
- how to send an email;
    
- how to call a local model.
    

All of these are **actions** performed by Python. The idea is simple: **describe the world in Prolog, and the actions in Python**.

```prolog

% Prolog describes what a "near city" means
near(City) :-
    city(City, X, Y),
    distance_to_target(X, Y, D),   ← Python action
    D < 5.0.
```

``` Python
# Python action
@simple_action('distance_to_target', 3, output_arg=2)
def distance_to_target(x, y, context):
    tx = context.get('target_x', 10.0)
    ty = context.get('target_y', 5.0)
    return ((x - tx) ** 2 + (y - ty) ** 2) ** 0.5
```
Prolog describes the **logic**, Python — the **computation**. This is the hybrid approach — "neuro-symbolic" in the broad sense.

### 24. Three Kinds of Decorators

PyAgentProlog provides three decorators from `prolog.actions`. All of them:

- turn an ordinary Python function into a Prolog predicate;
    
- accept arguments automatically converted from Prolog terms to Python values;
    
- pass `context` — a dictionary of external resources;
    
- are registered in `Agent` automatically.
    

The difference is in **how many solutions** the function yields.

#### 24.1. `simple_action` — One Solution

```python

from prolog.actions import simple_action
@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2
```

- **Signature:** `@simple_action(name, arity, output_arg=-1, description=None)`
    
- **Parameters:**
    
    - `name='double'` — the predicate name in Prolog.
        
    - `arity=2` — the number of arguments in Prolog (`double/2`).
        
    - `output_arg=1` — the index of the output argument (0-based). The rest are input.
        
    - `description` — description (by default — the function's docstring).
        
- **Input arguments** must be **instantiated** before the call (not be free variables). Otherwise the action will not fire.
    
- **Returns** one value → it is unified with `output_arg`.
    
- If the function returned `None` → the predicate is considered **failed**.
    

In Prolog:

```prolog
?- double(21, X).
X = 42.
```

With multiple inputs:

```python
@simple_action('distance', 3, output_arg=2)
def distance(x1, y1, context):
    return ((x1 - 10.0) ** 2 + (y1 - 5.0) ** 2) ** 0.5
```

```prolog
?- distance(13.0, 9.0, D).
D = 5.0.
```

#### 24.2. `effector_action` — Side Effect, No Return

```python
from prolog.actions import effector_action
@effector_action('log_msg', 1)
def log_msg(msg, context):
    print(f"[LOG] {msg}")
```

- **Signature:** `@effector_action(name, arity, description=None)`
    
- **All arguments** are input. They must be instantiated.
    
- **Returns** `None`. The side effect is printing, writing to a file, sending to an API, modifying `context`.
    
- The predicate **succeeds** if the function executed without exceptions.
    
- If there is an exception inside — the predicate **fails**.
    

In Prolog:

```prolog
run :-
    log_msg('start'),
    do_work,
    log_msg('done').
```

The query `run.` → in the IDE console `[LOG] start`, `[LOG] done`, and in the Result `── 1 solution(s) ──` (success without variables).

#### 24.3. `generator_action` — Many Solutions

```python
from prolog.actions import generator_action
@generator_action('count_to', 2, output_arg=1)
def count_to(n, context):
    for i in range(1, n + 1):
        yield i
```

- **Signature:** `@generator_action(name, arity, output_arg=-1, description=None)`
    
- The function is a **generator** (`yield` instead of `return`).
    
- Each value from the iterator → a **separate solution**.
    
- The order of solutions is the order of `yield`.
    

In Prolog:

```prolog
?- count_to(3, X).
X = 1 ;
X = 2 ;
X = 3.
```

`▶ Run` shows all solutions, `⏩ First` — only `X = 1`.

### 25. Converting Values Prolog ↔ Python

When an action is called, the Prolog arguments are converted to Python values. Conversely, the result is converted to a Prolog term.

**Prolog → Python:**

|Prolog|Python|
|---|---|
|`Number(42)`|`42`|
|`Number(3.14)`|`3.14`|
|`Term('ivan')` (atom)|`'ivan'`|
|`Term('"hello"')` (quoted string)|`'hello'`|
|List `[1, 2, 3]`|`[1, 2, 3]`|
|`Variable('X')` (free)|`None`|
|Compound term `f(a, b)`|`'f(a, b)'` (string)|

**Python → Prolog:**

|Python|Prolog|
|---|---|
|`42` / `3.14`|`Number(42)` / `Number(3.14)`|
|`True` / `False`|`Term('true')` / `Term('false')`|
|`'hello'`|`Term('"hello"')` (quoted string)|
|`[1, 2, 3]`|Prolog list `[1, 2, 3]`|
|`None`|Empty list `[]`|
|`{'a': 1}`|List of pairs `[a-1]`|

**Important consequence:** input arguments must be **instantiated**. If you pass `double(X, Y)` in Prolog with free `X, Y`, the action returns `[]` — failure. Always make sure the variables are bound before calling an action:

```prolog
?- X = 21, double(X, Y).
X = 21, Y = 42.
```

### 26. `context` — External Resources

All actions receive a **second invisible argument** — `context`. It is a dictionary that holds external resources: the LLM client, the DB connection, API keys, configuration.

```python
@simple_action('ask_llm', 2, output_arg=1)
def ask_llm(prompt, context):
    llm = context.get('llm')
    if llm is None:
        return "(no llm configured)"
    return llm.invoke(prompt).content
```

`context` **does not come from Prolog** — it is a global dictionary tied to the `Agent`. It is populated like this:

```python
from langchain_openai import ChatOpenAI
agent.context['llm'] = ChatOpenAI(model='gpt-4o-mini')
agent.context['api_key'] = 'sk-...'
agent.context['db'] = my_database_connection
```

Then in Prolog:

prolog

ask(P, R) :- ask_llm(P, R).

```prolog
?- ask('What is the capital of France?', R).
R = 'Paris'.
```

**Why `context` and not global variables?** Because different `Agent`s may have **different** LLMs/DBs/keys. `context` isolates them — like a closure in Python.

### 27. Registering Actions

There are **three ways** to connect an action to an `Agent`.

#### 27.1. A Single Function

```python
agent.register_action(double)
agent.register_action(log_msg)
agent.register_action(count_to)
```

Each decorated function must be registered **explicitly**.

#### 27.2. An Entire Module

```python
import my_actions
agent.register_action_module(my_actions)
```

`register_action_module` scans all attributes of the module, finds functions with `_agent_action=True` (set by any of the three decorators), and registers them.

#### 27.3. Automatically — in the PyAgentProlog IDE

In the IDE you **do not need** to register anything manually. Just place `my_actions.py` next to `kb.pl`:

```text
my_project/
├── kb.pl
└── my_actions.py
```

`my_actions.py`:

```python
from prolog.actions import simple_action, effector_action
@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2
@effector_action('log_msg', 1)
def log_msg(msg, context):
    print(f"[LOG] {msg}")
```

When loading the program, the IDE **itself** imports `my_actions.py` and calls `agent.register_action_module(mod)`. Your functions are immediately available in Prolog.

**What you **do not** need to do:**

❌ **Do not** call `agent.register_action(...)` inside `my_actions.py` — the IDE will do it itself.

❌ **Do not** import `Agent` and create it manually.

❌ **Do not** do `sys.path.insert` — the IDE has already added the project folder to `sys.path`, so neighboring modules are imported freely:

```python
# my_actions.py
from my_helpers import compute  # located in the same project folder
```

**Reimport.** If you changed `my_actions.py` — press **🔄 Reload** in column 2 of the IDE. The IDE compares `mtime_ns + size` of the file and calls `importlib.reload` if it has changed. No IDE restart is needed.

### 28. Error Handling in Actions

The default behavior: **any exception inside an action → the predicate fails** (`fail`). This is convenient for "silent" catching, but sometimes hides errors.

Example:

```python
@simple_action('divide', 3, output_arg=2)
def divide(x, y, context):
    return x / y
```

```prolog
?- divide(10, 0, R).
false.           ← not an error, just fail
```

If you want an **explicit** error, raise `PrologError`:

```python
from prolog.predicates import PrologError
@simple_action('divide', 3, output_arg=2)
def divide(x, y, context):
    if y == 0:
        raise PrologError('evaluation_error', 'zero_divisor')
    return x / y
```

Now `divide(10, 0, R)` returns `✗ evaluation_error: zero_divisor`.

**Rule:** silent `fail` — for "no result." `PrologError` — for a real error (division by zero, wrong type, unavailable service).

### 29. Agent Programming

`Agent` is a class from `prolog.agent` that combines:

- a Prolog **program** (facts and rules);
    
- **actions** (Python predicates);
    
- **context** (external resources).
    

A simple agent:

```python
from prolog.agent import Agent
from prolog.actions import simple_action, effector_action
# Program
agent = Agent("""
start :-
    log_msg('Agent started'),
    double(21, X),
    log_msg(X).
""")
# Actions
@effector_action('log_msg', 1)
def log_msg(msg, context):
    print(f"[AGENT] {msg}")
@simple_action('double', 2, output_arg=1)
def double(x, context):
    return x * 2
# Registration
agent.register_action(log_msg)
agent.register_action(double)
# Run
result = agent.query("start.")
print(result)
# [AGENT] Agent started
# [AGENT] 42
```

### 30. The `Agent` Class API

#### 30.1. Creation

```python
agent = Agent(
    program="parent(ivan, alexey).",    # Prolog program text
    pl_file="",                         # or a path to a .pl file
    context={"llm": client},            # external resources
)
```

#### 30.2. Registering Actions

```python
agent.register_action(func)             # a single function
agent.register_action_module(module)    # an entire module
agent.list_actions()                    # list of registered actions
```

#### 30.3. Queries

```python
# All solutions
result = agent.query("parent(ivan, X).")
# {'solution': [{'X': 'alexey'}, {'X': 'maria'}], ...}
# First solution
result = agent.query_one("parent(ivan, X).")
# {'X': 'alexey'}
# Streaming (for large results)
for solution in agent.stream("parent(ivan, X)."):
    print(solution)
# With tracing
result, explanation = agent.query("parent(ivan, X).", with_explanation=True)
print(explanation.get_proof_tree())
```

#### 30.4. Working with the Knowledge Base

```python
agent.add_fact("parent(ivan, alexey).")
agent.add_facts("""
    parent(ivan, maria).
    parent(alexey, petr).
""")
agent.retract_fact("parent(ivan, alexey).")
agent.retract_all("parent(ivan, _).")
agent.save_kb("kb.pl", format="prolog")
agent.load_kb("kb.pl", format="prolog", clear=True)
agent.clear_kb()
```

#### 30.5. Context

```python
agent.context['llm'] = client
agent.context['target_x'] = 10.0
value = agent.context.get('llm')
```

#### 30.6. Built-in Agent Predicates

`Agent` automatically registers two useful predicates:

**`set_state/1`** — replace all facts with the same functor/arity with one new fact:

```prolog
set_state(mode(search)).
```

If the base had `mode(idle).`, it is removed. `mode(search).` remains.

**`update_state/1`** — add a fact without removing old ones:

```prolog
update_state(observed(bravo, search, 10.0, 5.0, 17)).
```

Now the base has the record `observed(...)`, and the old `observed(...)` are preserved.

This is a classic pattern for agents: `set_state` — "set the current mode," `update_state` — "add an observation."

### 31. Example: An LLM Agent in 15 Lines

`my_actions.py`:

```python
from prolog.actions import simple_action
@simple_action('ask_llm', 2, output_arg=1)
def ask_llm(prompt, context):
    llm = context.get('llm')
    if llm is None:
        return "(no llm)"
    return llm.invoke(prompt).content
```

`kb.pl`:

```prolog
ask(P, R) :- ask_llm(P, R).
ask_many([], []).
ask_many([P | Ps], [R | Rs]) :-
    ask(P, R),
    ask_many(Ps, Rs).
```

IDE:

1. 📂 Open the folder with these files.
    
2. 🔌 Context: `llm` = `__import__('langchain_openai').ChatOpenAI(model='gpt-4o-mini')` → ➕ → 🔄 Apply.
    
3. Query: `ask('What is the capital of France?', R).` → ▶ Run → `R = 'Paris'`.
    
4. Or: `ask_many(['2+2?', 'What is the capital of Italy?'], L).` → `L = ['4', 'Rome']`.
    

**What happened:** Prolog described the structure ("for each prompt — its own answer"), Python performed the action (a real LLM call). Logic and computation are separated — as it should be in a hybrid system.

### 32. Legacy: `agent_action` and Direct Predicate Programming

If you need full control over the predicate signature (`goal, database, subst, depth`), there is the legacy decorator `agent_action`:

```python
from prolog.agent import agent_action
@agent_action('my_pred', 1, description="...", action_type="generic")
def my_pred(goal, database, subst, depth):
    # goal.arguments[0] — the Prolog argument
    arg = goal.arguments[0].substitute(subst)
    # ... your logic
    yield subst.copy()
```

This is a **low-level** API — you get control over the substitution, depth, and backtracking point. It is rarely needed; in 95% of cases `simple_action` / `effector_action` / `generator_action` is enough.

**Do not confuse:**

- `agent_action` — low-level, you **yourself** write all the logic of the predicate.
    
- `simple_action` / `effector_action` / `generator_action` — high-level, you write an **ordinary Python function**, and the decorator turns it into a predicate.
    

### 33. Best Practices

1. **Only decorators + functions** in `my_actions.py`, with no side effects at the module level (no `print` on import, no network calls).
    
2. **`context`** is the only channel to external resources. Do not use global variables.
    
3. **Input arguments** must be instantiated — check `X` before `double(X, Y)`.
    
4. **`None` from `simple_action`** means failure. Use it consciously.
    
5. **`PrologError`** — for real errors. Otherwise — a silent `fail`.
    
6. **One function — one responsibility.** Do not make a `simple_action` with five branches.
    
7. **Naming:** the Prolog predicate name is snake_case, as in Python. `distance_to_target`, not `DistanceToTarget`.
    
8. **Document** via `description=` or a docstring — this is visible in `⚡ Registered actions` in the IDE.
    
9. **Reimport** — by `mtime_ns + size`. If you edit the file and do not see changes, press 🔄 Reload in the IDE.
---
## Part IV. Practical Patterns

### 34. Recursive List Traversal

The classic two-clause pattern:

```prolog
predicate([], BaseCase).
predicate([H | T], Result) :-
    do_something(H, ...),
    predicate(T, ...).
```

Example — the sum of elements:

```prolog
sum([], 0).
sum([H | T], S) :-
    sum(T, S1),
    S is H + S1.
```

The tail version:

```prolog
sum(L, S) :- sum_acc(L, 0, S).
sum_acc([], Acc, Acc).
sum_acc([H | T], Acc, S) :-
    Acc1 is Acc + H,
    sum_acc(T, Acc1, S).
```

### 35. Generate and Test

Prolog lets you **generate** solutions and then **select** the suitable ones:

```prolog
% All pairs (X, Y) from the list where X < Y
pairs([], []).
pairs([H | T], Result) :- ...
```

Another example — a number generator:

```prolog
between_gen(Low, High, Low) :- Low =< High.
between_gen(Low, High, X) :-
    Low < High,
    Low1 is Low + 1,
    between_gen(Low1, High, X).
```

```prolog
?- between_gen(1, 5, X).
X = 1 ; X = 2 ; X = 3 ; X = 4 ; X = 5.
```

### 36. `findall` — Collect All Solutions

```prolog
findall(Template, Goal, List).
```

Collects all `Template`s for which `Goal` is provable into the list `List`:

```prolog
?- findall(X, parent(ivan, X), Children).
Children = [alexey, maria].
```

```prolog
?- findall(X-Y, parent(X, Y), Pairs).
Pairs = [ivan-alexey, ivan-maria, alexey-petr, ...].
```

`-` inside a term is just a functor. `X-Y` is a pair.

### 37. `assert` / `retract` — Dynamic Database

Sometimes you need to **modify** the knowledge base during execution:

```prolog
?- assertz(fact(5)).
true.

?- fact(X).
X = 1 ; X = 2 ; X = 3 ; X = 4 ; X = 5.
```

- `assertz` — add to the end.
- `asserta` — add to the beginning.
- `retract` — remove the first match.
- `retractall` — remove all.

Caution: `assert`/`retract` rebuild the index. In a loop this is noticeably slow.

### 38. Working with Terms

Prolog can decompose terms into parts:

```prolog
?- functor(parent(ivan, alexey), F, A).
F = parent, A = 2.

?- arg(1, parent(ivan, alexey), X).
X = ivan.

?- parent(ivan, alexey) =.. L.
L = [parent, ivan, alexey].
```

This is useful for metaprogramming (generating rules, processing expressions).

### 39. Debugging

**Write** is the simplest tool:

```prolog
debug(X) :- write('X = '), write(X), nl.
```

```prolog
run :- parent(ivan, X), debug(X), fail.
```

`fail` at the end guarantees that the engine will enumerate **all** solutions.

**Trace** in the IDE is the next step. Set a breakpoint on the predicate that "does not work" — and see which arguments arrive.

**`📚 KB`** — check how the rule **was parsed**. Often errors in parentheses or operator priorities are visible right there.

---

## Part IV. Common Beginner Mistakes

### 40. Syntax Errors

**Forgot the period at the end.** `parent(ivan, alexey)` is not a clause. You need `parent(ivan, alexey).`.

**Forgot the comma between goals.** `parent(X, Y) parent(Y, Z)` — a syntax error. You need `parent(X, Y), parent(Y, Z)`.

**Forgot the parentheses around `->`.** `( Cond -> Then ; Else )` — if you remove the outer parentheses, the priorities shift.

**Confusing `=` and `is`.** `X = 2 + 3` does not evaluate. You need `X is 2 + 3`.

### 41. Logical Errors

**Forgot the base case.** `length([], 0). length([_|T], N) :- ...` — without the first fact `length([], 0)` it will go into infinite recursion.

**Wrong goal order.** `length(X, N), N > 0` — first `length` generates an infinite number of solutions, then checks `N > 0`. The user will not wait. Correct: `N > 0, length(X, N)` — first instantiate `N`.

**Did not bind a variable before `\+`.** `\+ parent(X, Y)` with unbound `X` and `Y` — "is there any parent at all?". Not what you may have meant.

**Forgot `!` where it is needed.** `max(X, Y, X) :- X >= Y.` without `!` — if `max(5, 3, M)`, the user may get both `M = 5` (correct) and `M = 3` (incorrect).

**Placed `!` in the wrong place.** `p :- a, !, b.` — `!` cuts off the alternatives of `a`, but not of `p` as a whole. If you want "if `a` succeeds — only this rule," place `!` after `a`, but before any other goals that may yield alternatives.
### 42. Errors in Actions

**Forgot to register.** `agent.register_action(double)` — without it the predicate will not appear. In the IDE with `my_actions.py` registration is automatic, but with explicit `Agent` creation it is needed.

**Forgot to put it in context.** `context['llm']` — if `llm` is not set, the action returns `(no llm)` or crashes. Check via 🔌 Context.

**Passed a free variable.** `double(X, Y)` with unbound `X` → `fail`. Always `X = 21, double(X, Y)`.

**Exception instead of PrologError.** `x / 0` without a check → silent `fail`. If you want an explicit error — `raise PrologError(...)`.

**Forgot about reimport.** You changed `my_actions.py`, did not press 🔄 — old behavior.
### 43. Performance Errors

**Non-tail recursion on large data.** `sum([H|T], S) :- sum(T, S1), S is H + S1.` — the stack grows. Use an accumulator.

**`assert` in a loop.** Each `assert` rebuilds the index. If you need many facts — `assertz` in batches or via `agent.add_facts()`.

**`findall` on a large result.** Collects **all** solutions into memory. If there is only one solution — use `once` or `call_cleanup`.

**No index on the first argument.** `parent(X, Y) :- ...` — all rules are in one bucket. `parent(ivan, Y) :- ...` — only `ivan` is in the fast bucket.

---

## Part V. What's Next

You have learned:

- declarative thinking;
- syntax (terms, unification, facts, rules);
- arithmetic and lists;
- recursion, tail recursion, cut, negation;
- patterns (`findall`, `assert`/`retract`, working with terms).

Now you can:

- open the **family** example in the IDE (📚) and experiment;
- write your own program with recursion (e.g., sorting a list);
- figure out **my_actions.py** — add a Python function as a Prolog predicate;
- enable **semantic** and try fuzzy matching;
- figure out **tracing** and **breakpoints** — this is the main debugging tool.

**What is **not** included in PyAgentProlog:** modules, DCG, `catch/throw`, CLP(FD), `sort/2`, `between/3`, `atom_length/2`. If something is missing — use Python actions (`my_actions.py`), they cover almost anything.

**Further reading:**

- "Learn Prolog Now!" — a free online textbook.
- "The Art of Prolog" — a classic book.
- SWI-Prolog documentation — a reference for built-in predicates (most of them are also in PyAgentProlog).

---

## Cheat Sheet

```prolog
% Fact
parent(ivan, alexey).

% Rule
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Lists
[H | T]             % head H, tail T
[]                  % empty list
[a, b, c]           % [a, b, c]

% Arithmetic
X is 2 + 3          % evaluate
X =:= Y             % numeric equality

% Comparison
X = Y               % unification
X == Y              % identity (without binding)
X \= Y              % does not unify

% Control
!                   % cut
\+ Goal             % negation
( Cond -> Then ; Else )

% Meta
findall(T, G, L)    % all T where G, into list L
call(G)             % call G as a goal
once(G)             % first solution of G

% Dynamic database
assertz(fact(...))  % add
retract(fact(...))  % remove the first
```

Good luck! Open the IDE, load the `family` example, type `grandparent(ivan, X).` and press **▶ Run**. Then — experiment.