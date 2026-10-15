"""
prolog_semantic_demo.py

Large demonstration of the advantages of semantic unification.

Domain: a knowledge base about countries — capitals, currencies, languages.

Two backends:
  backend='st'     — via sentence-transformers (recommended).
  backend='ollama' — via Ollama.

Matching layers:
  exact  — strings matched after normalization;
  typo   — lev_ratio <= lev_typo_max (typos, cases) — without the model;
  hard   — cos >= threshold;
  soft   — soft_cos_min <= cos < threshold and lev_ratio <= lev_max.

Parameters are set in CONFIG (directly in the code). Env variables override.

Suppressing ST/HF output:
  By default, "[ST] resolved model path" messages and progress bars
  "Loading weights" are disabled. Enable them with: PROLOG_ST_VERBOSE=1
  or CONFIG["st_verbose"] = True.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prolog.solver import Solver
import prolog.embedding_unify as eu


# =====================================================================
# CONFIGURATION — edit here
# =====================================================================
CONFIG = {
    "backend": "st",
    "model": "D:/hf_embedding/paraphrase-multilingual-mpnet",
    "local_dir": "d:/hf_embedding",
    "cache_folder": None,
    "device": None,
    "local_files_only": True,

    # Show sentence-transformers messages and progress bars.
    # False (default) — silent; True — print the model path and bars.
    "st_verbose": False,

    "threshold": 0.75,          # hard cos threshold
    "soft_cos_min": 0.55,       # lower bound of the soft corridor
    "lev_max": 0.35,            # maximum lev_ratio in the soft corridor
    "lev_typo_max": 0.20,       # "typo" threshold (Layer 2a)

    "text_prefix": None,

    "ollama_url": "http://localhost:11434",
    "max_cache": 10000,
}


def _apply_env(cfg):
    def _get(name, current):
        v = os.environ.get(name)
        return v if v is not None else current

    def _bool(name, current):
        v = os.environ.get(name)
        if v is None:
            return current
        return v.lower() in ("1", "true", "yes", "on")

    def _float(name, current):
        v = os.environ.get(name)
        return float(v) if v is not None else current

    def _int(name, current):
        v = os.environ.get(name)
        return int(v) if v is not None else current

    cfg["backend"] = str(_get("PROLOG_SEMANTIC_BACKEND", cfg["backend"])).lower()
    cfg["model"] = _get("PROLOG_SEMANTIC_MODEL", cfg["model"])
    cfg["local_dir"] = _get("PROLOG_SEMANTIC_LOCAL_DIR", cfg["local_dir"]) or None
    cfg["cache_folder"] = _get("PROLOG_ST_CACHE_FOLDER", cfg["cache_folder"]) or None
    cfg["device"] = _get("PROLOG_ST_DEVICE", cfg["device"]) or None
    cfg["local_files_only"] = _bool("PROLOG_ST_LOCAL_FILES_ONLY", cfg["local_files_only"])
    cfg["st_verbose"] = _bool("PROLOG_ST_VERBOSE", cfg["st_verbose"])
    cfg["threshold"] = _float("PROLOG_SEMANTIC_THRESHOLD", cfg["threshold"])
    cfg["soft_cos_min"] = _float("PROLOG_SEMANTIC_SOFT_COS_MIN", cfg["soft_cos_min"])
    cfg["lev_max"] = _float("PROLOG_SEMANTIC_LEV_MAX", cfg["lev_max"])
    cfg["lev_typo_max"] = _float("PROLOG_SEMANTIC_LEV_TYPO_MAX", cfg["lev_typo_max"])

    prefix_env = os.environ.get("PROLOG_SEMANTIC_PREFIX")
    if prefix_env is not None:
        cfg["text_prefix"] = None if prefix_env == "auto" else prefix_env

    cfg["ollama_url"] = _get("PROLOG_OLLAMA_URL", cfg["ollama_url"])
    cfg["max_cache"] = _int("PROLOG_SEMANTIC_MAX_CACHE", cfg["max_cache"])
    return cfg


CONFIG = _apply_env(CONFIG)


# =====================================================================
# Prolog program
# =====================================================================
PROGRAM = """
country(russia, europe, 144).
country(germany, europe, 83).
country(france, europe, 67).
country(china, asia, 1400).
country(japan, asia, 125).
country(india, asia, 1400).
country(brazil, america, 215).

capital(russia, moscow).
capital(germany, berlin).
capital(france, paris).
capital(china, beijing).
capital(japan, tokyo).
capital(india, delhi).
capital(brazil, brasilia).

currency(russia, ruble).
currency(germany, euro).
currency(france, euro).
currency(china, yuan).
currency(japan, yen).
currency(india, rupee).
currency(brazil, real).

language(russia, russian).
language(germany, german).
language(france, french).
language(china, chinese).
language(japan, japanese).
language(india, hindi).
language(brazil, portuguese).

european_country(X) :- country(X, europe, _).
asian_country(X)    :- country(X, asia, _).
american_country(X) :- country(X, america, _).

large_country(X) :- country(X, _, N), N > 100.
small_country(X) :- country(X, _, N), N =< 100.

uses_euro(X) :- currency(X, euro).

country_by_capital(Country, Capital) :- capital(Country, Capital).

large_asian(X) :-
    country(X, asia, N),
    N > 100.

european_euro(X) :-
    european_country(X),
    uses_euro(X).
"""


QUERIES = [
    (1, "CONTROL",   "Exact match: 'moscow' == 'moscow'",             "capital(russia, moscow)."),
    (2, "CASE",      "Case: 'Moscow' vs 'moscow'",                    "capital(russia, 'Moscow')."),
    (3, "MORPH",     "Morphology: 'Moscow City' vs 'moscow'",         "capital(russia, 'Moscow City')."),
    (4, "CROSSLING", "Cross-language: 'Москва' vs 'moscow'",          "capital(russia, 'Москва')."),
    (5, "TYPO",      "Typo: 'Moskow' vs 'moscow'",                    "capital(russia, 'Moskow')."),
    (6, "SYNONYM",   "Abbreviation: 'RF' vs 'russia'",                "capital('RF', X)."),
    (7, "CASE",      "Case + possessive: 'Paris\\'s' vs 'paris'",     "capital(france, 'Paris\\'s')."),
    (8, "CASE",      "Find a country by capital 'Tokyo'",             "country_by_capital(X, 'Tokyo')."),
    (9, "CASE",      "Large country: 'China' vs 'china'",             "large_country('China')."),
    (10, "CASE",     "Asian country: 'Japan' vs 'japan'",             "asian_country('Japan')."),
    (11, "CASE",     "Uses euro: 'Germany' vs 'germany'",             "uses_euro('Germany')."),
    (12, "CROSSLING","Old spelling: 'Peking' vs 'beijing'",           "capital(china, 'Peking')."),
    (13, "NEGATIVE", "False: 'London' is not the capital of russia",  "capital(russia, 'London')."),
    (14, "NEGATIVE", "False: 'moon' is not a country",                "country('moon', _, _)."),
]


# =====================================================================
# Access to CONFIG
# =====================================================================
def _resolve_prefix():
    tp = CONFIG["text_prefix"]
    if tp is not None:
        return tp
    return eu.detect_text_prefix(eu.resolve_model_id(CONFIG["model"]))


def _semantic_on_kwargs():
    kw = dict(
        threshold=CONFIG["threshold"],
        model=CONFIG["model"],
        text_prefix=CONFIG["text_prefix"],
        backend=CONFIG["backend"],
        soft_cos_min=CONFIG["soft_cos_min"],
        lev_max=CONFIG["lev_max"],
        lev_typo_max=CONFIG["lev_typo_max"],
        max_cache=CONFIG["max_cache"],
    )
    if CONFIG["backend"] == "ollama":
        kw["base_url"] = CONFIG["ollama_url"]
    else:
        kw["local_dir"] = CONFIG["local_dir"]
        kw["st_cache_folder"] = CONFIG["cache_folder"]
        kw["st_device"] = CONFIG["device"]
        kw["st_local_files_only"] = CONFIG["local_files_only"]
        kw["st_verbose"] = CONFIG["st_verbose"]
    return kw


_backend_ok = None


def check_backend():
    global _backend_ok
    if _backend_ok is not None:
        return _backend_ok

    print(f"[CHECK] backend='{CONFIG['backend']}'")
    if CONFIG["backend"] == "ollama":
        print(f"[CHECK] Ollama: model='{CONFIG['model']}', "
              f"url='{CONFIG['ollama_url']}'")
    else:
        print(f"[CHECK] sentence-transformers: model='{CONFIG['model']}'")
        if CONFIG["local_dir"]:
            print(f"[CHECK]   local_dir='{CONFIG['local_dir']}'")
        if CONFIG["cache_folder"]:
            print(f"[CHECK]   cache_folder='{CONFIG['cache_folder']}'")

    try:
        if CONFIG["backend"] == "ollama":
            fn = eu.make_ollama_embed_fn(
                model=CONFIG["model"],
                base_url=CONFIG["ollama_url"],
            )
            v = fn(_resolve_prefix() + "ping")
        elif CONFIG["backend"] == "st":
            fn = eu.make_sentence_transformer_embed_fn(
                model=CONFIG["model"],
                local_dir=CONFIG["local_dir"],
                cache_folder=CONFIG["cache_folder"],
                device=CONFIG["device"],
                local_files_only=CONFIG["local_files_only"],
                verbose=CONFIG["st_verbose"],
            )
            v = fn("ping")
        else:
            print(f"[CHECK] Unknown backend='{CONFIG['backend']}'")
            _backend_ok = False
            return False

        _backend_ok = (v is not None and len(v) > 0)
        if _backend_ok:
            print(f"[CHECK] OK, vector dimension: {len(v)}")
    except Exception as e:
        print(f"[CHECK] Unavailable: {type(e).__name__}: {e}")
        _backend_ok = False
    return _backend_ok


def run_query(solver, query_text):
    try:
        result = solver.find_solutions(query_text)
    except Exception as e:
        print(f"    [error] {type(e).__name__}: {e}")
        return False, 0

    if result is True:
        return True, 1
    if result is False or result is None:
        return False, 0
    if isinstance(result, list):
        return (len(result) > 0), len(result)
    return False, 0


def run_all(solver, verbose=True):
    results = []
    for num, category, desc, query in QUERIES:
        if verbose:
            print(f"  [{num:>2}] {desc}")
        t0 = time.time()
        ok, cnt = run_query(solver, query)
        dt = time.time() - t0
        results.append((ok, cnt))
        if verbose:
            mark = "✓" if ok else "✗"
            print(f"       {mark} solutions: {cnt}  ({dt*1000:.0f} ms)")
    return results


def main():
    actual_prefix = _resolve_prefix()

    print("=" * 78)
    print("DEMO: Prolog with strict vs semantic unification")
    print("=" * 78)
    print(f"Backend:   {CONFIG['backend']}")
    print(f"Model:     {CONFIG['model']}")
    if CONFIG["backend"] == "ollama":
        print(f"Ollama:    {CONFIG['ollama_url']}")
    else:
        print(f"Local dir: {CONFIG['local_dir'] or '(not set)'}")
        print(f"Cache:     {CONFIG['cache_folder'] or CONFIG['local_dir'] or '(HF default)'}")
        print(f"Device:    {CONFIG['device'] or '(auto)'}")
        print(f"Offline:   {CONFIG['local_files_only']}")
        print(f"ST verbose:{CONFIG['st_verbose']}")
    print(f"Threshold: {CONFIG['threshold']}")
    print(f"Corridor:  lev_typo_max={CONFIG['lev_typo_max']}, "
          f"soft_cos_min={CONFIG['soft_cos_min']}, lev_max={CONFIG['lev_max']}")
    print(f"Prefix:    {actual_prefix!r}"
          + (" (auto)" if CONFIG['text_prefix'] is None else " (set explicitly)"))
    print(f"Queries:   {len(QUERIES)}")
    print("=" * 78)

    if not check_backend():
        print("\nBackend is unavailable. Check:")
        if CONFIG["backend"] == "ollama":
            print(f"  - is `ollama serve` running;")
            print(f"  - is the model loaded: `ollama pull {CONFIG['model']}`")
        else:
            print(f"  - is sentence-transformers installed: "
                  f"`pip install sentence-transformers`;")
            if CONFIG["local_dir"]:
                print(f"  - does '{CONFIG['local_dir']}' exist and contain the model folder;")
            print(f"  - or is HF Hub reachable for {CONFIG['model']}.")
        return 1

    print("\nLoading the knowledge base…")
    solver = Solver(PROGRAM)
    print(f"  Facts and rules: {len(solver.database.rules)}")

    # ---------- Stage 1: strict ----------
    print("\n" + "=" * 78)
    print("STAGE 1 — STRICT MODE (semantic_off)")
    print("=" * 78)
    eu.semantic_off()
    strict_results = run_all(solver, verbose=True)

    # ---------- Stage 2: semantic ----------
    print("\n" + "=" * 78)
    print("STAGE 2 — SEMANTIC MODE (semantic_on)")
    print("=" * 78)
    eu.semantic_off()

    on_kwargs = _semantic_on_kwargs()
    eu.semantic_on(**on_kwargs)
    semantic_results = run_all(solver, verbose=True)

    # ---------- Summary table ----------
    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    fmt = "{:<3} {:<10} {:<48} {:<9} {:<12} {}"
    print(fmt.format("#", "Category", "Query", "Strict", "Semantics", "Δ"))
    print("-" * 100)

    wins = 0
    losses = 0
    for i, (num, category, desc, query) in enumerate(QUERIES):
        s_ok, s_cnt = strict_results[i]
        m_ok, m_cnt = semantic_results[i]
        s_str = f"OK ({s_cnt})" if s_ok else "FAIL"
        m_str = f"OK ({m_cnt})" if m_ok else "FAIL"

        if s_ok and not m_ok:
            delta = "!! regression"
            losses += 1
        elif not s_ok and m_ok:
            delta = "+ semantics won"
            wins += 1
        else:
            delta = "="

        desc_short = (desc[:45] + "...") if len(desc) > 48 else desc
        print(fmt.format(num, category, desc_short, s_str, m_str, delta))

    print("-" * 100)
    strict_ok_total = sum(1 for ok, _ in strict_results if ok)
    semantic_ok_total = sum(1 for ok, _ in semantic_results if ok)
    print(f"Strict mode:           {strict_ok_total} / {len(QUERIES)}")
    print(f"Semantic mode:         {semantic_ok_total} / {len(QUERIES)}")
    print(f"Semantic wins:         {wins}")
    print(f"Regressions:           {losses}")

    # ---------- Statistics ----------
    u = eu.semantic_get_unifier()
    if u is not None:
        print("\n" + "=" * 78)
        print("SEMANTIC UNIFIER STATISTICS")
        print("=" * 78)
        st = u.stats
        print(f"  Backend:                       {u.backend}")
        print(f"  Total unification calls:       {st['calls']}")
        print(f"  Strict successes:              {st['strict_success']}")
        print(f"  Exact successes (normalization):{st.get('exact_success', 0)}")
        print(f"  Typo successes (typos/cases):  {st.get('typo_success', 0)}")
        print(f"  Hard successes (cos >= thresh):{st['embedding_success']}")
        print(f"  Soft successes (Levenshtein):  {st.get('soft_success', 0)}")
        print(f"  Failures:                      {st['failures']}")
        cs = u.cache.stats()
        print(f"  Embedding cache:               {cs['size']} entries")
        print(f"    hits:                        {cs['hits']}")
        print(f"    misses:                      {cs['misses']}")
        print(f"    hit rate:                    {cs['hit_rate']:.1%}")
        print(f"  Text prefix:                   {u.text_prefix!r}")
        print(f"  lev_typo_max:                  {u.lev_typo_max}")

    # ---------- Cosine table ----------
    print("\n" + "=" * 78)
    print("COSINE SIMILARITY OF KEY PAIRS (what the model sees)")
    print("=" * 78)
    if eu.semantic_get_unifier() is None:
        eu.semantic_on(**on_kwargs)
    u = eu.semantic_get_unifier()

    pairs = [
        ("Moscow", "moscow", "case"),
        ("Moscow City", "moscow", "morphology"),
        ("Paris's", "paris", "possessive"),
        ("Москва", "moscow", "cross-language"),
        ("Peking", "beijing", "old spelling"),
        ("Moskow", "moscow", "typo"),
        ("RF", "russia", "abbreviation"),
        ("Tokyo", "tokyo", "case"),
        ("China", "china", "case"),
        ("Japan", "japan", "case"),
        ("Germany", "germany", "case"),
        ("euro", "euro", "exact"),
        ("london", "moscow", "FALSE"),
        ("moon", "russia", "FALSE"),
        ("cat", "dog", "FALSE"),
    ]

    header = f"{'Word 1':<14} {'Word 2':<14} {'cos':>8}   {'lev':>6}   {'layer':<7} type"
    print(header)
    print("-" * 78)
    for w1, w2, ptype in pairs:
        s1 = eu._normalize_text(w1)
        s2 = eu._normalize_text(w2)
        lev = eu._lev_ratio(s1, s2)
        if s1 == s2:
            cos = 1.0
            layer = "exact"
        else:
            v1 = u._embed(s1)
            v2 = u._embed(s2)
            cos = u._cosine(v1, v2)
            if lev <= u.lev_typo_max:
                layer = "typo"
            elif cos >= CONFIG["threshold"]:
                layer = "hard"
            elif cos >= u.soft_cos_min and lev <= u.lev_max:
                layer = "soft"
            else:
                layer = "—"
        print(f"{w1:<14} {w2:<14} {cos:>8.4f}   {lev:>6.3f}   {layer:<7} {ptype}")

    # ---------- Conclusion ----------
    print("\n" + "=" * 78)
    print("CONCLUSION")
    print("=" * 78)
    if wins > 0 and losses == 0:
        print(f"✓ Semantic mode won in {wins} queries and lost in none.")
    elif losses > 0:
        print(f"⚠ Regressions detected: {losses}. The threshold may be too low.")
    else:
        print("All queries produced the same result — increase the complexity of the scenario.")

    eu.semantic_off()
    return 0


if __name__ == "__main__":
    sys.exit(main())