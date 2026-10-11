"""
prolog_semantic_demo.py

Большая демонстрация преимуществ семантической унификации.

Домен: база знаний о странах — столицы, валюты, языки.

Два backend'а:
  backend='st'     — через sentence-transformers (рекомендуется).
  backend='ollama' — через Ollama.

Слои сопоставления:
  exact  — строки совпали после нормализации;
  typo   — lev_ratio <= lev_typo_max (опечатки, падежи) — без модели;
  hard   — cos >= threshold;
  soft   — soft_cos_min <= cos < threshold и lev_ratio <= lev_max.

Параметры задаются в CONFIG (прямо в коде). Env-переменные переопределяют.

Подавление вывода ST/HF:
  По умолчанию сообщения "[ST] resolved model path" и прогресс-бары
  "Loading weights" отключены. Включить их: PROLOG_ST_VERBOSE=1
  или CONFIG["st_verbose"] = True.
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from prolog.solver import Solver
import prolog.embedding_unify as eu


# =====================================================================
# КОНФИГУРАЦИЯ — правь здесь
# =====================================================================
CONFIG = {
    "backend": "st",
    "model": "D:/hf_embedding/paraphrase-multilingual-mpnet",
    "local_dir": "d:/hf_embedding",
    "cache_folder": None,
    "device": None,
    "local_files_only": True,

    # Показывать сообщения sentence-transformers и прогресс-бары.
    # False (по умолчанию) — тихо; True — печатать путь к модели и полосы.
    "st_verbose": False,

    "threshold": 0.75,          # hard-порог cos
    "soft_cos_min": 0.55,       # нижняя граница soft-коридора
    "lev_max": 0.35,            # максимальное lev_ratio в soft-коридоре
    "lev_typo_max": 0.20,       # порог «опечаток» (Layer 2a)

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
# Программа Prolog
# =====================================================================
PROGRAM = """
страна(россия, европа, 144).
страна(германия, европа, 83).
страна(франция, европа, 67).
страна(китай, азия, 1400).
страна(япония, азия, 125).
страна(индия, азия, 1400).
страна(бразилия, америка, 215).

столица(россия, москва).
столица(германия, берлин).
столица(франция, париж).
столица(китай, пекин).
столица(япония, токио).
столица(индия, дели).
столица(бразилия, бразилиа).

валюта(россия, рубль).
валюта(германия, евро).
валюта(франция, евро).
валюта(китай, юань).
валюта(япония, иена).
валюта(индия, рупия).
валюта(бразилия, реал).

язык(россия, русский).
язык(германия, немецкий).
язык(франция, французский).
язык(китай, китайский).
язык(япония, японский).
язык(индия, хинди).
язык(бразилия, португальский).

европейская_страна(X) :- страна(X, европа, _).
азиатская_страна(X)   :- страна(X, азия, _).
американская_страна(X):- страна(X, америка, _).

большая_страна(X) :- страна(X, _, N), N > 100.
малая_страна(X)   :- страна(X, _, N), N =< 100.

использует_евро(X) :- валюта(X, евро).

страна_по_столице(Страна, Столица) :- столица(Страна, Столица).

большая_азиатская(X) :-
    страна(X, азия, N),
    N > 100.

европейская_евро(X) :-
    европейская_страна(X),
    использует_евро(X).
"""


QUERIES = [
    (1, "CONTROL",   "Точное совпадение: 'москва' == 'москва'",       "столица(россия, москва)."),
    (2, "CASE",      "Регистр: 'Москва' vs 'москва'",                  "столица(россия, 'Москва')."),
    (3, "MORPH",     "Падеж: 'Москве' vs 'москва'",                    "столица(россия, 'Москве')."),
    (4, "CROSSLING", "Кросс-язычность: 'Moscow' vs 'москва'",          "столица(россия, 'Moscow')."),
    (5, "TYPO",      "Опечатка: 'Масква' vs 'москва'",                 "столица(россия, 'Масква')."),
    (6, "SYNONYM",   "Аббревиатура: 'РФ' vs 'россия'",                 "столица('РФ', X)."),
    (7, "CASE",      "Регистр + падеж: 'Париже' vs 'париж'",           "столица(франция, 'Париже')."),
    (8, "CASE",      "Найти страну по столице 'Токио'",                "страна_по_столице(X, 'Токио')."),
    (9, "CASE",      "Большая страна: 'Китай' vs 'китай'",             "большая_страна('Китай')."),
    (10, "CASE",     "Азиатская страна: 'Япония' vs 'япония'",         "азиатская_страна('Япония')."),
    (11, "CASE",     "Использует евро: 'Германия' vs 'германия'",      "использует_евро('Германия')."),
    (12, "CROSSLING","Кросс-язычность: 'Beijing' vs 'пекин'",          "столица(китай, 'Beijing')."),
    (13, "NEGATIVE", "Ложное: 'Лондон' не столица россии",             "столица(россия, 'Лондон')."),
    (14, "NEGATIVE", "Ложное: 'луна' не страна",                       "страна('луна', _, _)."),
]


# =====================================================================
# Доступ к CONFIG
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

    print(f"[ПРОВЕРКА] backend='{CONFIG['backend']}'")
    if CONFIG["backend"] == "ollama":
        print(f"[ПРОВЕРКА] Ollama: model='{CONFIG['model']}', "
              f"url='{CONFIG['ollama_url']}'")
    else:
        print(f"[ПРОВЕРКА] sentence-transformers: model='{CONFIG['model']}'")
        if CONFIG["local_dir"]:
            print(f"[ПРОВЕРКА]   local_dir='{CONFIG['local_dir']}'")
        if CONFIG["cache_folder"]:
            print(f"[ПРОВЕРКА]   cache_folder='{CONFIG['cache_folder']}'")

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
            print(f"[ПРОВЕРКА] Неизвестный backend='{CONFIG['backend']}'")
            _backend_ok = False
            return False

        _backend_ok = (v is not None and len(v) > 0)
        if _backend_ok:
            print(f"[ПРОВЕРКА] OK, размерность вектора: {len(v)}")
    except Exception as e:
        print(f"[ПРОВЕРКА] Недоступно: {type(e).__name__}: {e}")
        _backend_ok = False
    return _backend_ok


def run_query(solver, query_text):
    try:
        result = solver.find_solutions(query_text)
    except Exception as e:
        print(f"    [ошибка] {type(e).__name__}: {e}")
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
            print(f"       {mark} решений: {cnt}  ({dt*1000:.0f} ms)")
    return results


def main():
    actual_prefix = _resolve_prefix()

    print("=" * 78)
    print("ДЕМОНСТРАЦИЯ: Prolog со строгой vs семантической унификацией")
    print("=" * 78)
    print(f"Backend:   {CONFIG['backend']}")
    print(f"Модель:    {CONFIG['model']}")
    if CONFIG["backend"] == "ollama":
        print(f"Ollama:    {CONFIG['ollama_url']}")
    else:
        print(f"Local dir: {CONFIG['local_dir'] or '(не задан)'}")
        print(f"Cache:     {CONFIG['cache_folder'] or CONFIG['local_dir'] or '(HF default)'}")
        print(f"Device:    {CONFIG['device'] or '(auto)'}")
        print(f"Offline:   {CONFIG['local_files_only']}")
        print(f"ST verbose:{CONFIG['st_verbose']}")
    print(f"Порог:     {CONFIG['threshold']}")
    print(f"Коридор:   lev_typo_max={CONFIG['lev_typo_max']}, "
          f"soft_cos_min={CONFIG['soft_cos_min']}, lev_max={CONFIG['lev_max']}")
    print(f"Префикс:   {actual_prefix!r}"
          + (" (авто)" if CONFIG['text_prefix'] is None else " (задан явно)"))
    print(f"Запросов:  {len(QUERIES)}")
    print("=" * 78)

    if not check_backend():
        print("\nBackend недоступен. Проверьте:")
        if CONFIG["backend"] == "ollama":
            print(f"  - запущен ли `ollama serve`;")
            print(f"  - загружена ли модель: `ollama pull {CONFIG['model']}`")
        else:
            print(f"  - установлен ли sentence-transformers: "
                  f"`pip install sentence-transformers`;")
            if CONFIG["local_dir"]:
                print(f"  - существует ли '{CONFIG['local_dir']}' и папка модели в ней;")
            print(f"  - или есть ли доступ к HF Hub для {CONFIG['model']}.")
        return 1

    print("\nЗагрузка базы знаний…")
    solver = Solver(PROGRAM)
    print(f"  Фактов и правил: {len(solver.database.rules)}")

    # ---------- Этап 1: строгий ----------
    print("\n" + "=" * 78)
    print("ЭТАП 1 — СТРОГИЙ РЕЖИМ (semantic_off)")
    print("=" * 78)
    eu.semantic_off()
    strict_results = run_all(solver, verbose=True)

    # ---------- Этап 2: семантический ----------
    print("\n" + "=" * 78)
    print("ЭТАП 2 — СЕМАНТИЧЕСКИЙ РЕЖИМ (semantic_on)")
    print("=" * 78)
    eu.semantic_off()

    on_kwargs = _semantic_on_kwargs()
    eu.semantic_on(**on_kwargs)
    semantic_results = run_all(solver, verbose=True)

    # ---------- Сводная таблица ----------
    print("\n" + "=" * 78)
    print("СВОДНАЯ ТАБЛИЦА")
    print("=" * 78)
    fmt = "{:<3} {:<10} {:<48} {:<9} {:<12} {}"
    print(fmt.format("#", "Категория", "Запрос", "Строгий", "Семантика", "Δ"))
    print("-" * 100)

    wins = 0
    losses = 0
    for i, (num, category, desc, query) in enumerate(QUERIES):
        s_ok, s_cnt = strict_results[i]
        m_ok, m_cnt = semantic_results[i]
        s_str = f"OK ({s_cnt})" if s_ok else "FAIL"
        m_str = f"OK ({m_cnt})" if m_ok else "FAIL"

        if s_ok and not m_ok:
            delta = "!! регрессия"
            losses += 1
        elif not s_ok and m_ok:
            delta = "+ семантика победила"
            wins += 1
        else:
            delta = "="

        desc_short = (desc[:45] + "...") if len(desc) > 48 else desc
        print(fmt.format(num, category, desc_short, s_str, m_str, delta))

    print("-" * 100)
    strict_ok_total = sum(1 for ok, _ in strict_results if ok)
    semantic_ok_total = sum(1 for ok, _ in semantic_results if ok)
    print(f"Строгий режим:         {strict_ok_total} / {len(QUERIES)}")
    print(f"Семантический режим:   {semantic_ok_total} / {len(QUERIES)}")
    print(f"Побед семантики:       {wins}")
    print(f"Регрессий:             {losses}")

    # ---------- Статистика ----------
    u = eu.semantic_get_unifier()
    if u is not None:
        print("\n" + "=" * 78)
        print("СТАТИСТИКА СЕМАНТИЧЕСКОГО УНИФИКАТОРА")
        print("=" * 78)
        st = u.stats
        print(f"  Backend:                       {u.backend}")
        print(f"  Всего вызовов унификации:      {st['calls']}")
        print(f"  Успешно строго:                {st['strict_success']}")
        print(f"  Успешно exact (нормализация):  {st.get('exact_success', 0)}")
        print(f"  Успешно typo (опечатки/падежи):{st.get('typo_success', 0)}")
        print(f"  Успешно hard (cos >= порог):   {st['embedding_success']}")
        print(f"  Успешно soft (Levenshtein):    {st.get('soft_success', 0)}")
        print(f"  Провалов:                      {st['failures']}")
        cs = u.cache.stats()
        print(f"  Кэш эмбеддингов:               {cs['size']} записей")
        print(f"    попаданий:                   {cs['hits']}")
        print(f"    промахов:                    {cs['misses']}")
        print(f"    hit rate:                    {cs['hit_rate']:.1%}")
        print(f"  Префикс текста:                {u.text_prefix!r}")
        print(f"  lev_typo_max:                  {u.lev_typo_max}")

    # ---------- Таблица косинусов ----------
    print("\n" + "=" * 78)
    print("КОСИНУСНАЯ БЛИЗОСТЬ КЛЮЧЕВЫХ ПАР (что видит модель)")
    print("=" * 78)
    if eu.semantic_get_unifier() is None:
        eu.semantic_on(**on_kwargs)
    u = eu.semantic_get_unifier()

    pairs = [
        ("Москва", "москва", "регистр"),
        ("Москве", "москва", "падеж"),
        ("Москвы", "москва", "падеж"),
        ("Moscow", "москва", "кросс-язычность"),
        ("Beijing", "пекин", "кросс-язычность"),
        ("Масква", "москва", "опечатка"),
        ("РФ", "россия", "аббревиатура"),
        ("Париже", "париж", "падеж"),
        ("Токио", "токио", "регистр"),
        ("Китай", "китай", "регистр"),
        ("Япония", "япония", "регистр"),
        ("Германия", "германия", "регистр"),
        ("евро", "евро", "точное"),
        ("лондон", "москва", "ЛОЖНОЕ"),
        ("луна", "россия", "ЛОЖНОЕ"),
        ("кошка", "собака", "ЛОЖНОЕ"),
    ]

    header = f"{'Слово 1':<14} {'Слово 2':<14} {'cos':>8}   {'lev':>6}   {'слой':<7} тип"
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

    # ---------- Итог ----------
    print("\n" + "=" * 78)
    print("ИТОГ")
    print("=" * 78)
    if wins > 0 and losses == 0:
        print(f"✓ Семантический режим выиграл в {wins} запросах и не проиграл ни одного.")
    elif losses > 0:
        print(f"⚠ Обнаружены регрессии: {losses}. Возможно, порог слишком низкий.")
    else:
        print("Все запросы дали одинаковый результат — увеличьте сложность сценария.")

    eu.semantic_off()
    return 0


if __name__ == "__main__":
    sys.exit(main())