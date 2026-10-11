""" PyAgentProlog

Семантическая унификация на эмбеддингах.

Поддерживаются два backend'а:
    - 'ollama' — через langchain_ollama.OllamaEmbeddings;
    - 'st'     — через sentence_transformers.SentenceTransformer.

Слои сопоставления атомов:
    Layer 1. Нормализация текста (_normalize_text).
    Layer 2a. Почти идентичные строки (lev_ratio <= lev_typo_max) — match
              без обращения к модели. Ловит опечатки и морфологические
              вариации ("Масква" ↔ "москва", "Москве" ↔ "москва").
    Layer 2b. Levenshtein-коридор: cos >= soft_cos_min и lev_ratio <= lev_max.
    Layer 3. Hard: cos >= threshold.

Префиксы эмбеддера:
    E5 (intfloat, qllama) требует "query: ", nomic — "search_query: ".
    Префикс применяется внутри EmbeddingUnifier._embed ПОСЛЕ нормализации,
    поэтому кэш-ключи содержат префикс, а lev_ratio — нет.

Подавление вывода sentence-transformers / transformers / HF Hub:
    По умолчанию все служебные сообщения отключены (env-переменные ниже
    выставляются ДО импорта transformers, плюс программно глушатся
    прогресс-бары). Чтобы вернуть их, задайте PROLOG_ST_VERBOSE=1.

Кастомный embed_fn:
    semantic_on() и install_embedding_unifier() принимают опциональный
    embed_fn — готовую функцию текст→эмбеддинг. Если она передана, то
    model/backend используются только как метка и источник text_prefix,
    а построение модели не выполняется. Это нужно для HF-обёртки и для
    внешних backend'ов (STEmbeddingBackend, кастомные энкодеры).

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""

from __future__ import annotations

import difflib
import os
import threading
from typing import Callable, Dict, Optional, Any, List

import numpy as np

from prolog.terms import Term, Variable, Number


# =====================================================================
# Подавление вывода Hugging Face / transformers / sentence-transformers.
# ВАЖНО: env-переменные выставляются ДО первого импорта transformers.
# Управление: PROLOG_ST_VERBOSE=1 → вернуть прогресс-бары и предупреждения.
# =====================================================================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")

_ST_VERBOSE = _env_bool("PROLOG_ST_VERBOSE", False)

if not _ST_VERBOSE:
    # HF Hub: убирает "Fetching N files…" и полосы скачивания
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    # transformers: убирает "Loading weights: 100%|…|" и предупреждения
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # tokenizers: убирает "tokenizer_config.json: 100%|…|"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # tqdm: универсальное отключение полос (там, где поддерживается)
    os.environ.setdefault("TQDM_DISABLE", "1")


def _silence_transformers() -> None:
    """Программно глушит логирование и прогресс-бары transformers.

    Вызывается перед загрузкой ST-модели — на случай, если transformers
    уже был импортирован раньше и env-переменные не сработали.
    """
    try:
        from transformers.utils import logging as hf_logging
    except Exception:
        return
    try:
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    try:
        hf_logging.disable_progress_bar()
    except Exception:
        pass


# =====================================================================
# Известные префиксы для популярных эмбеддеров
# =====================================================================
KNOWN_MODEL_PREFIXES = [
    ("multilingual-e5", "query: "),
    ("intfloat-e5", "query: "),
    ("e5-large", "query: "),
    ("e5-base", "query: "),
    ("e5-small", "query: "),
    ("nomic-embed-text-v2", "search_query: "),
    ("nomic-embed-text-v1", "search_query: "),
    ("nomic-ai/nomic-embed", "search_query: "),
]


# =====================================================================
# Алиасы: короткое имя → полный HF ID
# =====================================================================
KNOWN_MODEL_ALIASES = {
    "multilingual-e5-large": "intfloat/multilingual-e5-large",
    "multilingual-e5-base": "intfloat/multilingual-e5-base",
    "multilingual-e5-small": "intfloat/multilingual-e5-small",
    "e5-large": "intfloat/multilingual-e5-large",
    "e5-base": "intfloat/multilingual-e5-base",
    "e5-small": "intfloat/multilingual-e5-small",
    "paraphrase-multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-minilm-l12-v2": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-multilingual-mpnet": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-multilingual-mpnet-base-v2": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "bge-m3": "BAAI/bge-m3",
    "nomic-embed-text-v1.5": "nomic-ai/nomic-embed-text-v1.5",
    "nomic-embed-text-v1": "nomic-ai/nomic-embed-text-v1",
}


def resolve_model_id(model: str) -> str:
    """Нормализует имя модели к полному HF ID."""
    if not model:
        return model
    if os.path.isabs(model):
        return model
    if "/" in model:
        return model
    return KNOWN_MODEL_ALIASES.get(model.lower(), model)


def detect_text_prefix(model_name: str) -> str:
    """Определяет префикс по имени модели."""
    if not model_name:
        return ""
    lower = model_name.lower()
    for key, prefix in KNOWN_MODEL_PREFIXES:
        if key in lower:
            return prefix
    return ""


# =====================================================================
# Layer 1 — Нормализация текста
# =====================================================================
def _normalize_text(s: str) -> str:
    """Каноническая форма строки: strip → снятие парных кавычек → lower()."""
    if not isinstance(s, str):
        s = str(s)
    s = s.strip()
    if len(s) >= 2 and (
        (s[0] == "'" and s[-1] == "'") or (s[0] == '"' and s[-1] == '"')
    ):
        s = s[1:-1]
    return s.lower()


# =====================================================================
# Layer 2 — Лексическое расстояние
# =====================================================================
def _lev_ratio(s1: str, s2: str) -> float:
    """Нормализованное расстояние [0..1]: 0 — идентичны, 1 — максимально разные."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    return 1.0 - difflib.SequenceMatcher(None, s1, s2).ratio()


# =====================================================================
# Потокобезопасный кэш эмбеддингов
# =====================================================================
class EmbeddingCache:
    """Кэш эмбеддингов. Ключ — нормализованная строка С префиксом."""

    __slots__ = ('_embed_fn', '_max_size', '_cache', '_lock', 'hits', 'misses')

    def __init__(self, embed_fn: Callable[[str], np.ndarray], max_size: int = 10000):
        self._embed_fn = embed_fn
        self._max_size = int(max_size)
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.RLock()
        self.hits = 0
        self.misses = 0

    def get(self, text: str) -> np.ndarray:
        key = _normalize_text(text)
        with self._lock:
            vec = self._cache.get(key)
            if vec is not None:
                self.hits += 1
                return vec
        vec = np.asarray(self._embed_fn(key), dtype=np.float32)
        with self._lock:
            self.misses += 1
            if len(self._cache) >= self._max_size:
                drop = max(1, self._max_size // 10)
                for k in list(self._cache.keys())[:drop]:
                    del self._cache[k]
            self._cache[key] = vec
        return vec

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                'size': len(self._cache),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': (self.hits / total) if total else 0.0,
            }


# =====================================================================
# Семантический унификатор
# =====================================================================
class EmbeddingUnifier:
    """Унификатор на эмбеддингах с поддержкой префиксов и трёх слоёв.

    Параметры:
        embed_fn         — функция текст→эмбеддинг (может быть передана
                           извне через install_embedding_unifier / semantic_on).
        text_prefix      — строка, добавляемая к тексту перед подачей в модель.
        threshold        — hard-порог cos (Layer 3).
        soft_cos_min     — нижняя граница cos для soft-коридора (Layer 2b).
        lev_max          — максимальный lev_ratio в soft-коридоре.
        lev_typo_max     — верхняя граница lev_ratio для «опечаток» (Layer 2a).
                           Обычно lev_typo_max < lev_max.
        custom_embed_fn  — True, если embed_fn был передан извне (а не построен
                           внутри install_embedding_unifier). Используется только
                           для диагностики через semantic_status() и для
                           reload-логики в semantic_on().
    """

    __slots__ = ('_embed_fn', '_threshold', '_cache', 'stats',
                 '_soft_cos_min', '_lev_max', '_lev_typo_max',
                 '_text_prefix', '_backend', '_custom_embed_fn')

    def __init__(self,
                 embed_fn: Callable[[str], np.ndarray],
                 threshold: float = 0.8,
                 max_cache: int = 10000,
                 soft_cos_min: float = 0.55,
                 lev_max: float = 0.35,
                 lev_typo_max: float = 0.20,
                 text_prefix: str = "",
                 backend: str = "unknown",
                 custom_embed_fn: bool = False):
        self._embed_fn = embed_fn
        self._threshold = float(threshold)
        self._soft_cos_min = float(soft_cos_min)
        self._lev_max = float(lev_max)
        self._lev_typo_max = float(lev_typo_max)
        self._text_prefix = text_prefix or ""
        self._backend = backend
        self._custom_embed_fn = bool(custom_embed_fn)
        self._cache = EmbeddingCache(embed_fn, max_size=max_cache)
        self.stats = {
            'calls': 0,
            'strict_success': 0,
            'exact_success': 0,
            'typo_success': 0,
            'embedding_success': 0,
            'soft_success': 0,
            'failures': 0,
        }

    # --------- свойства ---------
    @property
    def cache(self) -> EmbeddingCache:
        return self._cache

    @property
    def embed_fn(self) -> Callable[[str], np.ndarray]:
        """Текущая функция текст→эмбеддинг.

        Нужна для identity-сравнения в semantic_on: если пользователь
        передаёт другой embed_fn, унификатор автоматически пересоздаётся.
        """
        return self._embed_fn

    @property
    def has_custom_embed_fn(self) -> bool:
        """True, если embed_fn был передан извне (а не построен внутри)."""
        return self._custom_embed_fn

    @property
    def threshold(self) -> float:
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        self._threshold = float(value)

    @property
    def soft_cos_min(self) -> float:
        return self._soft_cos_min

    @soft_cos_min.setter
    def soft_cos_min(self, value: float) -> None:
        self._soft_cos_min = float(value)

    @property
    def lev_max(self) -> float:
        return self._lev_max

    @lev_max.setter
    def lev_max(self, value: float) -> None:
        self._lev_max = float(value)

    @property
    def lev_typo_max(self) -> float:
        return self._lev_typo_max

    @lev_typo_max.setter
    def lev_typo_max(self, value: float) -> None:
        self._lev_typo_max = float(value)

    @property
    def text_prefix(self) -> str:
        return self._text_prefix

    @text_prefix.setter
    def text_prefix(self, value: str) -> None:
        self._text_prefix = value or ""

    @property
    def backend(self) -> str:
        return self._backend

    # --------- утилиты ---------
    @staticmethod
    def term_to_text(term) -> str:
        return _normalize_text(str(term))

    def _embed(self, text: str) -> np.ndarray:
        full = self._text_prefix + text
        return self._cache.get(full)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    @staticmethod
    def _is_atom_like(term) -> bool:
        if isinstance(term, Variable) or isinstance(term, Number):
            return False
        return isinstance(term, Term) and not term.arguments

    # --------- ядро сопоставления ---------
    def _semantic_match(self, t1, t2) -> Optional[str]:
        """Сопоставление двух атомов.

        Возвращает 'exact' | 'typo' | 'hard' | 'soft' | None.

        Последовательность:
            1. exact — строки совпали после нормализации (без модели).
            2. typo  — lev_ratio <= lev_typo_max (без модели, ловит опечатки).
            3. hard  — cos >= threshold.
            4. soft  — soft_cos_min <= cos < threshold и lev_ratio <= lev_max.
            5. None  — отказ.

        Модель вызывается только для пар с lev_ratio > lev_typo_max.
        """
        s1 = self.term_to_text(t1)
        s2 = self.term_to_text(t2)

        # Layer 1: точное совпадение после нормализации
        if s1 == s2:
            return 'exact'

        # Layer 2a: почти идентичны по написанию — match без модели
        lev = _lev_ratio(s1, s2)
        if lev <= self._lev_typo_max:
            return 'typo'

        # Семантическая проверка (модель вызывается только тут)
        try:
            v1 = self._embed(s1)
            v2 = self._embed(s2)
            cos = self._cosine(v1, v2)
        except Exception:
            return None

        # Layer 3: hard — уверенное семантическое совпадение
        if cos >= self._threshold:
            return 'hard'

        # Layer 2b: soft — коридор cos + лексическое сходство
        if cos >= self._soft_cos_min and lev <= self._lev_max:
            return 'soft'

        return None

    @staticmethod
    def _resolve(t, subst):
        if isinstance(t, Variable) and t in subst:
            return subst[t].substitute(subst)
        return t

    def unify(self, t1, t2, subst: Optional[Dict] = None, depth: int = 0,
              occurs_check: bool = False):
        if subst is None:
            subst = {}
        self.stats['calls'] += 1

        r1 = self._resolve(t1, subst)
        r2 = self._resolve(t2, subst)

        if isinstance(r1, Variable) or isinstance(r2, Variable):
            r = t1.unify_strict(t2, subst, depth, occurs_check)
            self.stats['strict_success' if r is not None else 'failures'] += 1
            return r

        r = t1.unify_strict(t2, subst, depth, occurs_check)
        if r is not None:
            self.stats['strict_success'] += 1
            return r

        if (isinstance(r1, Term) and isinstance(r2, Term)
                and not isinstance(r1, (Variable, Number))
                and not isinstance(r2, (Variable, Number))
                and r1.functor == r2.functor
                and len(r1.arguments) == len(r2.arguments)
                and r1.arguments):
            current = subst.copy()
            for a1, a2 in zip(r1.arguments, r2.arguments):
                res = self.unify(a1, a2, current, depth + 1, occurs_check)
                if res is None:
                    break
                current = res
            else:
                self.stats['embedding_success'] += 1
                return current

        if self._is_atom_like(r1) and self._is_atom_like(r2):
            match = self._semantic_match(r1, r2)
            if match == 'exact':
                self.stats['exact_success'] += 1
                return subst.copy()
            if match == 'typo':
                self.stats['typo_success'] += 1
                return subst.copy()
            if match == 'hard':
                self.stats['embedding_success'] += 1
                return subst.copy()
            if match == 'soft':
                self.stats['soft_success'] += 1
                return subst.copy()

        self.stats['failures'] += 1
        return None


# =====================================================================
# Backend 1: Ollama
# =====================================================================
def make_ollama_embed_fn(model: str = "paraphrase-multilingual:latest",
                         base_url: str = "http://localhost:11434"
                         ) -> Callable[[str], np.ndarray]:
    """Возвращает функцию текст→эмбеддинг на базе OllamaEmbeddings."""
    try:
        from langchain_ollama import OllamaEmbeddings
    except ImportError:
        from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

    emb = OllamaEmbeddings(model=model, base_url=base_url)

    def embed_fn(text: str) -> np.ndarray:
        return np.asarray(emb.embed_query(text), dtype=np.float32)

    return embed_fn


# =====================================================================
# Backend 2: sentence-transformers
# =====================================================================
def _resolve_st_model_path(model: str, local_dir: Optional[str]) -> str:
    """Разрешает путь к модели.

    Порядок:
      1. resolve_model_id(model) — короткое имя → полный HF ID.
      2. Абсолютный путь, если существует → как есть.
      3. local_dir/<short_name>, если существует → как есть.
      4. local_dir/models--<org>--<name>, если существует → как есть.
      5. Иначе → HF ID (ST попробует скачать в cache_folder).
    """
    if not model:
        raise ValueError("model name must be non-empty")

    model = resolve_model_id(model)

    if os.path.isabs(model) and os.path.isdir(model):
        return model

    if local_dir:
        short = model.split("/")[-1]
        candidate = os.path.join(local_dir, short)
        if os.path.isdir(candidate):
            return candidate
        hf_style = "models--" + model.replace("/", "--")
        candidate2 = os.path.join(local_dir, hf_style)
        if os.path.isdir(candidate2):
            return candidate2

    return model


def make_sentence_transformer_embed_fn(model: str,
                                        local_dir: Optional[str] = None,
                                        cache_folder: Optional[str] = None,
                                        device: Optional[str] = None,
                                        local_files_only: bool = False,
                                        verbose: Optional[bool] = None,
                                        ) -> Callable[[str], np.ndarray]:
    """Возвращает функцию текст→эмбеддинг на базе sentence_transformers.

    Префикс сюда НЕ входит — его добавляет EmbeddingUnifier._embed.

    verbose:
        None  → берётся из env PROLOG_ST_VERBOSE (по умолчанию False);
        True  → печатать путь к модели и прогресс-бары;
        False → полностью тихо.

    Требуется: pip install sentence-transformers
    """
    if verbose is None:
        verbose = _ST_VERBOSE

    if not verbose:
        _silence_transformers()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Для backend='st' требуется sentence-transformers. "
            "Установите: pip install sentence-transformers"
        ) from e

    resolved = _resolve_st_model_path(model, local_dir)
    if verbose:
        print(f"[ST] resolved model path: {resolved!r}")

    kwargs = {}
    if device:
        kwargs["device"] = device
    if not os.path.isdir(resolved):
        if cache_folder:
            kwargs["cache_folder"] = cache_folder
        elif local_dir:
            kwargs["cache_folder"] = local_dir
        if local_files_only:
            kwargs["local_files_only"] = True

    try:
        st_model = SentenceTransformer(resolved, **kwargs)
    except TypeError:
        kwargs.pop("local_files_only", None)
        kwargs.pop("cache_folder", None)
        st_model = SentenceTransformer(resolved, **kwargs)

    def embed_fn(text: str) -> np.ndarray:
        v = st_model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(v, dtype=np.float32)

    return embed_fn


# =====================================================================
# Установка/снятие глобального унификатора
# =====================================================================
def install_embedding_unifier(embed_fn: Optional[Callable[[str], np.ndarray]] = None,
                              threshold: float = 0.8,
                              model: str = "paraphrase-multilingual:latest",
                              base_url: str = "http://localhost:11434",
                              max_cache: int = 10000,
                              soft_cos_min: float = 0.55,
                              lev_max: float = 0.35,
                              lev_typo_max: float = 0.20,
                              text_prefix: Optional[str] = None,
                              backend: str = "ollama",
                              local_dir: Optional[str] = None,
                              st_device: Optional[str] = None,
                              st_cache_folder: Optional[str] = None,
                              st_local_files_only: bool = False,
                              st_verbose: Optional[bool] = None,
                              ) -> EmbeddingUnifier:
    """Устанавливает глобальный семантический унификатор.

    backend:
        'ollama' — через langchain_ollama.OllamaEmbeddings.
        'st'     — через sentence_transformers.SentenceTransformer.
        'custom' — ничего не строим, используем переданный embed_fn.
                   Если embed_fn=None, поведение совпадает с 'ollama'.

    embed_fn:
        Готовая функция текст→эмбеддинг. Если передана — ветка построения
        модели пропускается; model/backend служат только меткой и
        источником text_prefix.

    text_prefix=None → автоопределение по имени модели (detect_text_prefix).
    st_verbose=None  → берётся из env PROLOG_ST_VERBOSE (по умолчанию False).
    """
    from prolog import terms as _terms

    custom_embed_fn = embed_fn is not None

    if text_prefix is None:
        text_prefix = detect_text_prefix(resolve_model_id(model))

    if embed_fn is None:
        if backend == "ollama":
            embed_fn = make_ollama_embed_fn(model=model, base_url=base_url)
        elif backend == "st":
            embed_fn = make_sentence_transformer_embed_fn(
                model=model,
                local_dir=local_dir,
                cache_folder=st_cache_folder,
                device=st_device,
                local_files_only=st_local_files_only,
                verbose=st_verbose,
            )
        elif backend == "custom":
            # Пользователь попросил 'custom', но не передал embed_fn —
            # это ошибка конфигурации, падаем явно.
            raise ValueError(
                "backend='custom' требует аргумент embed_fn. "
                "Передайте готовую функцию текст→эмбеддинг."
            )
        else:
            raise ValueError(
                f"Неизвестный backend={backend!r}. "
                f"Допустимые: 'ollama', 'st', 'custom'."
            )

    unifier = EmbeddingUnifier(
        embed_fn,
        threshold=threshold,
        max_cache=max_cache,
        soft_cos_min=soft_cos_min,
        lev_max=lev_max,
        lev_typo_max=lev_typo_max,
        text_prefix=text_prefix,
        backend=backend,
        custom_embed_fn=custom_embed_fn,
    )
    _terms.set_custom_unifier(unifier)
    return unifier


def uninstall_embedding_unifier() -> None:
    """Возвращает строгую унификацию по умолчанию."""
    from prolog import terms as _terms
    _terms.clear_custom_unifier()


# =====================================================================
# API для управления из Prolog
# =====================================================================
_state_lock = threading.RLock()
_current_unifier: Optional[EmbeddingUnifier] = None

DEFAULT_MODEL = "paraphrase-multilingual:latest"
DEFAULT_ST_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_THRESHOLD = 0.82
DEFAULT_SOFT_COS_MIN = 0.55
DEFAULT_LEV_MAX = 0.35
DEFAULT_LEV_TYPO_MAX = 0.20
DEFAULT_MAX_CACHE = 10000


def semantic_on(threshold: Optional[float] = None,
                model: Optional[str] = None,
                base_url: Optional[str] = None,
                max_cache: Optional[int] = None,
                soft_cos_min: Optional[float] = None,
                lev_max: Optional[float] = None,
                lev_typo_max: Optional[float] = None,
                text_prefix: Optional[str] = None,
                backend: Optional[str] = None,
                local_dir: Optional[str] = None,
                st_device: Optional[str] = None,
                st_cache_folder: Optional[str] = None,
                st_local_files_only: bool = False,
                st_verbose: Optional[bool] = None,
                embed_fn: Optional[Callable[[str], np.ndarray]] = None,
                force_reload: bool = False) -> EmbeddingUnifier:
    """Включить семантическую унификацию.

    Параметры:
        embed_fn:
            None            → построить embed_fn внутри по model/backend.
                              Поведение полностью совпадает с прежним.
            Callable        → использовать готовую функцию текст→эмбеддинг.
                              model/backend в этом случае служат только
                              меткой и источником text_prefix. Если identity
                              embed_fn отличается от текущего унификатора,
                              унификатор автоматически пересоздаётся —
                              force_reload не требуется.

        text_prefix:
            None → автоопределение; "" → без префикса; "query: " → явный.

        backend:
            None → 'custom' при embed_fn, иначе 'ollama'.
            'ollama' | 'st' | 'custom'.

        st_verbose:
            None → из env PROLOG_ST_VERBOSE (по умолчанию тихо).

        force_reload:
            True → гарантированно пересоздать унификатор.
    """
    global _current_unifier
    with _state_lock:
        # ---- Авто-detect смены embed_fn ----
        # Если пользователь передал кастомную функцию и она отличается
        # по identity от той, что стоит в текущем унификаторе,
        # принудительно пересоздаём — иначе новые эмбеддинги просто
        # не подхватятся, потому что унификатор хранит старый _embed_fn.
        if embed_fn is not None and _current_unifier is not None:
            if _current_unifier.embed_fn is not embed_fn:
                force_reload = True

        need_new = (_current_unifier is None) or force_reload

        if not need_new:
            if backend is not None and backend != _current_unifier.backend:
                need_new = True
            if not need_new:
                # ---- in-place update параметров ----
                if threshold is not None:
                    _current_unifier.threshold = float(threshold)
                if soft_cos_min is not None:
                    _current_unifier.soft_cos_min = float(soft_cos_min)
                if lev_max is not None:
                    _current_unifier.lev_max = float(lev_max)
                if lev_typo_max is not None:
                    _current_unifier.lev_typo_max = float(lev_typo_max)
                if text_prefix is not None:
                    _current_unifier.text_prefix = text_prefix
                return _current_unifier

        # ---- Значения по умолчанию ----
        if backend is None:
            # С кастомным embed_fn метка 'custom' информативнее, чем 'ollama'.
            # На построение это не влияет: install_embedding_unifier
            # пропускает свою ветку, когда embed_fn уже задан.
            backend = "custom" if embed_fn is not None else "ollama"
        if model is None:
            model = DEFAULT_ST_MODEL if backend == "st" else DEFAULT_MODEL
        if base_url is None:
            base_url = DEFAULT_BASE_URL
        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        if soft_cos_min is None:
            soft_cos_min = DEFAULT_SOFT_COS_MIN
        if lev_max is None:
            lev_max = DEFAULT_LEV_MAX
        if lev_typo_max is None:
            lev_typo_max = DEFAULT_LEV_TYPO_MAX
        if max_cache is None:
            max_cache = DEFAULT_MAX_CACHE

        _current_unifier = install_embedding_unifier(
            embed_fn=embed_fn,
            model=model, base_url=base_url,
            threshold=threshold, max_cache=max_cache,
            soft_cos_min=soft_cos_min, lev_max=lev_max,
            lev_typo_max=lev_typo_max,
            text_prefix=text_prefix,
            backend=backend,
            local_dir=local_dir,
            st_device=st_device,
            st_cache_folder=st_cache_folder,
            st_local_files_only=st_local_files_only,
            st_verbose=st_verbose,
        )
        return _current_unifier


def semantic_off() -> None:
    global _current_unifier
    with _state_lock:
        uninstall_embedding_unifier()
        _current_unifier = None


def semantic_get_unifier() -> Optional[EmbeddingUnifier]:
    return _current_unifier


def semantic_set_threshold(value: float) -> bool:
    with _state_lock:
        if _current_unifier is None:
            return False
        _current_unifier.threshold = float(value)
        return True


def semantic_set_corridor(soft_cos_min: Optional[float] = None,
                          lev_max: Optional[float] = None,
                          lev_typo_max: Optional[float] = None) -> bool:
    """Изменить параметры коридоров. Возвращает False, если унификатор не установлен."""
    with _state_lock:
        if _current_unifier is None:
            return False
        if soft_cos_min is not None:
            _current_unifier.soft_cos_min = float(soft_cos_min)
        if lev_max is not None:
            _current_unifier.lev_max = float(lev_max)
        if lev_typo_max is not None:
            _current_unifier.lev_typo_max = float(lev_typo_max)
        return True


def semantic_set_prefix(text_prefix: str) -> bool:
    with _state_lock:
        if _current_unifier is None:
            return False
        if _current_unifier.text_prefix != text_prefix:
            _current_unifier.text_prefix = text_prefix
            _current_unifier.cache.clear()
        return True


def semantic_status() -> Dict[str, Any]:
    with _state_lock:
        if _current_unifier is None:
            return {'enabled': False}
        return {
            'enabled': True,
            'backend': _current_unifier.backend,
            'has_custom_embed_fn': _current_unifier.has_custom_embed_fn,
            'threshold': _current_unifier.threshold,
            'soft_cos_min': _current_unifier.soft_cos_min,
            'lev_max': _current_unifier.lev_max,
            'lev_typo_max': _current_unifier.lev_typo_max,
            'text_prefix': _current_unifier.text_prefix,
            'cache': _current_unifier.cache.stats(),
            'stats': dict(_current_unifier.stats),
        }