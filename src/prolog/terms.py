""" PyAgentProlog

Базовые классы для термов с оптимизированной унификацией (occurs_check отключён по умолчанию)
и уменьшенным потреблением памяти (__slots__).

Поддерживается подключаемый унификатор (например, семантический на эмбеддингах):
публичный Term.unify(...) — точка входа-диспетчер, а строгая унификация
вынесена в Term.unify_strict(...) и не уходит повторно в кастомный унификатор.

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""

import sys
import threading

DEBUG_UNIFY = False   # Отладку можно включить при необходимости


def debug(*args, indent=0, **kwargs):
    if DEBUG_UNIFY:
        print("  " * indent + "[UNIFY]", *args, **kwargs)


# =====================================================================
# Подключаемый унификатор
# =====================================================================
_custom_unifier = None
_thread_local = threading.local()


def set_custom_unifier(unifier) -> None:
    """Установить пользовательский унификатор (например, EmbeddingUnifier)."""
    global _custom_unifier
    _custom_unifier = unifier


def get_custom_unifier():
    """Получить текущий пользовательский унификатор (или None)."""
    return _custom_unifier


def clear_custom_unifier() -> None:
    """Вернуть строгую унификацию по умолчанию."""
    global _custom_unifier
    _custom_unifier = None
# =====================================================================


class Term:
    """Базовый класс для термов Prolog"""
    __slots__ = ('functor', 'arguments')

    def __init__(self, functor, arguments=None):
        self.functor = functor
        self.arguments = arguments or []

    # -----------------------------------------------------------------
    # Публичная точка входа — диспетчер
    # -----------------------------------------------------------------
    def unify(self, other, subst=None, depth=0, occurs_check=False):
        """
        Унификация термов.

        Если установлен кастомный унификатор (например, семантический
        на эмбеддингах) и мы не находимся внутри него — вызываем его.
        В противном случае выполняется строгая унификация unify_strict.

        Параметр occurs_check управляет проверкой вхождения (по умолчанию выключена).
        """
        if subst is None:
            subst = {}
        custom = _custom_unifier
        if custom is not None and not getattr(_thread_local, 'inside', False):
            _thread_local.inside = True
            try:
                return custom.unify(self, other, subst, depth, occurs_check)
            finally:
                _thread_local.inside = False
        return self.unify_strict(other, subst, depth, occurs_check)

    # -----------------------------------------------------------------
    # Строгая унификация (исходный алгоритм без изменений)
    # -----------------------------------------------------------------
    def unify_strict(self, other, subst=None, depth=0, occurs_check=False):
        """
        Строгая унификация термов (без обращения к кастомному унификатору).
        Параметр occurs_check управляет проверкой вхождения (по умолчанию выключена).
        """
        if subst is None:
            subst = {}

        debug(f"unify({self}, {other})", indent=depth)
        debug(f"  subst = { {str(k):str(v) for k,v in subst.items()} }", indent=depth)

        # Если это одна и та же переменная
        if self is other:
            debug(f"  одна и та же ссылка -> {subst}", indent=depth)
            return subst

        # Если self - переменная
        if isinstance(self, Variable):
            debug(f"  self - переменная {self.name}", indent=depth)
            res = self.unify_with(other, subst, depth + 1, occurs_check)
            debug(f"  результат унификации переменной: {res}", indent=depth)
            return res

        # Если other - переменная
        if isinstance(other, Variable):
            debug(f"  other - переменная {other.name}", indent=depth)
            res = other.unify_with(self, subst, depth + 1, occurs_check)
            debug(f"  результат унификации переменной: {res}", indent=depth)
            return res

        # Если self - число
        if isinstance(self, Number):
            debug(f"  self - число {self.value}", indent=depth)
            if isinstance(other, Number):
                if self.value == other.value:
                    debug(f"    числа равны -> {subst}", indent=depth)
                    return subst
            elif isinstance(other, Term):
                try:
                    val = float(other.functor) if '.' in other.functor else int(other.functor)
                    if self.value == val:
                        debug(f"    терм {other} приводится к числу {val} -> {subst}", indent=depth)
                        return subst
                except:
                    pass
            debug(f"    числа/термы не равны -> None", indent=depth)
            return None

        # Если other - число
        if isinstance(other, Number):
            debug(f"  other - число {other.value}", indent=depth)
            return other.unify_strict(self, subst, depth + 1, occurs_check)

        # Если оба - обычные термы
        if isinstance(self, Term) and isinstance(other, Term):
            # Специальная обработка списков
            if self.functor == '.' and other.functor == '.':
                debug(f"  унификация списков: {self} vs {other}", indent=depth)
                head1, tail1 = self.arguments[0], self.arguments[1]
                head2, tail2 = other.arguments[0], other.arguments[1]
                debug(f"    голова1={head1}, голова2={head2}", indent=depth + 1)
                debug(f"    хвост1={tail1}, хвост2={tail2}", indent=depth + 1)
                subst1 = head1.unify_strict(head2, subst, depth + 1, occurs_check)
                if subst1 is None:
                    debug(f"    унификация голов не удалась -> None", indent=depth)
                    return None
                debug(f"    после унификации голов: {subst1}", indent=depth + 1)
                subst2 = tail1.unify_strict(tail2, subst1, depth + 1, occurs_check)
                if subst2 is None:
                    debug(f"    унификация хвостов не удалась -> None", indent=depth)
                    return None
                debug(f"    унификация списков успешна -> {subst2}", indent=depth)
                return subst2

            # Обычные термы
            debug(f"  оба - обычные термы: {self.functor}/{len(self.arguments)} vs {other.functor}/{len(other.arguments)}",
                  indent=depth)
            if self.functor != other.functor or len(self.arguments) != len(other.arguments):
                debug(f"    функторы или арность не совпадают -> None", indent=depth)
                return None

            current_subst = subst.copy()
            for i, (arg1, arg2) in enumerate(zip(self.arguments, other.arguments)):
                debug(f"    унификация аргумента {i}: {arg1} vs {arg2}", indent=depth + 1)
                result = arg1.unify_strict(arg2, current_subst, depth + 1, occurs_check)
                if result is None:
                    debug(f"    аргумент {i} не унифицирован -> None", indent=depth + 1)
                    return None
                current_subst = result
                debug(f"    подстановка после аргумента {i}: {current_subst}", indent=depth + 1)

            debug(f"  унификация термов успешна -> {current_subst}", indent=depth)
            return current_subst

        debug(f"  не удалось унифицировать -> None", indent=depth)
        return None

    def substitute(self, subst):
        """Подстановка значений переменных (без защиты от циклов)"""
        if isinstance(subst, dict):
            if isinstance(self, Variable):
                return self.substitute_from_dict(subst)
            elif isinstance(self, Number):
                return self
            elif isinstance(self, Term):
                new_args = [arg.substitute(subst) for arg in self.arguments]
                return Term(self.functor, new_args)
        return self

    def substitute_from_dict(self, subst, visited=None):
        """Подстановка из словаря с защитой от циклических ссылок"""
        if visited is None:
            visited = set()
        if isinstance(self, Variable):
            if self in subst:
                if self in visited:
                    # Циклическая ссылка – возвращаем переменную как есть
                    return self
                visited.add(self)
                value = subst[self]
                return value.substitute_from_dict(subst, visited)
            return self
        elif isinstance(self, Number):
            return self
        elif isinstance(self, Term):
            new_args = [arg.substitute_from_dict(subst, visited) for arg in self.arguments]
            return Term(self.functor, new_args)
        return self

    def is_list(self):
        """Проверяет, является ли терм списком"""
        if self.functor == '[]' and len(self.arguments) == 0:
            return True
        elif self.functor == '.' and len(self.arguments) == 2:
            return self.arguments[1].is_list()
        return False

    def get_list_elements(self):
        """Возвращает элементы списка, если терм является списком"""
        elements = []
        current = self
        while isinstance(current, Term) and current.functor == '.' and len(current.arguments) == 2:
            elements.append(current.arguments[0])
            current = current.arguments[1]
        if isinstance(current, Term) and current.functor == '[]' and len(current.arguments) == 0:
            return elements
        else:
            return None

    def __str__(self):
        # Обработка конъюнкции
        if self.functor == ',' and len(self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            return f"{left}, {right}"
        # Обработка дизъюнкции
        if self.functor == ';' and len(self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            return f"{left} ; {right}"
        # Обработка отрицания
        if self.functor == '\\+' and len(self.arguments) == 1:
            return f"\\+({self.arguments[0]})"
        # Обработка арифметических операторов
        if self.functor in ['+', '-', '*', '/', 'is', '=', '=:=', r'=\=', '<', '>', '=<', '>='] and len(
                self.arguments) == 2:
            left = str(self.arguments[0])
            right = str(self.arguments[1])
            precedence = {
                'is': 1,
                '*': 2, '/': 2,
                '+': 3, '-': 3,
                '=:=': 4, r'=\=': 4, '<': 4, '>': 4, '=<': 4, '>=': 4,
                '=': 5
            }
            if isinstance(self.arguments[0], Term) and self.arguments[0].functor in precedence:
                if precedence.get(self.arguments[0].functor, 10) > precedence.get(self.functor, 10):
                    left = f"({left})"
            if isinstance(self.arguments[1], Term) and self.arguments[1].functor in precedence:
                if precedence.get(self.arguments[1].functor, 10) > precedence.get(self.functor, 10):
                    right = f"({right})"
                elif self.functor in ['-', '/'] and precedence.get(self.arguments[1].functor, 10) >= precedence.get(
                        self.functor, 10):
                    right = f"({right})"
            return f"{left} {self.functor} {right}"
        # Обработка списков
        if self.functor == '.' and len(self.arguments) == 2:
            elements = []
            current = self
            while isinstance(current, Term) and current.functor == '.' and len(current.arguments) == 2:
                elements.append(str(current.arguments[0]))
                current = current.arguments[1]
            if isinstance(current, Term) and current.functor == '[]' and len(current.arguments) == 0:
                return '[' + ','.join(elements) + ']'
            else:
                return '[' + ','.join(elements) + '|' + str(current) + ']'
        # Обработка строк в кавычках
        if len(self.arguments) == 0:
            functor_str = str(self.functor)
            if (functor_str.startswith("'") and functor_str.endswith("'")) or \
                    (functor_str.startswith('"') and functor_str.endswith('"')):
                return functor_str
            return functor_str
        else:
            args_str = ', '.join(str(arg) for arg in self.arguments)
            return f"{self.functor}({args_str})"

    def __repr__(self):
        return str(self)


class Variable(Term):
    """Переменная Prolog с оптимизированной унификацией (occurs_check по умолчанию выключен)"""
    __slots__ = ('name',)

    def __init__(self, name):
        super().__init__(name, [])   # <-- вызываем конструктор Term для установки атрибутов
        self.name = name

    def unify_with(self, other, subst, depth=0, occurs_check=False):
        """
        Унификация переменной с другим термом.
        Параметр occurs_check включает проверку вхождения (по умолчанию False).

        Этот метод — внутренний для строгой унификации: он вызывается только
        из Term.unify_strict и не уходит в кастомный унификатор.
        """
        debug(f"unify_with: {self.name} with {other}", indent=depth)

        # Анонимная переменная всегда унифицируется
        if self.name == "_":
            debug(f"  анонимная переменная -> {subst}", indent=depth)
            return subst

        # Если переменная уже имеет значение в подстановке
        if self in subst:
            debug(f"  переменная {self.name} уже связана со значением {subst[self]}", indent=depth)
            res = subst[self].unify_strict(other, subst, depth + 1, occurs_check)
            debug(f"  результат рекурсивной унификации: {res}", indent=depth)
            return res

        # Если other - переменная, которая уже имеет значение
        if isinstance(other, Variable) and other in subst:
            debug(f"  other - переменная {other.name}, уже связана со значением {subst[other]}", indent=depth)
            res = self.unify_with(subst[other], subst, depth + 1, occurs_check)
            debug(f"  результат рекурсивной унификации: {res}", indent=depth)
            return res

        # Если other - переменная, причём та же самая (по объекту или по имени)
        if isinstance(other, Variable):
            if self is other or self.name == other.name:
                debug(f"  унификация переменной {self.name} с самой собой -> {subst}", indent=depth)
                return subst

        # Occurs-check – по умолчанию отключён для производительности
        if occurs_check and self.occurs_in(other, subst):
            debug(f"  occurs_check не пройден: {self.name} входит в {other}", indent=depth)
            return None

        # Создаём новую подстановку
        new_subst = subst.copy()
        new_subst[self] = other
        debug(f"  создана новая подстановка: { {str(k):str(v) for k,v in new_subst.items()} }", indent=depth)
        return new_subst

    def occurs_in(self, term, subst):
        """Проверка вхождения переменной в терм (используется только при occurs_check=True)"""
        if self is term:
            return False
        if self == term:
            return False
        if isinstance(term, Variable) and term in subst:
            return self.occurs_in(subst[term], subst)
        if isinstance(term, Term):
            for arg in term.arguments:
                if self.occurs_in(arg, subst):
                    return True
        return False

    def __eq__(self, other):
        if isinstance(other, Variable):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name


class Number(Term):
    """Число в Prolog"""
    __slots__ = ('value',)

    def __init__(self, value):
        super().__init__(str(value), [])   # <-- вызываем конструктор Term
        self.value = value

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, Number):
            return self.value == other.value
        return self.value == other

    def __lt__(self, other):
        if isinstance(other, Number):
            return self.value < other.value
        return self.value < other

    def __le__(self, other):
        if isinstance(other, Number):
            return self.value <= other.value
        return self.value <= other

    def __gt__(self, other):
        if isinstance(other, Number):
            return self.value > other.value
        return self.value > other

    def __ge__(self, other):
        if isinstance(other, Number):
            return self.value >= other.value
        return self.value >= other

    def __hash__(self):
        return hash(self.value)


class Rule:
    """Правило Prolog"""
    __slots__ = ('head', 'body')

    def __init__(self, head, body):
        self.head = head
        self.body = body

    def __str__(self):
        if isinstance(self.body, Term) and self.body.functor == "true":
            return f"{self.head}."
        else:
            return f"{self.head} :- {self.body}."

    def __repr__(self):
        return str(self)