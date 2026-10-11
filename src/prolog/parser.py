""" PyAgentProlog

Парсер для полного Prolog

Поддерживает распознавание директив семантической унификации:
    :- semantic_on.
    :- semantic_on(0.85).
    :- semantic_off.
    :- semantic_threshold(0.75).

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""

import re

# Регулярные выражения для токенов
TOKEN_REGEX = r'\"[^\"]*\"|\'[^\']*\'|[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?|[a-zа-яё][a-zA-ZА-Яа-яЁё0-9_]*|[A-ZА-ЯЁ_][A-Za-zА-Яа-яЁё0-9_]*|:\-|->|\\\+|[()\.,;]|\[|\]|\||=:=|=\\=|=<|>=|==|\\==|\\=|@>=|@>|@=<|@<|=\.\.|=|\+\+|\+|\-|\*\*|//|>>|<<|\*|/|mod|div|!|is|[><]=?|\^'
ATOM_NAME_REGEX = r"^[a-zа-яё][A-Za-zА-Яа-яЁё0-9_]*$"
VARIABLE_REGEX = r"^[A-ZА-ЯЁ_][A-Za-zА-Яа-яЁё0-9_]*$"
NUMBER_REGEX = r"^[+-]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?$"

# Директивы, распознаваемые как «команды» и передаваемые в Solver/Database
SEMANTIC_DIRECTIVE_NAMES = {
    'semantic_on',
    'semantic_off',
    'semantic_threshold',
}


def remove_comments(input_text):
    """
    Удаление комментариев из текста с поддержкой вложенных многострочных
    комментариев /* ... /* ... */ ... */.
    """
    result = []
    i = 0
    n = len(input_text)
    in_single_quote = False
    in_double_quote = False
    comment_level = 0

    while i < n:
        ch = input_text[i]

        if comment_level > 0:
            if ch == '/' and i + 1 < n and input_text[i + 1] == '*':
                comment_level += 1
                i += 2
                continue
            if ch == '*' and i + 1 < n and input_text[i + 1] == '/':
                comment_level -= 1
                i += 2
                continue
            i += 1
            continue

        if ch == '%' and not (in_single_quote or in_double_quote):
            i += 1
            while i < n and input_text[i] != '\n':
                i += 1
            if i < n and input_text[i] == '\n':
                result.append('\n')
                i += 1
            continue

        if ch == '/' and i + 1 < n and input_text[i + 1] == '*' and not (in_single_quote or in_double_quote):
            comment_level = 1
            i += 2
            continue

        if ch == '"' and not in_single_quote:
            result.append(ch)
            if not in_double_quote:
                in_double_quote = True
            else:
                if i + 1 < n and input_text[i + 1] == '"':
                    result.append('"')
                    i += 2
                    continue
                else:
                    in_double_quote = False
            i += 1
            continue

        if ch == "'" and not in_double_quote:
            result.append(ch)
            if not in_single_quote:
                in_single_quote = True
            else:
                if i + 1 < n and input_text[i + 1] == "'":
                    result.append("'")
                    i += 2
                    continue
                else:
                    in_single_quote = False
            i += 1
            continue

        result.append(ch)
        i += 1

    return ''.join(result)


def parse_tokens_with_positions(input_text):
    """
    Возвращает (tokens, positions), где tokens - список строк-токенов,
    positions - список кортежей (start, end) для каждого токена в очищенном тексте.
    """
    clean_text = remove_comments(input_text)
    tokens = []
    positions = []
    for match in re.finditer(TOKEN_REGEX, clean_text):
        tokens.append(match.group())
        positions.append((match.start(), match.end()))
    return tokens, positions


def parse_query_from_string(input_text):
    """Вспомогательная функция для парсинга запроса из строки"""
    parser = Parser(input_text)
    return parser.parse_query()


class Parser(object):
    """
    ПРИМЕЧАНИЕ: Экземпляр можно использовать только один раз!
    """

    OPERATOR_PRECEDENCE = {
        ':-': 1200,
        ';': 900,
        '->': 950,
        ',': 1000,
        '=..': 1100,
        'is': 1100,
        '=': 1100, '\\=': 1100, '==': 1100, '\\==': 1100,
        '=:=': 1100, '=\\=': 1100, '<': 1100, '>': 1100, '=<': 1100, '>=': 1100,
        '@<': 1100, '@>': 1100, '@=<': 1100, '@>=': 1100,
        '+': 1200, '-': 1200,
        '*': 1300, '/': 1300, 'mod': 1300, 'div': 1300,
        '**': 1300, '//': 1200, '>>': 1200, '<<': 1200,
        '^': 1200,
    }
    
    OPERATOR_ASSOCIATIVITY = {
        'is': 'xfx',
        '=:=': 'xfx', '=\\=': 'xfx', '<': 'xfx', '>': 'xfx', '=<': 'xfx', '>=': 'xfx',
        '=': 'xfx', '\\=': 'xfx', '==': 'xfx', '\\==': 'xfx',
        '+': 'yfx', '-': 'yfx',
        '*': 'yfx', '/': 'yfx', 'mod': 'xfx', 'div': 'xfx',
        '**': 'xfy', '//': 'yfx', '>>': 'yfx', '<<': 'yfx',
        '->': 'xfy',
        '\\+': 'fy',
        ',': 'xfy',
        ';': 'xfy',
        ':-': 'xfx',
        '@<': 'xfx', '@>': 'xfx', '@=<': 'xfx', '@>=': 'xfx',
        '^': 'xfy',
        '=..': 'xfx',
    }
    
    def __init__(self, input_text, debug=False):
        self._clean_text = remove_comments(input_text)
        self._tokens, self._token_positions = parse_tokens_with_positions(input_text)
        self._scope = {}
        self._index = 0
        self.debug = debug
    
    def _error(self, message, token_index=None):
        if token_index is None:
            token_index = self._index
        if token_index < len(self._tokens):
            start, end = self._token_positions[token_index]
            line_no = self._clean_text[:start].count('\n') + 1
            col_no = start - self._clean_text.rfind('\n', 0, start) - 1
            lines = self._clean_text.split('\n')
            if line_no <= len(lines):
                line_text = lines[line_no - 1]
                indicator = ' ' * col_no + '^'
                return f"Ошибка: {message}\n   в строке {line_no}, колонка {col_no+1}:\n   {line_text}\n   {indicator}"
            else:
                return f"Ошибка: {message}\n   позиция токена {token_index} (start={start})"
        elif token_index == len(self._tokens) and self._token_positions:
            start, end = self._token_positions[-1]
            line_no = self._clean_text[:start].count('\n') + 1
            col_no = start - self._clean_text.rfind('\n', 0, start) - 1
            lines = self._clean_text.split('\n')
            if line_no <= len(lines):
                line_text = lines[line_no - 1]
                indicator = ' ' * col_no + '^'
                return f"Ошибка: {message}\n   в строке {line_no}, колонка {col_no+1}:\n   {line_text}\n   {indicator}"
        return f"Ошибка: {message}\n   (конец входных данных)"
    
    def parse_rules(self, return_directives=False):
        """Разбор списка правил.

        Если return_directives=False (по умолчанию), возвращает список Rule.
        Если return_directives=True, возвращает кортеж (rules, directives),
        где directives — список Term для известных директив (semantic_on и т. п.).
        Неизвестные директивы, как и раньше, игнорируются.
        """
        rules = []
        directives = []
        self._index = 0
        self._scope = {}
        
        while self._index < len(self._tokens):
            if self._current == ":-":
                d = self.parse_directive()
                if d is not None:
                    directives.append(d)
                continue
            try:
                rule = self.parse_rule_or_query()
                if rule:
                    rules.append(rule)
            except Exception as e:
                print(f"Ошибка при разборе правила: {e}")
                raise
        
        if return_directives:
            return rules, directives
        return rules
    
    def parse_query(self):
        """Разбор запроса"""
        self._index = 0
        self._scope = {}
        
        if not self._tokens:
            raise Exception(self._error("Пустой запрос"))
        
        term = self.parse_conjunction()
        
        if self._current and self._current == ".":
            self._next()
        else:
            raise Exception(self._error(f"Ожидалась '.' в конце запроса, но получено '{self._current}'"))
        
        return term
    
    @property
    def _current(self):
        return self._tokens[self._index] if self._index < len(self._tokens) else None
    
    def _next(self):
        if self._index < len(self._tokens):
            token = self._tokens[self._index]
            self._index += 1
            return token
        return None
    
    def _match(self, expected):
        if self._current == expected:
            self._index += 1
            return True
        return False
    
    def _expect(self, expected, message=None):
        if self._current == expected:
            return self._next()
        if message is None:
            message = f"Ожидался токен '{expected}', но получен '{self._current}'"
        raise Exception(self._error(message))
    
    def parse_directive(self):
        """Разобрать директиву :- Goal.

        Возвращает Term, если это известная директива из SEMANTIC_DIRECTIVE_NAMES,
        иначе None (директива просто пропускается, как раньше).
        """
        self._expect(":-", "Ожидалось ':-' для директивы")

        # Пытаемся аккуратно распарсить содержимое
        start_index = self._index
        start_scope = dict(self._scope)

        directive_term = None
        try:
            directive_term = self.parse_conjunction()
        except Exception:
            directive_term = None

        # Проверяем, что действительно дошли до точки
        if self._current == ".":
            self._next()
        else:
            # Откатываемся и используем старую «тупую» логику пропуска до точки
            self._index = start_index
            self._scope = start_scope
            directive_term = None
            paren_level = 0
            bracket_level = 0
            brace_level = 0
            while self._index < len(self._tokens):
                token = self._current
                if token == '.' and paren_level == 0 and bracket_level == 0 and brace_level == 0:
                    self._next()
                    break
                elif token == '(':
                    paren_level += 1
                elif token == ')':
                    paren_level -= 1
                elif token == '[':
                    bracket_level += 1
                elif token == ']':
                    bracket_level -= 1
                elif token == '{':
                    brace_level += 1
                elif token == '}':
                    brace_level -= 1
                self._next()

        if directive_term is None:
            return None
        from prolog.terms import Term
        if isinstance(directive_term, Term) and directive_term.functor in SEMANTIC_DIRECTIVE_NAMES:
            return directive_term
        return None
    
    def parse_atom(self):
        """Разбор атома"""
        from prolog.terms import Term, Variable, Number
        
        if not self._current:
            return None
        
        token = self._next()
        
        if token.startswith("'") and token.endswith("'"):
            content = token[1:-1]
            return Term(f"'{content}'")
        
        if token.startswith('"') and token.endswith('"'):
            content = token[1:-1]
            return Term(f'"{content}"')
        
        if re.match(NUMBER_REGEX, token):
            try:
                if '.' in token or 'e' in token.lower():
                    return Number(float(token))
                else:
                    return Number(int(token))
            except:
                return Term(token)
        
        if re.match(VARIABLE_REGEX, token):
            if token == "_":
                return Variable("_")
        
            variable = self._scope.get(token)
            if variable is None:
                self._scope[token] = Variable(token)
                variable = self._scope[token]
            
            return variable
        
        if re.match(ATOM_NAME_REGEX, token) or token in self.OPERATOR_PRECEDENCE:
            try:
                if '.' in token or 'e' in token.lower():
                    val = float(token)
                else:
                    val = int(token)
                return Number(val)
            except:
                return Term(token)
        
        return Term(token)
    
    def parse_primary(self):
        """Разбор первичного выражения"""
        from prolog.terms import Term, Variable, Number

        if not self._current:
            return None

        if self._current == '.':
            return None

        if self._current == '\\+':
            self._next()
            expr = self.parse_primary()
            return Term('\\+', [expr])

        if self._current == "[":
            return self.parse_list()

        if self._current == "(":
            self._next()
            expr = self.parse_expression()
            self._expect(")", "Ожидалось ')' после выражения в скобках")
            return expr

        if self._current == "-":
            current_index = self._index
            if self._index < len(self._tokens):
                next_token = self._tokens[self._index]
                if re.match(NUMBER_REGEX, next_token):
                    self._next()
                    token = self._next()
                    try:
                        if '.' in token or 'e' in token.lower():
                            return Number(-float(token))
                        else:
                            return Number(-int(token))
                    except:
                        return Term("-" + token)
                else:
                    self._next()
                    expr = self.parse_primary()
                    if isinstance(expr, Number):
                        return Number(-expr.value)
                    else:
                        return Term("-", [expr])
            else:
                self._next()
                expr = self.parse_primary()
                return Term("-", [expr])

        if self._current == '+':
            self._next()
            expr = self.parse_primary()
            if expr is None:
                raise Exception(self._error("Ожидалось выражение после '+'"))
            return expr

        atom = self.parse_atom()

        if isinstance(atom, Term) and not atom.arguments:
            name = atom.functor

            if name in ["!", "\\+"]:
                return atom

            if self._current == "(":
                self._next()
                if self._current == ")":
                    self._next()
                    return Term(name, [])
                arguments = self.parse_arguments()
                self._expect(")", "Ожидалось ')' после аргументов")
                return Term(name, arguments)
            else:
                return atom

        return atom
    
    def parse_expression(self, min_precedence=0):
        """Разбор выражения с учетом приоритета операторов (алгоритм Пратта)"""
        from prolog.terms import Term
        
        left = self.parse_primary()
        
        if left is None:
            return None

        while self._current in self.OPERATOR_PRECEDENCE:
            op = self._current
            precedence = self.OPERATOR_PRECEDENCE.get(op, 0)
            
            if precedence < min_precedence:
                break
            
            if op == ',' and min_precedence > self.OPERATOR_PRECEDENCE[',']:
                break
            
            assoc = self.OPERATOR_ASSOCIATIVITY.get(op, 'xfx')
            
            self._next()
            
            next_precedence = precedence if assoc == 'xfy' else precedence + 1
            
            right = self.parse_expression(next_precedence)
            
            if right is None:
                raise Exception(self._error(f"Ожидалось выражение после оператора {op}"))
            
            left = Term(op, [left, right])
        
        return left
    
    def parse_list(self):
        """Разбор списка"""
        from prolog.terms import Term
        
        self._expect("[")
        
        if self._match("]"):
            return Term("[]", [])
        
        elements = []
        
        while True:
            element = self.parse_expression(min_precedence=1001)
            if element is None:
                raise Exception(self._error("Ожидался элемент списка"))
            
            elements.append(element)
            
            if self._current == ",":
                self._next()
                continue
            elif self._current == "|":
                self._next()
                tail = self.parse_expression()
                if tail is None:
                    raise Exception(self._error("Ожидалось выражение после '|'"))
                self._expect("]", "Ожидалось ']' после хвоста списка")
                result = tail
                for element in reversed(elements):
                    result = Term('.', [element, result])
                return result
            elif self._current == "]":
                self._next()
                break
            else:
                raise Exception(self._error(f"Ожидался ',', '|' или ']', но получено '{self._current}'"))
        
        result = Term("[]", [])
        for element in reversed(elements):
            result = Term('.', [element, result])
        
        return result
    
    def parse_arguments(self):
        """Разбор аргументов терма"""
        from prolog.terms import Term
        
        arguments = []
        
        if self._current == ")":
            return arguments
        
        while self._current and self._current != ")":
            arg = self.parse_expression(min_precedence=1001)
            
            if arg is None:
                raise Exception(self._error(f"Ожидалось выражение для аргумента, но получено '{self._current}'"))
            
            arguments.append(arg)
            
            if self._current == ",":
                self._next()
                while self._current == ",":
                    self._next()
            elif self._current == ")":
                break
            else:
                raise Exception(self._error(f"Ожидалась ',' или ')', но получено '{self._current}'"))
        
        return arguments
    
    def parse_rule_or_query(self):
        """Разбор правила или факта (с поддержкой оператора :- в термах)"""
        from prolog.terms import Rule, Term
        expr = self.parse_expression(min_precedence=1201)
        if expr is None:
            raise Exception(self._error("Ожидалось выражение для головы правила"))

        if isinstance(expr, Term) and expr.functor == ':-' and len(expr.arguments) == 2:
            head = expr.arguments[0]
            body = expr.arguments[1]
            if self._current == '.':
                self._next()
                return Rule(head, body)
            else:
                raise Exception(self._error(f"Ожидалась '.' в конце правила, но получено '{self._current}'"))

        head = expr
        if self._current == ".":
            self._next()
            return Rule(head, Term("true", []))
        if self._current == ":-":
            self._next()
            body = self.parse_conjunction()
            if body is None:
                raise Exception(self._error("Ожидалось тело правила после ':-'"))
            if self._current != ".":
                raise Exception(self._error(f"Ожидалась '.' в конце правила, но получено '{self._current}'"))
            self._next()
            return Rule(head, body)
        return Rule(head, Term("true", []))
        
    def parse_conjunction(self):
        """Разбор списка выражений, разделенных запятыми как конъюнкции"""
        from prolog.terms import Term
        goals = []
        while True:
            goal = self.parse_expression(min_precedence=0)
            if goal is None:
                break
            goals.append(goal)
            if self._current == ',':
                self._next()
                continue
            else:
                break
        if len(goals) == 0:
            return None
        elif len(goals) == 1:
            return goals[0]
        else:
            result = goals[-1]
            for g in reversed(goals[:-1]):
                result = Term(',', [g, result])
            return result