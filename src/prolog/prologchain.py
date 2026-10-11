""" PyAgentProlog

Интеграция с langchain

(c) 2026 Andrey Vorfolomeev (Ворфоломеев Андрей Викторович)

версия 5.9.1
"""

from typing import Dict, Any, List, Union, Optional, ClassVar
from langchain_core.runnables import RunnableSerializable, RunnableSequence, RunnableLambda
from langchain_core.prompts import StringPromptTemplate, BasePromptTemplate
from pydantic import Field, ConfigDict, PrivateAttr
import re
from prolog.solver import Solver, ParseSolution


class PrologQueryTemplate(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """Шаблон для Prolog-запросов, совместимый с LangChain Expression Language"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    template: str = Field(description="Шаблон Prolog-запроса")
    
    # Приватные атрибуты
    _input_variables: List[str] = PrivateAttr(default_factory=list)
    
    def __init__(self, template: str, **kwargs):
        """Инициализация шаблона Prolog-запроса"""
        # Извлекаем переменные из шаблона
        input_variables = self._extract_variables(template)
        
        super().__init__(template=template, **kwargs)
        self._input_variables = input_variables
    
    @classmethod
    def from_template(cls, template: str, **kwargs) -> "PrologQueryTemplate":
        """Создание шаблона из строки (аналогично ChatPromptTemplate.from_template)"""
        return cls(template=template, **kwargs)
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Извлечение переменных из шаблона"""
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))
    
    @property
    def input_variables(self) -> List[str]:
        """Получение списка переменных"""
        return self._input_variables
    
    def format(self, **kwargs: Any) -> str:
        """Форматирование шаблона с переменными"""
        try:
            # Проверяем, что все переменные предоставлены
            for var in self.input_variables:
                if var not in kwargs:
                    raise ValueError(f"Missing variable '{var}' in template '{self.template}'")
            
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} in template '{self.template}'")
    
    def format_prompt(self, **kwargs: Any) -> Dict[str, Any]:
        """Форматирование для создания промпта"""
        return {"query": self.format(**kwargs)}
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Выполнение шаблона - форматирование запроса"""
        try:
            # Если передали query напрямую, используем его
            if "query" in input:
                return {"query": input["query"]}
            
            # Иначе форматируем шаблон
            query = self.format(**input)
            return {"query": query}
        except Exception as e:
            print(f"Ошибка форматирования Prolog-запроса: {e}")
            return {"error": str(e)}
    
    def __or__(self, other):
        """Поддержка оператора | для цепочек"""
        # Используем стандартную цепочку LangChain
        return RunnableSequence(first=self, last=other)


class PrologChain(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """Runnable-цепь для выполнения Prolog-запросов"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Pydantic поля
    program: str = Field(description="Prolog программа")
    include_explanation: bool = Field(default=False, description="Включать объяснения")
    
    # Приватные атрибуты (не являются Pydantic полями)
    _solver: Optional[Solver] = PrivateAttr(default=None)
    
    def __init__(self, 
                 program: str,
                 context: Optional[Dict],
                 include_explanation: bool = False):
        """
        Инициализация Prolog цепи.
        
        Args:
            program: Программа на Prolog
            context: внешний контекст - окружение, к котрому могут обращаться предикаты
                     словарь вида {"object_name":python_object}
            include_explanation: Включать ли объяснения
        """
        super().__init__(
            program=program,
            include_explanation=include_explanation
        )
        
        # Инициализируем решатель
        self._solver = Solver(program,context=context)
    
    def _format_solution_for_output(self, solution_data) -> str:
        """Форматирование решения для вывода в output"""
        if solution_data is True:
            return "Запрос успешно выполнен"
        elif solution_data is False:
            return "Нет решений"
        elif isinstance(solution_data, dict):
            # Словарь с одним решением
            items = []
            for var_name, value in solution_data.items():
                # Форматируем значение
                val_str = self._format_prolog_term(value)
                items.append(f"{var_name} = {val_str}")
            
            if len(items) == 1:
                return f"Решение: {items[0]}"
            else:
                return f"Решения: {', '.join(items)}"
        elif isinstance(solution_data, list):
            # Список нескольких решений
            results = []
            for i, solution in enumerate(solution_data, 1):
                items = []
                for var_name, value in solution.items():
                    val_str = self._format_prolog_term(value)
                    items.append(f"{var_name} = {val_str}")
                results.append(f"Решение {i}: {', '.join(items)}")
            return "\n".join(results)
        else:
            return str(solution_data)
    
    def _format_prolog_term(self, term) -> str:
        """Форматирование Prolog-терма в строку"""
        # Импортируем здесь, чтобы избежать циклического импорта
        try:
            from prolog.terms import Term, Variable, Number
            
            if isinstance(term, Number):
                return str(term.value)
            elif isinstance(term, Variable):
                return term.name
            elif isinstance(term, Term):
                if term.arguments:
                    args_str = ', '.join(self._format_prolog_term(arg) for arg in term.arguments)
                    return f"{term.functor}({args_str})"
                else:
                    return str(term.functor)
        except ImportError:
            pass
        
        # Если не смогли импортировать, используем строковое представление
        return str(term)
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Выполнение Prolog-запроса"""
        
        # Получаем запрос
        query = input.get("query", "")
        if not query:
            # Если query нет, попробуем использовать весь input как query
            # (например, если передали строку напрямую)
            if isinstance(input, str):
                query = input
            else:
                raise ValueError("Входные данные должны содержать 'query' или быть строкой")
        
        # Выполняем запрос
        try:
            if self.include_explanation:
                solutions, explanation = self._solver.find_solutions(
                    query, 
                    with_explanation=True
                )
            else:
                solutions = self._solver.find_solutions(query, with_explanation=False)
                explanation = None
            
            # Форматируем результат
            parsed_result = ParseSolution(query, solutions, explanation)
            
            # Создаем output, если он пустой
            output = parsed_result.get("output", "")
            if not output and "solution" in parsed_result:
                output = self._format_solution_for_output(parsed_result["solution"])
            
            # Структура результата
            result = {
                "success": True,
                "query": query,
                "program": self.program,
                "explanation": explanation,
                "output": output,
                "count": parsed_result.get("count", 0)
            }
            
            # Добавляем solution в зависимости от типа
            solution = parsed_result.get("solution")
            if solution is not None:
                if isinstance(solution, dict) or isinstance(solution, list):
                    result["solutions"] = solution
                elif solution in [True, False]:
                    result["result"] = solution
            
            return result
            
        except Exception as e:
            print(f"Ошибка выполнения Prolog-запроса: {e}")
            return {
                "success": False,
                "query": query,
                "program": self.program,
                "error": str(e)
            }


class PrologOutputFormatter(RunnableSerializable):
    """Форматирование вывода Prolog для LLM"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    include_query: bool = Field(default=True, description="Включать запрос в вывод")
    include_stats: bool = Field(default=True, description="Включать статистику")
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> str:
        """Форматирование результата Prolog в строку"""
        
        if not isinstance(input, dict):
            return str(input)
        
        if not input.get("success", False):
            return f"Ошибка Prolog: {input.get('error', 'Неизвестная ошибка')}"
        
        result_str = "Результат Prolog-запроса:\n"
        
        if self.include_query:
            result_str += f"Запрос: {input.get('query', 'N/A')}\n"
        
        if self.include_stats:
            count = input.get("count", 0)
            result_str += f"Количество решений: {count}\n"
        
        output = input.get("output", "")
        if output:
            result_str += f"\n{output}"
        
        return result_str


# Фабричные функции для удобства

def create_prolog_chain(program: str, include_explanation: bool = False) -> PrologChain:
    """Создание Prolog-цепи"""
    return PrologChain(program=program, include_explanation=include_explanation)


def create_prolog_query_template(template: str) -> PrologQueryTemplate:
    """Создание шаблона Prolog-запроса"""
    return PrologQueryTemplate.from_template(template)


# Упрощенный вариант для быстрого создания цепочек

def prolog_pipeline(program: str, query_template: Optional[str] = None) -> RunnableSerializable:
    """
    Создание готовой Prolog-цепи
    
    Args:
        program: Программа на Prolog
        query_template: Шаблон запроса (опционально)
    
    Returns:
        Готовая цепь
    """
    prolog_chain = PrologChain(program=program)
    
    if query_template:
        query_template_obj = PrologQueryTemplate.from_template(query_template)
        return query_template_obj | prolog_chain
    else:
        return prolog_chain


# Пример использования
if __name__ == "__main__":
    # Тест 1: Цепь с шаблоном
    print("Тест 1: Цепь с шаблоном и оператором |")
    
    program = """
    родитель(иван, алексей).
    родитель(алексей, петр).
    """
    
    # Создание цепи как в примере
    query_template = PrologQueryTemplate.from_template("родитель({person}, X).")
    prolog = PrologChain(program=program)
    
    chain = query_template | prolog
    
    result = chain.invoke({"person": "иван"})
    print(f"Результат: {result}")
    print(f"Output: {result.get('output')}")
    print()
    
    # Тест 2: Альтернативный способ
    print("Тест 2: Использование prolog_pipeline")
    
    chain2 = prolog_pipeline(
        program=program,
        query_template="родитель({person}, X)."
    )
    
    result2 = chain2.invoke({"person": "иван"})
    print(f"Результат: {result2}")
    print()
    
    # Тест 3: Цепь без шаблона
    print("Тест 3: Цепь без шаблона")
    
    chain3 = prolog_pipeline(program=program)
    result3 = chain3.invoke({"query": "родитель(иван, X)."})
    print(f"Результат: {result3}")
    
    # Тест 4: Прямой вызов строкой
    print("\nТест 4: Прямой вызов строкой")
    result4 = chain3.invoke("родитель(иван, X).")
    print(f"Результат: {result4}")