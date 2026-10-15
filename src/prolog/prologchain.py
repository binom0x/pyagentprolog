""" PyAgentProlog

LangChain integration.

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""

from typing import Dict, Any, List, Union, Optional, ClassVar
from langchain_core.runnables import RunnableSerializable, RunnableSequence, RunnableLambda
from langchain_core.prompts import StringPromptTemplate, BasePromptTemplate
from pydantic import Field, ConfigDict, PrivateAttr
import re
from prolog.solver import Solver, ParseSolution


class PrologQueryTemplate(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """Template for Prolog queries, compatible with LangChain Expression Language"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    template: str = Field(description="Prolog query template")
    
    # Private attributes
    _input_variables: List[str] = PrivateAttr(default_factory=list)
    
    def __init__(self, template: str, **kwargs):
        """Initialize the Prolog query template"""
        # Extract variables from the template
        input_variables = self._extract_variables(template)
        
        super().__init__(template=template, **kwargs)
        self._input_variables = input_variables
    
    @classmethod
    def from_template(cls, template: str, **kwargs) -> "PrologQueryTemplate":
        """Create a template from a string (similar to ChatPromptTemplate.from_template)"""
        return cls(template=template, **kwargs)
    
    @staticmethod
    def _extract_variables(template: str) -> List[str]:
        """Extract variables from the template"""
        variables = re.findall(r'\{(\w+)\}', template)
        return list(set(variables))
    
    @property
    def input_variables(self) -> List[str]:
        """Get the list of variables"""
        return self._input_variables
    
    def format(self, **kwargs: Any) -> str:
        """Format the template with variables"""
        try:
            # Check that all variables are provided
            for var in self.input_variables:
                if var not in kwargs:
                    raise ValueError(f"Missing variable '{var}' in template '{self.template}'")
            
            return self.template.format(**kwargs)
        except KeyError as e:
            raise ValueError(f"Missing variable {e} in template '{self.template}'")
    
    def format_prompt(self, **kwargs: Any) -> Dict[str, Any]:
        """Format for creating a prompt"""
        return {"query": self.format(**kwargs)}
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Execute the template — format the query"""
        try:
            # If query is passed directly, use it
            if "query" in input:
                return {"query": input["query"]}
            
            # Otherwise, format the template
            query = self.format(**input)
            return {"query": query}
        except Exception as e:
            print(f"Error formatting the Prolog query: {e}")
            return {"error": str(e)}
    
    def __or__(self, other):
        """Support the | operator for chains"""
        # Use the standard LangChain chain
        return RunnableSequence(first=self, last=other)


class PrologChain(RunnableSerializable[Dict[str, Any], Dict[str, Any]]):
    """Runnable chain for executing Prolog queries"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    # Pydantic fields
    program: str = Field(description="Prolog program")
    include_explanation: bool = Field(default=False, description="Include explanations")
    
    # Private attributes (not Pydantic fields)
    _solver: Optional[Solver] = PrivateAttr(default=None)
    
    def __init__(self, 
                 program: str,
                 context: Optional[Dict],
                 include_explanation: bool = False):
        """
        Initialize the Prolog chain.
        
        Args:
            program: Prolog program
            context: external context — the environment that predicates
                     may access; a dict of the form {"object_name": python_object}
            include_explanation: Whether to include explanations
        """
        super().__init__(
            program=program,
            include_explanation=include_explanation
        )
        
        # Initialize the solver
        self._solver = Solver(program,context=context)
    
    def _format_solution_for_output(self, solution_data) -> str:
        """Format the solution for output"""
        if solution_data is True:
            return "Query executed successfully"
        elif solution_data is False:
            return "No solutions"
        elif isinstance(solution_data, dict):
            # Dict with a single solution
            items = []
            for var_name, value in solution_data.items():
                # Format the value
                val_str = self._format_prolog_term(value)
                items.append(f"{var_name} = {val_str}")
            
            if len(items) == 1:
                return f"Solution: {items[0]}"
            else:
                return f"Solutions: {', '.join(items)}"
        elif isinstance(solution_data, list):
            # List of multiple solutions
            results = []
            for i, solution in enumerate(solution_data, 1):
                items = []
                for var_name, value in solution.items():
                    val_str = self._format_prolog_term(value)
                    items.append(f"{var_name} = {val_str}")
                results.append(f"Solution {i}: {', '.join(items)}")
            return "\n".join(results)
        else:
            return str(solution_data)
    
    def _format_prolog_term(self, term) -> str:
        """Format a Prolog term into a string"""
        # Import here to avoid a circular import
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
        
        # If the import failed, use the string representation
        return str(term)
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> Dict[str, Any]:
        """Execute the Prolog query"""
        
        # Get the query
        query = input.get("query", "")
        if not query:
            # If there is no query, try to use the whole input as the query
            # (e.g., if a string was passed directly)
            if isinstance(input, str):
                query = input
            else:
                raise ValueError("Input must contain 'query' or be a string")
        
        # Execute the query
        try:
            if self.include_explanation:
                solutions, explanation = self._solver.find_solutions(
                    query, 
                    with_explanation=True
                )
            else:
                solutions = self._solver.find_solutions(query, with_explanation=False)
                explanation = None
            
            # Format the result
            parsed_result = ParseSolution(query, solutions, explanation)
            
            # Create output if it is empty
            output = parsed_result.get("output", "")
            if not output and "solution" in parsed_result:
                output = self._format_solution_for_output(parsed_result["solution"])
            
            # Result structure
            result = {
                "success": True,
                "query": query,
                "program": self.program,
                "explanation": explanation,
                "output": output,
                "count": parsed_result.get("count", 0)
            }
            
            # Add solution depending on its type
            solution = parsed_result.get("solution")
            if solution is not None:
                if isinstance(solution, dict) or isinstance(solution, list):
                    result["solutions"] = solution
                elif solution in [True, False]:
                    result["result"] = solution
            
            return result
            
        except Exception as e:
            print(f"Error executing the Prolog query: {e}")
            return {
                "success": False,
                "query": query,
                "program": self.program,
                "error": str(e)
            }


class PrologOutputFormatter(RunnableSerializable):
    """Formatting Prolog output for an LLM"""
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    include_query: bool = Field(default=True, description="Include the query in the output")
    include_stats: bool = Field(default=True, description="Include statistics")
    
    def invoke(self, 
               input: Dict[str, Any],
               config: Optional[Dict[str, Any]] = None,
               **kwargs: Any) -> str:
        """Format the Prolog result into a string"""
        
        if not isinstance(input, dict):
            return str(input)
        
        if not input.get("success", False):
            return f"Prolog error: {input.get('error', 'Unknown error')}"
        
        result_str = "Prolog query result:\n"
        
        if self.include_query:
            result_str += f"Query: {input.get('query', 'N/A')}\n"
        
        if self.include_stats:
            count = input.get("count", 0)
            result_str += f"Number of solutions: {count}\n"
        
        output = input.get("output", "")
        if output:
            result_str += f"\n{output}"
        
        return result_str


# Factory functions for convenience

def create_prolog_chain(program: str, include_explanation: bool = False) -> PrologChain:
    """Create a Prolog chain"""
    return PrologChain(program=program, include_explanation=include_explanation)


def create_prolog_query_template(template: str) -> PrologQueryTemplate:
    """Create a Prolog query template"""
    return PrologQueryTemplate.from_template(template)


# Simplified variant for quickly creating chains

def prolog_pipeline(program: str, query_template: Optional[str] = None) -> RunnableSerializable:
    """
    Create a ready Prolog chain
    
    Args:
        program: Prolog program
        query_template: Query template (optional)
    
    Returns:
        Ready chain
    """
    prolog_chain = PrologChain(program=program)
    
    if query_template:
        query_template_obj = PrologQueryTemplate.from_template(query_template)
        return query_template_obj | prolog_chain
    else:
        return prolog_chain


# Usage example
if __name__ == "__main__":
    # Test 1: Chain with a template
    print("Test 1: Chain with a template and | operator")
    
    program = """
    parent(ivan, alexey).
    parent(alexey, peter).
    """
    
    # Creating a chain as in the example
    query_template = PrologQueryTemplate.from_template("parent({person}, X).")
    prolog = PrologChain(program=program)
    
    chain = query_template | prolog
    
    result = chain.invoke({"person": "ivan"})
    print(f"Result: {result}")
    print(f"Output: {result.get('output')}")
    print()
    
    # Test 2: Alternative way
    print("Test 2: Using prolog_pipeline")
    
    chain2 = prolog_pipeline(
        program=program,
        query_template="parent({person}, X)."
    )
    
    result2 = chain2.invoke({"person": "ivan"})
    print(f"Result: {result2}")
    print()
    
    # Test 3: Chain without a template
    print("Test 3: Chain without a template")
    
    chain3 = prolog_pipeline(program=program)
    result3 = chain3.invoke({"query": "parent(ivan, X)."})
    print(f"Result: {result3}")
    
    # Test 4: Direct call with a string
    print("\nTest 4: Direct call with a string")
    result4 = chain3.invoke("parent(ivan, X).")
    print(f"Result: {result4}")