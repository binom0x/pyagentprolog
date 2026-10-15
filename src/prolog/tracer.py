""" PyAgentProlog

Module for execution tracing and building a proof tree
(fixed success logic: a step is successful only if it was explicitly
finished with success=True).

(c) 2026 Andrey Vorfolomeev

version 1.0.1
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import uuid

@dataclass
class ProofStep:
    """A single proof step"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    goal: str = ""
    rule_used: Optional[str] = None
    unification: List[Tuple[str, str]] = field(default_factory=list)
    success: bool = False
    children: List['ProofStep'] = field(default_factory=list)
    subst_before: Dict[str, str] = field(default_factory=dict)
    subst_after: Dict[str, str] = field(default_factory=dict)
    depth: int = 0
    builtin: bool = False

    def __str__(self) -> str:
        indent = "  " * self.depth
        result = f"{indent}{self.goal}"
        if self.rule_used:
            result += f" [by rule: {self.rule_used}]"
        if self.unification:
            unif_str = ", ".join(f"{var}={val}" for var, val in self.unification)
            result += f" [unification: {unif_str}]"
        if self.builtin:
            result += " [built-in predicate]"
        result += f" - {'SUCCESS' if self.success else 'FAILURE'}"
        return result

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'goal': self.goal,
            'rule_used': self.rule_used,
            'unification': self.unification,
            'success': self.success,
            'depth': self.depth,
            'builtin': self.builtin,
            'subst_before': self.subst_before,
            'subst_after': self.subst_after,
            'children': [child.to_dict() for child in self.children]
        }
        
    def to_tree_string(self) -> str:
        result = str(self) + "\n"
        for child in self.children:
            result += child.to_tree_string()
        return result


class ProofTracer:
    """Tracer for recording the proof process"""
    
    def __init__(self):
        self.root_steps: List[ProofStep] = []
        self.current_step: Optional[ProofStep] = None
        self.step_stack: List[ProofStep] = []
    
    def start_goal(self, goal_str: str, depth: int = 0) -> ProofStep:
        step = ProofStep(goal=goal_str, depth=depth)
        if not self.step_stack:
            self.root_steps.append(step)
        else:
            parent = self.step_stack[-1]
            parent.children.append(step)
        self.step_stack.append(step)
        self.current_step = step
        return step
    
    def end_goal(self, success: bool = True):
        """Finish the current goal and set the success flag"""
        if self.step_stack:
            step = self.step_stack.pop()
            step.success = success
            if self.step_stack:
                self.current_step = self.step_stack[-1]
            else:
                self.current_step = None
    
    def add_unification(self, var: str, value: str):
        if self.current_step:
            self.current_step.unification.append((var, value))
    
    def set_rule(self, rule_str: str):
        if self.current_step:
            self.current_step.rule_used = rule_str
    
    def set_builtin(self, builtin: bool = True):
        if self.current_step:
            self.current_step.builtin = builtin
    
    def set_substitutions(self, before: Dict, after: Dict):
        if self.current_step:
            self.current_step.subst_before = {str(k): str(v) for k, v in before.items()}
            self.current_step.subst_after = {str(k): str(v) for k, v in after.items()}
    
    def get_proof_tree(self) -> str:
        result = "PROOF TREE:\n"
        result += "=" * 50 + "\n"
        for step in self.root_steps:
            result += step.to_tree_string()
        return result
    
    def get_proof_dict(self) -> Dict[str, Any]:
        return {
            'steps': [step.to_dict() for step in self.root_steps],
            'success': any(step.success for step in self.root_steps)
        }
    
    def clear(self):
        self.root_steps.clear()
        self.current_step = None
        self.step_stack.clear()