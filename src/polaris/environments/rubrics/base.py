"""
Base class for task success/progress rubrics.
"""

from dataclasses import dataclass
from typing import Callable
from isaaclab.envs import ManagerBasedRLEnv


@dataclass
class RubricResult:
    """Result from evaluating a rubric."""

    success: bool  # Binary task success
    progress: float  # Progress score 0.0 - 1.0
    metrics: dict[str, float]  # Additional metrics for logging


class Rubric:
    """
    Rubrics compute success/progress by inspecting simulation state.
    They're called after each step and on reset to populate info dict.
    """

    def __init__(self, criteria: list[Callable | tuple[Callable, list[int]]], **kwargs):
        """
        Initialize the rubric with access to the environment.

        Args:
            env: The ManagerBasedRLEnv instance
            **kwargs: Task-specific configuration
        """
        self.config = kwargs
        self.criteria = criteria
        self.criteria_reached = [False] * len(criteria)

    def evaluate(self, env: ManagerBasedRLEnv) -> RubricResult:
        """
        Evaluate current simulation state and return result.

        Supports criteria with optional dependencies.
        Criteria can be:
            - callable (no dependency, can be achieved in any order)
            - (callable, [dep_indices]) (only counts if all deps by index are met)
        This allows for some to require others, but leaves most unconstrained.

        Tracks the max-ever reached state for each criterion using self.criteria_reached.
        """
        metrics = {}
        num_criteria = len(self.criteria)

        criteria_met_now = []
        for idx, c in enumerate(self.criteria):
            # Check if c is (callable, [deps]), else treat as callable only
            if isinstance(c, tuple):
                fn, deps = c
                # Only evaluate if all deps ever reached
                deps_met = all(self.criteria_reached[d] for d in deps)
                result = fn(env) if deps_met else False
            else:
                fn = c
                result = fn(env)
            # Update max-ever reached for this criterion
            self.criteria_reached[idx] = self.criteria_reached[idx] or bool(result)
            criteria_met_now.append(bool(result))

        num_reached_ever = sum(self.criteria_reached)
        progress = num_reached_ever / num_criteria if num_criteria > 0 else 0.0
        metrics["criteria_ever_reached"] = num_reached_ever
        metrics["criteria_total"] = num_criteria

        success = num_reached_ever == num_criteria
        return RubricResult(success=success, progress=progress, metrics=metrics)

    def reset(self):
        """Called when environment resets. Override for stateful rubrics."""
        self.criteria_reached = [False] * len(self.criteria)
