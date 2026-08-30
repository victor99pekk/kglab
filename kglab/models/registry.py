"""Model registry — load trained models by task name.

Maps task names (e.g. ``"entity_resolution"``) to their inference tool
classes.  Pipelines use ``ModelRegistry.load(task, checkpoint_path)``
to get an inference-ready tool without importing training code.

Usage:
    from kglab.models import ModelRegistry

    tool = ModelRegistry.load(
        "entity_resolution",
        "experiments/ML_models/001_er/models/best.pt",
    )
    is_match = tool.resolve(entity_a, entity_b)
"""

from __future__ import annotations

from typing import Protocol


class _InferenceTool(Protocol):
    """Protocol that all inference tools must satisfy."""

    def load(self, path: str) -> None:
        """Load a trained checkpoint from disk."""
        ...


class ModelRegistry:
    """Central registry for loading trained model checkpoints.

    Register new inference tools here when you add a new ML task::

        from kglab.models.entity_resolution import EntityResolutionTool
        ModelRegistry._tools["entity_resolution"] = EntityResolutionTool
    """

    _tools: dict[str, type[_InferenceTool]] = {}

    @classmethod
    def register(cls, task: str, tool_cls: type[_InferenceTool]) -> None:
        """Register an inference tool class for a task name."""
        cls._tools[task] = tool_cls

    @classmethod
    def load(cls, task: str, checkpoint_path: str) -> _InferenceTool:
        """Load a trained model by task name and checkpoint path.

        Args:
            task: Registered task name (e.g. ``"entity_resolution"``).
            checkpoint_path: Path to the saved model checkpoint.

        Returns:
            An inference-ready tool instance.

        Raises:
            KeyError: If *task* is not registered.
            FileNotFoundError: If *checkpoint_path* does not exist.
        """
        if task not in cls._tools:
            available = ", ".join(sorted(cls._tools.keys())) or "(none)"
            raise KeyError(f"Unknown model task: '{task}'. Registered tasks: {available}")
        tool = cls._tools[task]()
        tool.load(checkpoint_path)
        return tool

    @classmethod
    def available_tasks(cls) -> list[str]:
        """Return the list of registered task names."""
        return list(cls._tools.keys())
