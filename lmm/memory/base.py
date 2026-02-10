from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True, slots=True)
class MemoryContext:
    layer_index: int
    total_layers: int
    call_index: int
    backend: str
    model_id: str


class MemoryModule(ABC):
    name: str = "base"

    def select_layer_index(self, total_layers: int) -> int:
        if total_layers <= 0:
            raise ValueError("total_layers must be > 0")
        return total_layers // 2

    @abstractmethod
    def rewrite(self, hidden_states: Any, context: MemoryContext) -> Any:
        """Rewrite hidden states and return the new hidden states."""

    def update(self, info: dict[str, Any]) -> None:
        """Receive event updates from hooks/callbacks."""
        _ = info

    def attach_runtime(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_builder: Callable[[list[dict[str, str]], bool], str],
    ) -> None:
        """Attach runtime objects needed for advanced modules."""
        _ = (model, tokenizer, prompt_builder)

    def set_prefix_cache(self, cache: Any) -> None:
        """Provide cache for prefix context (optional)."""
        _ = cache

    def save(self, path: str) -> dict[str, Any]:
        """Persist module state to disk."""
        _ = path
        return {"saved": False, "reason": "save-not-implemented"}

    def load(self, path: str) -> dict[str, Any]:
        """Load module state from disk."""
        _ = path
        return {"loaded": False, "reason": "load-not-implemented"}
