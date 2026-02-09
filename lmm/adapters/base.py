from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any
from typing import Dict, List


class ModelAdapter(ABC):
    backend_name: str = "base"

    @abstractmethod
    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate one assistant reply from chat messages."""

    def on_user_message(self, user_text: str, messages: List[Dict[str, str]]) -> None:
        """Optional callback fired when a new user message is appended."""
        _ = (user_text, messages)

    def preload(self) -> None:
        """Optional startup hook to warm up/load backend resources."""
        return

    def load_memory(self, path: str) -> dict[str, Any]:
        """Optional hook to load memory-module state."""
        _ = path
        return {"loaded": False, "reason": "load-memory-not-implemented"}

    def save_memory(self, path: str) -> dict[str, Any]:
        """Optional hook to save memory-module state."""
        _ = path
        return {"saved": False, "reason": "save-memory-not-implemented"}
