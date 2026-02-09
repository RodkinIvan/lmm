from __future__ import annotations

from typing import Any

from lmm.memory.base import MemoryContext, MemoryModule


class IdentityMemoryModule(MemoryModule):
    name = "identity"

    def rewrite(self, hidden_states: Any, context: MemoryContext) -> Any:
        _ = context
        return hidden_states
