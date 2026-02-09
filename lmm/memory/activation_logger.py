from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lmm.memory.base import MemoryContext


def _slice_tail(hidden_states: Any) -> Any:
    try:
        return hidden_states[:, -1, -5:]
    except Exception:
        return "<slice_unavailable>"


def _to_serializable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    return {
        "repr": repr(value),
        "shape": list(shape) if shape is not None else None,
        "dtype": str(dtype) if dtype is not None else None,
    }


class ActivationLogger:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _write(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True))
            f.write("\n")

    def log_activation(
        self,
        *,
        context: MemoryContext,
        input_hidden_states: Any,
        output_hidden_states: Any,
    ) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "activation",
            "context": asdict(context),
            "input_slice": _to_serializable(_slice_tail(input_hidden_states)),
            "output_slice": _to_serializable(_slice_tail(output_hidden_states)),
        }
        self._write(payload)

    def log_update(
        self,
        *,
        memory_module_name: str,
        stage: str,
        info: dict[str, Any],
    ) -> None:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": "update",
            "memory_module": memory_module_name,
            "stage": stage,
            "info": _to_serializable(info),
        }
        self._write(payload)
