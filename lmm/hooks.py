from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List

HookCallback = Callable[[Dict[str, Any]], None]


class HookManager:
    def __init__(self) -> None:
        self._callbacks: DefaultDict[str, List[HookCallback]] = defaultdict(list)

    def register(self, event: str, callback: HookCallback) -> None:
        self._callbacks[event].append(callback)

    def emit(self, event: str, info: Dict[str, Any]) -> None:
        for callback in self._callbacks.get(event, []):
            callback(info)


def hook_last_user_response(info: Dict[str, Any]) -> None:
    user_message = info.get("user_message", "")
    info["last_user_response"] = user_message
    hooked_fields = info.setdefault("hooked_fields", [])
    if isinstance(hooked_fields, list) and "last_user_response" not in hooked_fields:
        hooked_fields.append("last_user_response")
    callbacks = info.setdefault("callbacks", [])
    if isinstance(callbacks, list) and "hook_last_user_response" not in callbacks:
        callbacks.append("hook_last_user_response")
