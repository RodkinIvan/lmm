from __future__ import annotations

from typing import Dict, List

from lmm.adapters.base import ModelAdapter


class ChatEngine:
    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        system_prompt: str = "",
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> None:
        self.adapter = adapter
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.messages: List[Dict[str, str]] = []
        if system_prompt.strip():
            self.messages.append({"role": "system", "content": system_prompt.strip()})

    def ask(self, user_text: str) -> str:
        self.messages.append({"role": "user", "content": user_text})
        self.adapter.on_user_message(user_text, self.messages)
        reply = self.adapter.generate_reply(
            self.messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        self.messages.append({"role": "assistant", "content": reply})
        return reply
