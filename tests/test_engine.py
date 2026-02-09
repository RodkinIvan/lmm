from lmm.adapters.base import ModelAdapter
from lmm.engine import ChatEngine


class StubAdapter(ModelAdapter):
    backend_name = "stub"

    def __init__(self):
        self.last_messages = None
        self.last_user_message = None

    def generate_reply(self, messages, *, max_tokens, temperature, top_p) -> str:
        self.last_messages = messages
        return f"reply-{len(messages)}-{max_tokens}-{temperature}-{top_p}"

    def on_user_message(self, user_text, messages) -> None:
        self.last_user_message = user_text


def test_chat_engine_tracks_history_and_calls_adapter():
    adapter = StubAdapter()
    engine = ChatEngine(
        adapter=adapter,
        system_prompt="system",
        max_tokens=12,
        temperature=0.2,
        top_p=0.8,
    )

    response = engine.ask("hello")

    assert response.startswith("reply-")
    assert len(engine.messages) == 3
    assert engine.messages[0]["role"] == "system"
    assert engine.messages[1]["role"] == "user"
    assert engine.messages[2]["role"] == "assistant"
    assert adapter.last_messages is engine.messages
    assert adapter.last_user_message == "hello"
