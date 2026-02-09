import json

from lmm.adapters.mlx_lm_adapter import LayerMemoryHook, MlxLmAdapter
from lmm.memory.activation_logger import ActivationLogger
from lmm.memory.base import MemoryContext, MemoryModule
from lmm.memory.identity import IdentityMemoryModule
from lmm.memory.lora_on_user import LoraOnUserMemoryModule


class RecordingMemoryModule(MemoryModule):
    name = "recording"

    def __init__(self) -> None:
        self.calls = []

    def rewrite(self, hidden_states, context: MemoryContext):
        self.calls.append((hidden_states, context))
        return f"{hidden_states}|rewritten"


def test_identity_memory_module_is_noop():
    module = IdentityMemoryModule()
    context = MemoryContext(
        layer_index=4,
        total_layers=8,
        call_index=0,
        backend="mlx_lm",
        model_id="dummy",
    )
    hidden = {"k": "v"}
    assert module.rewrite(hidden, context) is hidden


def test_memory_hook_rewrites_direct_output(tmp_path):
    class FakeLayer:
        def __call__(self, x):
            return f"{x}|layer"

    module = RecordingMemoryModule()
    hook = LayerMemoryHook(
        wrapped_layer=FakeLayer(),
        memory_module=module,
        layer_index=3,
        total_layers=6,
        backend_name="mlx_lm",
        model_id="model",
        activation_logger=ActivationLogger(str(tmp_path / "direct.jsonl")),
    )

    out = hook("h")
    assert out == "h|layer|rewritten"
    assert len(module.calls) == 1
    assert module.calls[0][1].layer_index == 3
    assert module.calls[0][1].total_layers == 6
    assert module.calls[0][1].call_index == 0


def test_memory_hook_rewrites_first_tuple_item_only(tmp_path):
    class FakeLayer:
        def __call__(self, x):
            return (f"{x}|layer", "cache")

    module = RecordingMemoryModule()
    hook = LayerMemoryHook(
        wrapped_layer=FakeLayer(),
        memory_module=module,
        layer_index=2,
        total_layers=5,
        backend_name="mlx_lm",
        model_id="model",
        activation_logger=ActivationLogger(str(tmp_path / "tuple.jsonl")),
    )

    out = hook("h")
    assert out == ("h|layer|rewritten", "cache")


def test_default_layer_selection_is_middle():
    module = IdentityMemoryModule()
    assert module.select_layer_index(1) == 0
    assert module.select_layer_index(5) == 2
    assert module.select_layer_index(10) == 5


class FakeHiddenState:
    def __init__(self, value):
        self.value = value

    def __getitem__(self, key):
        if key == (slice(None, None, None), -1, slice(-5, None, None)):
            return self
        raise KeyError(key)

    def tolist(self):
        return self.value


def test_activation_logger_writes_slices(tmp_path):
    path = tmp_path / "memory_activations.jsonl"
    logger = ActivationLogger(str(path))
    context = MemoryContext(
        layer_index=4,
        total_layers=8,
        call_index=1,
        backend="mlx_lm",
        model_id="model",
    )
    logger.log_activation(
        context=context,
        input_hidden_states=FakeHiddenState([[1, 2, 3, 4, 5]]),
        output_hidden_states=FakeHiddenState([[6, 7, 8, 9, 10]]),
    )

    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["context"]["layer_index"] == 4
    assert payload["input_slice"] == [[1, 2, 3, 4, 5]]
    assert payload["output_slice"] == [[6, 7, 8, 9, 10]]
    assert payload["kind"] == "activation"


class UpdateRecordingMemoryModule(MemoryModule):
    name = "update_recorder"

    def __init__(self):
        self.updates = []

    def rewrite(self, hidden_states, context: MemoryContext):
        _ = context
        return hidden_states

    def update(self, info):
        self.updates.append(info)


def test_user_message_hooks_update_and_log(tmp_path):
    log_path = tmp_path / "memory_events.jsonl"
    module = UpdateRecordingMemoryModule()
    adapter = MlxLmAdapter(
        model_id="dummy/model",
        memory_module=module,
        activation_log_path=str(log_path),
    )
    messages = [{"role": "user", "content": "hello"}]
    adapter.on_user_message("hello", messages)

    assert len(module.updates) == 2
    pre_update, post_update = module.updates
    assert pre_update["stage"] == "pre_hooks"
    assert pre_update["user_message"] == "hello"
    assert "last_user_response" not in pre_update

    assert post_update["stage"] == "post_hooks"
    assert post_update["user_message"] == "hello"
    assert post_update["last_user_response"] == "hello"
    assert "last_user_response" in post_update["hooked_fields"]
    assert "hook_last_user_response" in post_update["callbacks"]

    entries = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").strip().splitlines()
    ]
    update_entries = [e for e in entries if e.get("kind") == "update"]
    assert len(update_entries) == 2
    assert update_entries[0]["stage"] == "pre_hooks"
    assert update_entries[1]["stage"] == "post_hooks"
    assert update_entries[1]["info"]["last_user_response"] == "hello"


def test_lora_on_user_module_is_identity_and_tracks_updates():
    module = LoraOnUserMemoryModule()
    context = MemoryContext(
        layer_index=1,
        total_layers=3,
        call_index=0,
        backend="mlx_lm",
        model_id="model",
    )
    hidden = {"h": 1}
    assert module.rewrite(hidden, context) is hidden

    module.update({"last_user_response": "hi"})
    assert module.info["last_user_response"] == "hi"
    assert module.update_history[-1]["last_user_response"] == "hi"


def test_lora_on_user_token_loss_dict_uses_decoded_tokens_with_suffixes():
    class FakeTokenizer:
        def decode(self, token_ids):
            if token_ids == [1]:
                return "a"
            if token_ids == [2]:
                return "a"
            if token_ids == [3]:
                return "b"
            return "?"

    module = LoraOnUserMemoryModule()
    module._tokenizer = FakeTokenizer()
    result = module._token_loss_dict([1, 2, 3], [0.1, 0.2, 0.3])
    assert result == {"a": 0.1, "a#2": 0.2, "b": 0.3}


def test_lora_on_user_excludes_scaffold_tokens_from_loss():
    class FakeTokenizer:
        all_special_ids = [101, 102]

    module = LoraOnUserMemoryModule()
    module._tokenizer = FakeTokenizer()
    assert module._is_excluded_loss_token(101, "<start_of_turn>")
    assert module._is_excluded_loss_token(42, "user")
    assert module._is_excluded_loss_token(42, "\n")
    assert not module._is_excluded_loss_token(42, "Ivan")


def test_lora_on_user_selects_only_losses_above_average():
    module = LoraOnUserMemoryModule()
    ids, losses, avg, fallback = module._select_high_ce_tokens(
        [10, 11, 12, 13],
        [1.0, 3.0, 2.0, 6.0],
    )
    assert avg == 3.0
    assert ids == [13]
    assert losses == [6.0]
    assert not fallback
