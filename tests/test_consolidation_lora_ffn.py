import json

from lmm.memory.consolidation_lora_ffn import ConsolidationLoraFfnMemoryModule
from lmm.memory.registry import create_memory_module


def test_consolidation_module_collects_session_examples_on_update():
    module = ConsolidationLoraFfnMemoryModule()
    info = {
        "stage": "post_hooks",
        "event": "user_message",
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "My name is Alice."},
        ],
    }
    module.update(info)
    assert info["update_status"] == "ok"
    assert info["consolidation_mode"] == "collect_only"
    assert info["consolidation_pending_examples"] == 1


def test_consolidation_module_is_registered():
    module = create_memory_module("consolidation_lora_ffn")
    assert isinstance(module, ConsolidationLoraFfnMemoryModule)


def test_consolidation_save_without_runtime_is_safe():
    module = ConsolidationLoraFfnMemoryModule()
    module.update(
        {
            "stage": "post_hooks",
            "event": "user_message",
            "messages": [{"role": "user", "content": "I am Bob."}],
        }
    )
    result = module.save("artifacts/non_initialized_consolidation_test.safetensors")
    assert result["saved"] is False
    assert "consolidation" in result


def test_prepare_sample_tensors_allows_empty_prefix():
    class _Tokenizer:
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            _ = add_special_tokens
            return [ord(c) % 127 for c in text]

    module = ConsolidationLoraFfnMemoryModule()
    module._tokenizer = _Tokenizer()
    module._prompt_builder = lambda messages, add_generation_prompt: " ".join(
        m.get("content", "") for m in messages
    )

    sample = module._prepare_sample_tensors(
        prefix_messages=[],
        target_role="user",
        target_text="I am Alice.",
    )
    assert sample is not None


def test_general_batch_size_allows_zero():
    module = ConsolidationLoraFfnMemoryModule(general_batch_size=0)
    assert module.general_batch_size == 0


def test_consolidation_step_logging_writes_text_preview(tmp_path):
    log_path = tmp_path / "consolidation_training.jsonl"
    module = ConsolidationLoraFfnMemoryModule(
        consolidation_log_path=str(log_path),
        log_max_samples_per_step=2,
        log_text_max_chars=40,
    )

    sample = {
        "prefix_messages": [
            {"role": "system", "content": "You are concise."},
            {"role": "user", "content": "Remember my city is Barcelona."},
        ],
        "target_role": "assistant",
        "target_text": "Got it, your city is Barcelona.",
    }
    preview = module._sample_record_preview(sample)
    assert "user:" in preview["prefix_text"]
    assert "Barcelona" in preview["target_text"]

    payload = {
        "kind": "consolidation_step",
        "run_id": "unit-test",
        "step": 1,
        "total_loss": 1.23,
        "general_batch_preview": [preview],
        "user_batch_preview": [preview],
    }
    module._append_consolidation_log(payload)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["kind"] == "consolidation_step"
    assert parsed["general_batch_preview"][0]["target_text"].startswith("Got it")


def test_fallback_general_pool_produces_samples_with_source():
    class _Tokenizer:
        bos_token = None

        def encode(self, text, add_special_tokens=True):
            _ = add_special_tokens
            return [ord(c) % 127 for c in text]

    module = ConsolidationLoraFfnMemoryModule()
    module._tokenizer = _Tokenizer()
    module._prompt_builder = lambda messages, add_generation_prompt: " ".join(
        m.get("content", "") for m in messages
    )

    samples = module._fallback_general_pool(2)
    assert len(samples) == 2
    assert all(s.get("source") == "synthetic_fallback" for s in samples)
