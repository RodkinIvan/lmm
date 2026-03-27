import os

from lmm.memory.base import MemoryModule
from lmm.memory.consolidation_lora_ffn import ConsolidationLoraFfnMemoryModule
from lmm.memory.hash_gradient import HashGradientMemoryModule
from lmm.memory.identity import IdentityMemoryModule
from lmm.memory.lora_on_user import LoraOnUserMemoryModule


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def create_memory_module(name: str) -> MemoryModule:
    normalized = name.strip().lower()
    if normalized == "identity":
        return IdentityMemoryModule()
    if normalized in {"lora_on_user", "loraonuser"}:
        return LoraOnUserMemoryModule()
    if normalized in {
        "consolidation_lora_ffn",
        "consolidation_ffn",
        "consolidation",
    }:
        return ConsolidationLoraFfnMemoryModule(
            rank=_env_int("LMM_CONS_RANK", 8),
            init_std=_env_float("LMM_CONS_INIT_STD", 0.02),
            optimizer_name=os.getenv("LMM_CONS_OPT", "adamw"),
            learning_rate=_env_float("LMM_CONS_LR", 1e-3),
            weight_decay=_env_float("LMM_CONS_WEIGHT_DECAY", 0.01),
            l2_regularization=_env_float("LMM_CONS_L2", 0.0),
            consolidation_steps=_env_int("LMM_CONS_STEPS", 12),
            general_batch_size=_env_int("LMM_CONS_GENERAL_BATCH", 2),
            user_batch_size=_env_int("LMM_CONS_USER_BATCH", 2),
            max_tokens_per_sample=_env_int("LMM_CONS_MAX_TOKENS", 512),
            user_alpha=_env_float("LMM_CONS_USER_ALPHA", 0.1),
            consolidate_on_save=_env_bool("LMM_CONS_ON_SAVE", True),
            max_session_examples=_env_int("LMM_CONS_MAX_SESSION_EXAMPLES", 256),
            general_dataset_name=os.getenv(
                "LMM_CONS_DATASET", "HuggingFaceH4/ultrachat_200k"
            ),
            general_split=os.getenv("LMM_CONS_SPLIT", "train_sft"),
            general_shuffle_buffer=_env_int("LMM_CONS_SHUFFLE_BUFFER", 10000),
            general_seed=_env_int("LMM_CONS_SEED", 0),
            general_init_timeout_sec=_env_float("LMM_CONS_GENERAL_INIT_TIMEOUT", 4.0),
            general_next_timeout_sec=_env_float("LMM_CONS_GENERAL_NEXT_TIMEOUT", 3.0),
            use_synthetic_general_fallback=_env_bool(
                "LMM_CONS_SYNTHETIC_GENERAL_FALLBACK", True
            ),
            consolidation_log_path=os.getenv(
                "LMM_CONS_LOG_PATH", "logs/consolidation_training.jsonl"
            ),
            log_max_samples_per_step=_env_int("LMM_CONS_LOG_MAX_SAMPLES", 3),
            log_text_max_chars=_env_int("LMM_CONS_LOG_TEXT_CHARS", 200),
            log_prefix_turns=_env_int("LMM_CONS_LOG_PREFIX_TURNS", 4),
        )
    if normalized in {"hash_gradient", "hashgradient"}:
        return HashGradientMemoryModule()
    raise ValueError(
        f"Unknown memory module '{name}'. "
        "Available modules: identity, lora_on_user, consolidation_lora_ffn, hash_gradient"
    )
