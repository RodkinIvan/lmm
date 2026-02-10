from dataclasses import dataclass


@dataclass(slots=True)
class ChatConfig:
    backend: str = "mlx_lm"
    model_id: str = "google/gemma-3-1b-it"
    memory_module: str = "identity"
    mm_load_path: str = ""
    mm_save_path: str = ""
    activation_log_path: str = "logs/memory_activations.jsonl"
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.95
    deterministic: bool = True
    decoding_strategy: str = "auto"
    local_files_only: bool = False
    system_prompt: str = ""
    verbose: bool = False
