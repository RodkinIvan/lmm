from lmm.memory.base import MemoryModule
from lmm.memory.identity import IdentityMemoryModule
from lmm.memory.lora_on_user import LoraOnUserMemoryModule


def create_memory_module(name: str) -> MemoryModule:
    normalized = name.strip().lower()
    if normalized == "identity":
        return IdentityMemoryModule()
    if normalized in {"lora_on_user", "loraonuser"}:
        return LoraOnUserMemoryModule()
    raise ValueError(
        f"Unknown memory module '{name}'. "
        "Available modules: identity, lora_on_user"
    )
