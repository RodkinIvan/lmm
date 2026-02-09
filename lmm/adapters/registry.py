from lmm.adapters.base import ModelAdapter
from lmm.adapters.mlx_lm_adapter import MlxLmAdapter
from lmm.memory.base import MemoryModule


def create_adapter(
    *,
    backend: str,
    model_id: str,
    memory_module: MemoryModule,
    local_files_only: bool,
    verbose: bool,
    activation_log_path: str,
) -> ModelAdapter:
    normalized = backend.strip().lower()
    if normalized in {"mlx", "mlx_lm", "mlxlm"}:
        return MlxLmAdapter(
            model_id=model_id,
            memory_module=memory_module,
            local_files_only=local_files_only,
            verbose=verbose,
            activation_log_path=activation_log_path,
        )
    raise ValueError(f"Unknown backend '{backend}'. Available backends: mlx_lm")
