from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from lmm.adapters.base import ModelAdapter
from lmm.hooks import HookManager, hook_last_user_response
from lmm.memory.activation_logger import ActivationLogger
from lmm.memory.base import MemoryContext, MemoryModule


class LayerMemoryHook:
    """Wrap one transformer layer and run memory rewrite on its output."""

    def __init__(
        self,
        *,
        wrapped_layer: Any,
        memory_module: MemoryModule,
        layer_index: int,
        total_layers: int,
        backend_name: str,
        model_id: str,
        activation_logger: ActivationLogger,
    ) -> None:
        self._wrapped_layer = wrapped_layer
        self._memory_module = memory_module
        self._layer_index = layer_index
        self._total_layers = total_layers
        self._backend_name = backend_name
        self._model_id = model_id
        self._activation_logger = activation_logger
        self._call_index = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_layer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        output = self._wrapped_layer(*args, **kwargs)
        hidden_states, rebuild = _extract_hidden_and_rebuild(output)
        context = MemoryContext(
            layer_index=self._layer_index,
            total_layers=self._total_layers,
            call_index=self._call_index,
            backend=self._backend_name,
            model_id=self._model_id,
        )
        self._call_index += 1
        rewritten_hidden_states = self._memory_module.rewrite(hidden_states, context)
        try:
            self._activation_logger.log_activation(
                context=context,
                input_hidden_states=hidden_states,
                output_hidden_states=rewritten_hidden_states,
            )
        except Exception:
            # Logging must never break generation.
            pass
        return rebuild(rewritten_hidden_states)


def _extract_hidden_and_rebuild(
    layer_output: Any,
) -> Tuple[Any, Any]:
    # Most decoder layers return hidden states directly.
    # For tuple/list outputs, first item is hidden states and the tail is preserved.
    if isinstance(layer_output, tuple):
        if not layer_output:
            return layer_output, lambda x: x
        return layer_output[0], lambda hidden: (hidden, *layer_output[1:])
    if isinstance(layer_output, list):
        if not layer_output:
            return layer_output, lambda x: x
        return layer_output[0], lambda hidden: [hidden, *layer_output[1:]]
    return layer_output, lambda hidden: hidden


class MlxLmAdapter(ModelAdapter):
    backend_name = "mlx_lm"

    def __init__(
        self,
        *,
        model_id: str,
        memory_module: MemoryModule,
        local_files_only: bool = False,
        verbose: bool = False,
        activation_log_path: str = "logs/memory_activations.jsonl",
    ) -> None:
        self.model_id = model_id
        self.memory_module = memory_module
        self.local_files_only = local_files_only
        self.verbose = verbose
        self.activation_log_path = activation_log_path
        self._activation_logger = ActivationLogger(activation_log_path)
        self._hook_manager = HookManager()
        self._hook_manager.register("user_message", hook_last_user_response)

        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._sampler_factory: Optional[Any] = None
        self._generate_fn: Optional[Any] = None
        self._hook_info: Optional[Tuple[int, int]] = None

    def register_hook(self, event: str, callback) -> None:
        self._hook_manager.register(event, callback)

    def preload(self) -> None:
        self._ensure_loaded()

    def load_memory(self, path: str) -> dict[str, Any]:
        self._ensure_loaded()
        return self.memory_module.load(path)

    def save_memory(self, path: str) -> dict[str, Any]:
        return self.memory_module.save(path)

    def _call_memory_update(self, stage: str, info: Dict[str, Any]) -> None:
        payload = copy.deepcopy(info)
        payload["stage"] = stage
        try:
            self.memory_module.update(payload)
        except Exception as exc:
            payload["update_error"] = str(exc)
        self._activation_logger.log_update(
            memory_module_name=self.memory_module.name,
            stage=stage,
            info=payload,
        )

    def on_user_message(self, user_text: str, messages: List[Dict[str, str]]) -> None:
        info: Dict[str, Any] = {
            "event": "user_message",
            "user_message": user_text,
            "message_count": len(messages),
            "messages": copy.deepcopy(messages),
            "hooked_fields": [],
            "callbacks": [],
        }
        try:
            self._ensure_loaded()
            info["model_ready"] = True
        except Exception as exc:
            info["model_ready"] = False
            info["model_load_error"] = str(exc)
        self._call_memory_update("pre_hooks", info)
        self._hook_manager.emit("user_message", info)
        self._call_memory_update("post_hooks", info)

    def _resolve_model_reference(self) -> str:
        path = Path(self.model_id)
        if path.exists():
            return str(path)
        if not self.local_files_only:
            return self.model_id

        try:
            from huggingface_hub import snapshot_download
        except Exception as exc:  # pragma: no cover - runtime dependency detail
            raise RuntimeError(
                "local_files_only=True requires huggingface_hub to resolve local cache"
            ) from exc

        try:
            return snapshot_download(self.model_id, local_files_only=True)
        except Exception as exc:
            raise RuntimeError(
                f"Model '{self.model_id}' not found in local Hugging Face cache."
            ) from exc

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            from mlx_lm import generate, load
            from mlx_lm.sample_utils import make_sampler
        except Exception as exc:
            raise RuntimeError(
                "Failed to import mlx_lm. Ensure MLX runtime is healthy in your environment."
            ) from exc

        model_reference = self._resolve_model_reference()
        self._model, self._tokenizer = load(model_reference)
        self._sampler_factory = make_sampler
        self._generate_fn = generate
        self._install_memory_hook()
        self.memory_module.attach_runtime(
            model=self._model,
            tokenizer=self._tokenizer,
            prompt_builder=self._build_prompt,
        )

    def _install_memory_hook(self) -> None:
        if self._model is None:
            raise RuntimeError("Model must be loaded before installing memory hook")

        layers = getattr(self._model, "layers", None)
        if layers is None:
            raise RuntimeError("Loaded model does not expose `layers` for hooking.")

        total_layers = len(layers)
        if total_layers <= 0:
            raise RuntimeError("Loaded model has no transformer layers to hook.")

        target_layer_index = self.memory_module.select_layer_index(total_layers)
        if not (0 <= target_layer_index < total_layers):
            raise RuntimeError(
                f"Memory module selected invalid layer index {target_layer_index} "
                f"for total layers {total_layers}."
            )

        original_layer = layers[target_layer_index]
        layers[target_layer_index] = LayerMemoryHook(
            wrapped_layer=original_layer,
            memory_module=self.memory_module,
            layer_index=target_layer_index,
            total_layers=total_layers,
            backend_name=self.backend_name,
            model_id=self.model_id,
            activation_logger=self._activation_logger,
        )
        self._hook_info = (target_layer_index, total_layers)

    def _build_prompt(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = True,
    ) -> str:
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded.")

        if hasattr(self._tokenizer, "apply_chat_template"):
            return self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            parts.append(f"{role}: {content}")
        if add_generation_prompt:
            parts.append("ASSISTANT:")
        return "\n".join(parts)

    def generate_reply(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        self._ensure_loaded()
        if (
            self._model is None
            or self._tokenizer is None
            or self._sampler_factory is None
            or self._generate_fn is None
        ):
            raise RuntimeError("Model backend failed to initialize.")

        prompt = self._build_prompt(messages, add_generation_prompt=True)
        sampler = self._sampler_factory(temp=temperature, top_p=top_p)
        text = self._generate_fn(
            self._model,
            self._tokenizer,
            prompt,
            max_tokens=max_tokens,
            sampler=sampler,
            verbose=self.verbose,
        )
        return text.strip()

    @property
    def hook_info(self) -> Optional[Tuple[int, int]]:
        return self._hook_info
