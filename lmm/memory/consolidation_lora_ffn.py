from __future__ import annotations

import copy
import json
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, cast
from uuid import uuid4

from lmm.memory.base import MemoryContext, MemoryModule


def _extract_layer_input(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if len(args) > 0:
        return args[0]
    if "x" in kwargs:
        return kwargs["x"]
    if "hidden_states" in kwargs:
        return kwargs["hidden_states"]
    return None


class _LinearLoraWrapper:
    def __init__(
        self,
        *,
        wrapped_layer: Any,
        module: "ConsolidationLoraFfnMemoryModule",
        target_key: str,
    ) -> None:
        self._wrapped_layer = wrapped_layer
        self._module = module
        self._target_key = target_key

    def __getattr__(self, name: str) -> Any:
        return getattr(self._wrapped_layer, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        layer_input = _extract_layer_input(args, kwargs)
        output = self._wrapped_layer(*args, **kwargs)

        if layer_input is None:
            return output
        if not hasattr(layer_input, "shape") or not hasattr(output, "shape"):
            return output

        A, B = self._module._get_target_ab(self._target_key, layer_input, output)
        if A is None or B is None:
            return output

        delta = (layer_input @ A.astype(layer_input.dtype)) @ B.astype(
            layer_input.dtype
        )
        return output + delta.astype(output.dtype)


class ConsolidationLoraFfnMemoryModule(MemoryModule):
    """
    Consolidation LoRA attached to FFN linear projections.

    - During a session: collect user targets, do NOT optimize.
    - On save (session end): run K consolidation steps combining:
      * general-chat loss from streamed dataset samples
      * session-user loss from just-ended dialogue
    - LoRA is injected into FFN projection modules (e.g. gate/up/down), not as
      a post-FFN hidden-state rewrite.
    """

    name = "consolidation_lora_ffn"

    def __init__(
        self,
        *,
        rank: int = 8,
        init_std: float = 0.02,
        optimizer_name: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        l2_regularization: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        consolidation_steps: int = 12,
        general_batch_size: int = 2,
        user_batch_size: int = 2,
        max_tokens_per_sample: int = 512,
        user_alpha: float = 0.1,
        consolidate_on_save: bool = True,
        max_session_examples: int = 256,
        general_dataset_name: str = "HuggingFaceH4/ultrachat_200k",
        general_split: str = "train_sft",
        general_shuffle_buffer: int = 10000,
        general_seed: int = 0,
        general_init_timeout_sec: float = 4.0,
        general_next_timeout_sec: float = 3.0,
        use_synthetic_general_fallback: bool = True,
        consolidation_log_path: str = "logs/consolidation_training.jsonl",
        log_max_samples_per_step: int = 3,
        log_text_max_chars: int = 200,
        log_prefix_turns: int = 4,
    ) -> None:
        self.rank = max(1, int(rank))
        self.init_std = float(init_std)
        self.optimizer_name = optimizer_name.strip().lower()
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.l2_regularization = max(0.0, float(l2_regularization))
        self.betas = betas
        self.eps = float(eps)

        self.consolidation_steps = max(1, int(consolidation_steps))
        self.general_batch_size = max(0, int(general_batch_size))
        self.user_batch_size = max(1, int(user_batch_size))
        self.max_tokens_per_sample = max(8, int(max_tokens_per_sample))
        self.user_alpha = min(1.0, max(0.0, float(user_alpha)))
        self.consolidate_on_save = bool(consolidate_on_save)
        self.max_session_examples = max(1, int(max_session_examples))

        self.general_dataset_name = general_dataset_name
        self.general_split = general_split
        self.general_shuffle_buffer = max(100, int(general_shuffle_buffer))
        self.general_seed = int(general_seed)
        self.general_init_timeout_sec = max(0.0, float(general_init_timeout_sec))
        self.general_next_timeout_sec = max(0.0, float(general_next_timeout_sec))
        self.use_synthetic_general_fallback = bool(use_synthetic_general_fallback)
        self.consolidation_log_path = consolidation_log_path
        self.log_max_samples_per_step = max(1, int(log_max_samples_per_step))
        self.log_text_max_chars = max(32, int(log_text_max_chars))
        self.log_prefix_turns = max(1, int(log_prefix_turns))

        self.A: Optional[Dict[str, Any]] = None
        self.B: Optional[Dict[str, Any]] = None
        self._target_dims: Dict[str, Tuple[int, int]] = {}
        self._active_params = None
        self._optimizer = None

        self._model = None
        self._tokenizer = None
        self._prompt_builder: Optional[Callable[[list[dict[str, str]], bool], str]] = (
            None
        )

        self._ffn_wrapped = False
        self._num_ffn_layers = 0
        self._num_ffn_targets = 0
        self._target_keys: List[str] = []

        self._general_stream = None
        self._general_iter = None
        self._general_stream_seed = self.general_seed
        self._general_disabled_reason: Optional[str] = None

        self._session_examples: List[Dict[str, Any]] = []

        self.last_consolidation: Dict[str, Any] = {}
        self.info: Dict[str, Any] = {}
        self.update_history: List[Dict[str, Any]] = []

    def _import_mlx(self):
        import mlx.core as mx
        import mlx.nn as nn

        return mx, nn

    def rewrite(self, hidden_states: Any, context: MemoryContext) -> Any:
        _ = context
        return hidden_states

    def attach_runtime(
        self,
        *,
        model: Any,
        tokenizer: Any,
        prompt_builder: Callable[[list[dict[str, str]], bool], str],
    ) -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._prompt_builder = prompt_builder
        self._install_ffn_wrappers()

    def _iter_ffn_blocks(self) -> List[Tuple[int, str, Any]]:
        if self._model is None:
            return []
        layers = getattr(self._model, "layers", None)
        if layers is None:
            return []

        candidates = ("mlp", "ffn", "feed_forward", "feedforward")
        out: List[Tuple[int, str, Any]] = []
        for layer_idx, layer in enumerate(layers):
            for attr in candidates:
                if not hasattr(layer, attr):
                    continue
                sub = getattr(layer, attr)
                if callable(sub):
                    out.append((layer_idx, attr, sub))
                    break
        return out

    def _iter_ffn_linear_targets(self) -> List[Tuple[Any, str, Any, str]]:
        projection_candidates = (
            "gate_proj",
            "up_proj",
            "down_proj",
            "w1",
            "w2",
            "w3",
            "fc1",
            "fc2",
            "linear1",
            "linear2",
            "dense_h_to_4h",
            "dense_4h_to_h",
            "proj",
        )
        out: List[Tuple[Any, str, Any, str]] = []

        for layer_idx, ffn_attr, ffn_module in self._iter_ffn_blocks():
            seen: set[str] = set()
            for proj_attr in projection_candidates:
                if not hasattr(ffn_module, proj_attr):
                    continue
                sub = getattr(ffn_module, proj_attr)
                if not callable(sub):
                    continue
                target_key = f"L{layer_idx}.{ffn_attr}.{proj_attr}"
                out.append((ffn_module, proj_attr, sub, target_key))
                seen.add(proj_attr)

            if len(seen) > 0:
                continue

            # Fallback for unknown model naming: pick callable children with weight.
            for attr in sorted(dir(ffn_module)):
                if attr.startswith("_") or attr in seen:
                    continue
                try:
                    sub = getattr(ffn_module, attr)
                except Exception:
                    continue
                if not callable(sub):
                    continue
                if getattr(sub, "weight", None) is None:
                    continue
                target_key = f"L{layer_idx}.{ffn_attr}.{attr}"
                out.append((ffn_module, attr, sub, target_key))
        return out

    def _install_ffn_wrappers(self) -> None:
        ffn_blocks = self._iter_ffn_blocks()
        self._num_ffn_layers = len(ffn_blocks)

        targets = self._iter_ffn_linear_targets()
        self._num_ffn_targets = len(targets)
        self._target_keys = [target_key for _, _, _, target_key in targets]
        if self._num_ffn_targets == 0:
            return

        if self._ffn_wrapped:
            return

        for ffn_module, attr, sub, target_key in targets:
            if isinstance(sub, _LinearLoraWrapper):
                continue
            setattr(
                ffn_module,
                attr,
                _LinearLoraWrapper(
                    wrapped_layer=sub,
                    module=self,
                    target_key=target_key,
                ),
            )
        self._ffn_wrapped = True

    def _ensure_param_maps(self) -> None:
        if self.A is None:
            self.A = {}
        if self.B is None:
            self.B = {}

    def _ensure_target_matrices(
        self, target_key: str, in_dim: int, out_dim: int
    ) -> None:
        if in_dim <= 0 or out_dim <= 0:
            return
        # Do not mutate params while running under autograd with local params.
        if self._active_params is not None:
            return

        self._ensure_param_maps()
        if self.A is None or self.B is None:
            return

        current_a = self.A.get(target_key)
        current_b = self.B.get(target_key)
        if (
            current_a is not None
            and current_b is not None
            and tuple(current_a.shape) == (in_dim, self.rank)
            and tuple(current_b.shape) == (self.rank, out_dim)
        ):
            self._target_dims[target_key] = (in_dim, out_dim)
            return

        mx, _ = self._import_mlx()
        self.A[target_key] = (
            self.init_std * mx.random.normal((in_dim, self.rank), dtype=mx.float32)
        ).astype(mx.float32)
        self.B[target_key] = mx.zeros((self.rank, out_dim), dtype=mx.float32)
        self._target_dims[target_key] = (in_dim, out_dim)

    def _ensure_optimizer(self) -> None:
        if self._optimizer is not None:
            return
        import mlx.optimizers as optim

        if self.optimizer_name == "adamw":
            self._optimizer = optim.AdamW(
                learning_rate=self.learning_rate,
                betas=list(self.betas),
                eps=self.eps,
                weight_decay=self.weight_decay,
            )
            return
        if self.optimizer_name == "adam":
            self._optimizer = optim.Adam(
                learning_rate=self.learning_rate,
                betas=list(self.betas),
                eps=self.eps,
            )
            return
        if self.optimizer_name == "sgd":
            self._optimizer = optim.SGD(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            return
        raise ValueError(
            f"Unsupported optimizer '{self.optimizer_name}'. Supported: adamw, adam, sgd"
        )

    def _current_params(self) -> Dict[str, Any]:
        if self._active_params is not None:
            return self._active_params
        self._ensure_param_maps()
        return {
            "A": self.A if self.A is not None else {},
            "B": self.B if self.B is not None else {},
        }

    def _get_target_ab(
        self,
        target_key: str,
        layer_input: Any,
        layer_output: Any,
    ) -> Tuple[Any, Any]:
        input_shape = getattr(layer_input, "shape", None)
        output_shape = getattr(layer_output, "shape", None)
        if not input_shape or not output_shape:
            return None, None

        in_dim = int(input_shape[-1])
        out_dim = int(output_shape[-1])
        if in_dim <= 0 or out_dim <= 0:
            return None, None

        if self._active_params is None:
            self._ensure_target_matrices(target_key, in_dim, out_dim)

        params = self._current_params()
        a_map = params.get("A", {})
        b_map = params.get("B", {})
        A = a_map.get(target_key)
        B = b_map.get(target_key)
        if A is None or B is None:
            return None, None
        if tuple(A.shape) != (in_dim, self.rank) or tuple(B.shape) != (
            self.rank,
            out_dim,
        ):
            if self._active_params is not None:
                return None, None
            self._ensure_target_matrices(target_key, in_dim, out_dim)
            params = self._current_params()
            a_map = params.get("A", {})
            b_map = params.get("B", {})
            A = a_map.get(target_key)
            B = b_map.get(target_key)
        return A, B

    def _encode_prompt(self, text: str) -> List[int]:
        if self._tokenizer is None:
            return []
        bos_token = getattr(self._tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not text.startswith(bos_token)
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def _prepare_sample_tensors(
        self,
        *,
        prefix_messages: List[Dict[str, str]],
        target_role: str,
        target_text: str,
    ) -> Optional[Tuple[Any, Any, Any]]:
        mx, _ = self._import_mlx()
        if self._prompt_builder is None:
            return None
        if not target_text.strip():
            return None

        full_messages = [
            *prefix_messages,
            {"role": target_role, "content": target_text},
        ]
        full_prompt = self._prompt_builder(full_messages, False)
        if prefix_messages:
            prefix_prompt = self._prompt_builder(prefix_messages, False)
        else:
            prefix_prompt = ""

        full_tokens = self._encode_prompt(full_prompt)
        prefix_tokens = self._encode_prompt(prefix_prompt)
        target_start = min(len(prefix_tokens), len(full_tokens))
        if target_start >= len(full_tokens):
            return None

        if len(full_tokens) > self.max_tokens_per_sample:
            offset = len(full_tokens) - self.max_tokens_per_sample
            full_tokens = full_tokens[offset:]
            target_start = max(0, target_start - offset)
            if target_start >= len(full_tokens):
                return None

        if len(full_tokens) < 2:
            return None

        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]
        mask_start = max(0, target_start - 1)
        mask = [1.0 if i >= mask_start else 0.0 for i in range(len(target_tokens))]
        if sum(mask) <= 0.0:
            return None

        return (
            mx.array(input_tokens, dtype=mx.uint32),
            mx.array(target_tokens, dtype=mx.uint32),
            mx.array(mask, dtype=mx.float32),
        )

    def _make_sample_record(
        self,
        *,
        prefix_messages: List[Dict[str, str]],
        target_role: str,
        target_text: str,
    ) -> Optional[Dict[str, Any]]:
        tensors = self._prepare_sample_tensors(
            prefix_messages=prefix_messages,
            target_role=target_role,
            target_text=target_text,
        )
        if tensors is None:
            return None
        return {
            "tensors": tensors,
            "prefix_messages": copy.deepcopy(prefix_messages),
            "target_role": str(target_role),
            "target_text": str(target_text),
        }

    def _clean_text(self, text: str) -> str:
        cleaned = " ".join(str(text).split())
        if len(cleaned) <= self.log_text_max_chars:
            return cleaned
        return cleaned[: self.log_text_max_chars - 3] + "..."

    def _messages_preview(self, messages: List[Dict[str, str]]) -> str:
        if not messages:
            return ""
        tail = messages[-self.log_prefix_turns :]
        parts: List[str] = []
        for message in tail:
            role = str(message.get("role", "user")).strip().lower() or "user"
            content = self._clean_text(str(message.get("content", "")))
            if content:
                parts.append(f"{role}: {content}")
        return " || ".join(parts)

    def _sample_record_preview(self, record: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source": str(record.get("source", "unknown")),
            "prefix_text": self._messages_preview(record.get("prefix_messages", [])),
            "target_role": str(record.get("target_role", "")),
            "target_text": self._clean_text(str(record.get("target_text", ""))),
        }

    def _append_consolidation_log(self, payload: Dict[str, Any]) -> None:
        path_text = str(self.consolidation_log_path).strip()
        if not path_text:
            return
        try:
            out = Path(path_text)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
        except Exception:
            # Logging should never break training or save flow.
            pass

    def _loss_on_sample(self, sample_tensors: Tuple[Any, Any, Any]) -> Any:
        mx, nn = self._import_mlx()
        if self._model is None:
            raise RuntimeError("Model runtime is not attached.")

        input_tokens, target_tokens, mask = sample_tensors
        logits = self._model(input_tokens[None])[0]
        token_losses = nn.losses.cross_entropy(
            logits.astype(mx.float32),
            target_tokens,
            reduction="none",
        )
        weights = mask.astype(token_losses.dtype)
        denom = mx.maximum(mx.sum(weights), mx.array(1.0, dtype=weights.dtype))
        return mx.sum(token_losses * weights) / denom

    def _build_session_pool(self) -> List[Dict[str, Any]]:
        pool: List[Dict[str, Any]] = []
        for item in self._session_examples:
            sample = self._make_sample_record(
                prefix_messages=item["prefix_messages"],
                target_role=item["target_role"],
                target_text=item["target_text"],
            )
            if sample is not None:
                pool.append(sample)
        return pool

    def _call_with_timeout(
        self,
        fn: Callable[[], Any],
        timeout_sec: float,
    ) -> Tuple[str, Any]:
        if timeout_sec <= 0.0:
            try:
                return "ok", fn()
            except Exception as exc:
                return "error", exc

        queue: Queue = Queue(maxsize=1)

        def runner() -> None:
            try:
                queue.put(("ok", fn()))
            except Exception as exc:
                queue.put(("error", exc))

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        thread.join(timeout=timeout_sec)
        if thread.is_alive():
            return "timeout", TimeoutError(
                f"operation timed out after {timeout_sec:.1f}s"
            )
        if queue.empty():
            return "error", RuntimeError("timeout worker produced no result")
        status, value = queue.get()
        return str(status), value

    def _fallback_general_pool(self, sample_count: int) -> List[Dict[str, Any]]:
        templates = [
            (
                "How do I reverse a Python list?",
                "Use list.reverse() in-place or list[::-1] to return a reversed copy.",
            ),
            (
                "Give one sentence about HTTP status 404.",
                "HTTP 404 means the requested resource could not be found on the server.",
            ),
            (
                "What is gradient descent?",
                "Gradient descent is an optimization method that updates parameters in the direction of lower loss.",
            ),
            (
                "What does JSON stand for?",
                "JSON stands for JavaScript Object Notation.",
            ),
            (
                "Explain unit testing briefly.",
                "Unit testing checks small, isolated pieces of code to catch regressions early.",
            ),
        ]

        out: List[Dict[str, Any]] = []
        for _ in range(max(0, sample_count)):
            prompt, response = random.choice(templates)
            sample = self._make_sample_record(
                prefix_messages=[{"role": "user", "content": prompt}],
                target_role="assistant",
                target_text=response,
            )
            if sample is not None:
                sample["source"] = "synthetic_fallback"
                out.append(sample)
        return out

    def _init_general_stream(self) -> None:
        if self._general_disabled_reason is not None:
            return

        try:
            from datasets import load_dataset
        except Exception as exc:
            self._general_disabled_reason = f"datasets-import-error: {exc}"
            return

        def build_stream() -> Any:
            stream = load_dataset(
                self.general_dataset_name,
                split=self.general_split,
                streaming=True,
            )
            seed = self._general_stream_seed
            self._general_stream_seed += 1
            return stream.shuffle(buffer_size=self.general_shuffle_buffer, seed=seed)

        status, value = self._call_with_timeout(
            build_stream, self.general_init_timeout_sec
        )
        if status == "ok":
            self._general_stream = value
            self._general_iter = iter(value)
            return

        self._general_disabled_reason = f"general-stream-init-{status}: {value}"
        self._general_stream = None
        self._general_iter = None

    def _extract_general_example(
        self, row: Dict[str, Any]
    ) -> Optional[Tuple[List[Dict[str, str]], str, str]]:
        messages = row.get("messages")
        if isinstance(messages, list):
            parsed: List[Dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                role = str(message.get("role", message.get("from", ""))).strip().lower()
                content = str(
                    message.get(
                        "content", message.get("value", message.get("text", ""))
                    )
                ).strip()
                if role not in {"user", "assistant", "system"}:
                    continue
                if not content:
                    continue
                parsed.append({"role": role, "content": content})

            assistant_indices = [
                i
                for i, msg in enumerate(parsed)
                if msg.get("role") == "assistant" and i > 0
            ]
            if assistant_indices:
                idx = random.choice(assistant_indices)
                return parsed[:idx], "assistant", parsed[idx]["content"]

        prompt = str(row.get("prompt", "")).strip()
        response = str(row.get("response", "")).strip()
        if prompt and response:
            return ([{"role": "user", "content": prompt}], "assistant", response)

        instruction = str(row.get("instruction", "")).strip()
        output = str(row.get("output", row.get("answer", ""))).strip()
        if instruction and output:
            return ([{"role": "user", "content": instruction}], "assistant", output)

        return None

    def _sample_general_pool(self, sample_count: int) -> List[Dict[str, Any]]:
        if sample_count <= 0:
            return []

        out: List[Dict[str, Any]] = []
        max_attempts = max(8 * sample_count, 32)
        attempts = 0
        start = time.monotonic()

        while len(out) < sample_count and attempts < max_attempts:
            attempts += 1
            if self.general_next_timeout_sec > 0.0:
                if (time.monotonic() - start) > (self.general_next_timeout_sec * 2.0):
                    self._general_disabled_reason = (
                        "general-sampling-time-budget-exceeded"
                    )
                    break

            if self._general_iter is None:
                self._init_general_stream()
            if self._general_iter is None:
                break
            iterator = self._general_iter
            if iterator is None:
                break
            iter_non_none = cast(Iterator[Any], iterator)
            try:
                status, value = self._call_with_timeout(
                    lambda: next(iter_non_none),
                    self.general_next_timeout_sec,
                )
                if status != "ok":
                    self._general_disabled_reason = (
                        f"general-stream-next-{status}: {value}"
                    )
                    self._general_iter = None
                    break
                row = value
            except StopIteration:
                self._init_general_stream()
                iterator = self._general_iter
                if iterator is None:
                    break
                iter_non_none = cast(Iterator[Any], iterator)
                status, value = self._call_with_timeout(
                    lambda: next(iter_non_none),
                    self.general_next_timeout_sec,
                )
                if status != "ok":
                    self._general_disabled_reason = (
                        f"general-stream-next-{status}: {value}"
                    )
                    self._general_iter = None
                    break
                row = value
            except Exception:
                self._general_iter = None
                continue

            if not isinstance(row, dict):
                continue
            example = self._extract_general_example(row)
            if example is None:
                continue

            prefix_messages, target_role, target_text = example
            sample = self._make_sample_record(
                prefix_messages=prefix_messages,
                target_role=target_role,
                target_text=target_text,
            )
            if sample is not None:
                sample["source"] = "dataset"
                out.append(sample)

        if len(out) < sample_count and self.use_synthetic_general_fallback:
            out.extend(self._fallback_general_pool(sample_count - len(out)))

        return out

    def _sample_from_pool(
        self,
        pool: Sequence[Dict[str, Any]],
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        if len(pool) == 0:
            return []
        k = max(1, int(batch_size))
        if len(pool) >= k:
            indices = random.sample(range(len(pool)), k=k)
            return [pool[i] for i in indices]
        return [random.choice(pool) for _ in range(k)]

    def _l2_penalty(self, local_params: Dict[str, Any]) -> Any:
        mx, _ = self._import_mlx()
        terms: List[Any] = []
        for value in local_params.get("A", {}).values():
            terms.append(mx.sum(value * value))
        for value in local_params.get("B", {}).values():
            terms.append(mx.sum(value * value))
        if len(terms) == 0:
            return mx.array(0.0, dtype=mx.float32)
        return self.l2_regularization * mx.sum(mx.stack(terms, axis=0))

    def _prime_matrices_from_sample(self, sample_record: Dict[str, Any]) -> None:
        mx, _ = self._import_mlx()
        probe_loss = self._loss_on_sample(sample_record["tensors"])
        eval_items = [probe_loss]
        if self.A is not None:
            eval_items.extend(self.A.values())
        if self.B is not None:
            eval_items.extend(self.B.values())
        mx.eval(*eval_items)

    def _run_consolidation(self) -> Dict[str, Any]:
        mx, _ = self._import_mlx()
        if self._model is None:
            return {"consolidated": False, "reason": "runtime-not-attached"}
        if self._prompt_builder is None or self._tokenizer is None:
            return {
                "consolidated": False,
                "reason": "runtime-tokenizer-or-template-missing",
            }

        if self._num_ffn_targets <= 0:
            self._install_ffn_wrappers()
        if self._num_ffn_targets <= 0:
            return {"consolidated": False, "reason": "no-ffn-linear-targets-found"}

        session_pool = self._build_session_pool()
        if len(session_pool) == 0:
            return {"consolidated": False, "reason": "no-session-examples"}

        # Lazily initialize LoRA matrices from one real forward pass before autograd.
        try:
            self._prime_matrices_from_sample(session_pool[0])
        except Exception as exc:
            return {
                "consolidated": False,
                "reason": f"failed-to-prime-lora-targets: {exc}",
            }

        self._ensure_param_maps()
        if self.A is None or self.B is None or len(self.A) == 0 or len(self.B) == 0:
            return {"consolidated": False, "reason": "lora-targets-not-initialized"}

        self._ensure_optimizer()
        if self._optimizer is None:
            return {"consolidated": False, "reason": "optimizer-not-initialized"}

        run_id = uuid4().hex
        self._append_consolidation_log(
            {
                "kind": "consolidation_start",
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "memory_module": self.name,
                "consolidation_steps": self.consolidation_steps,
                "general_batch_size": self.general_batch_size,
                "user_batch_size": self.user_batch_size,
                "user_alpha": self.user_alpha,
                "session_examples": len(session_pool),
                "ffn_linear_targets": self._num_ffn_targets,
            }
        )

        step_logs: List[Dict[str, Any]] = []
        for step in range(self.consolidation_steps):
            general_samples = self._sample_general_pool(self.general_batch_size)
            user_samples = self._sample_from_pool(session_pool, self.user_batch_size)

            params = {
                "A": self.A,
                "B": self.B,
            }
            step_metrics: Dict[str, Any] = {}

            def loss_fn(local_params):
                self._active_params = local_params
                try:
                    general_losses: List[Any] = [
                        self._loss_on_sample(sample["tensors"])
                        for sample in general_samples
                    ]
                    user_losses: List[Any] = [
                        self._loss_on_sample(sample["tensors"])
                        for sample in user_samples
                    ]
                finally:
                    self._active_params = None

                if general_losses:
                    g_loss = mx.mean(mx.stack(general_losses, axis=0))
                else:
                    g_loss = mx.array(0.0, dtype=mx.float32)
                if user_losses:
                    u_loss = mx.mean(mx.stack(user_losses, axis=0))
                else:
                    u_loss = mx.array(0.0, dtype=mx.float32)

                if general_losses and user_losses:
                    total = ((1.0 - self.user_alpha) * g_loss) + (
                        self.user_alpha * u_loss
                    )
                elif user_losses:
                    total = u_loss
                else:
                    total = g_loss

                if self.l2_regularization > 0.0:
                    total = total + self._l2_penalty(local_params)

                step_metrics["general_losses"] = general_losses
                step_metrics["user_losses"] = user_losses
                step_metrics["g_loss"] = g_loss
                step_metrics["u_loss"] = u_loss
                return total

            loss, grads = mx.value_and_grad(loss_fn)(params)
            new_params = self._optimizer.apply_gradients(grads, params)
            self.A = new_params.get("A", {})
            self.B = new_params.get("B", {})

            eval_items = [loss]
            if self.A:
                eval_items.extend(self.A.values())
            if self.B:
                eval_items.extend(self.B.values())
            if step_metrics.get("g_loss") is not None:
                eval_items.append(step_metrics["g_loss"])
            if step_metrics.get("u_loss") is not None:
                eval_items.append(step_metrics["u_loss"])
            eval_items.extend(step_metrics.get("general_losses", []))
            eval_items.extend(step_metrics.get("user_losses", []))
            mx.eval(*eval_items)

            g_loss_value = (
                float(step_metrics["g_loss"].item())
                if step_metrics.get("g_loss") is not None
                else 0.0
            )
            u_loss_value = (
                float(step_metrics["u_loss"].item())
                if step_metrics.get("u_loss") is not None
                else 0.0
            )
            general_loss_values = [
                float(x.item()) for x in step_metrics.get("general_losses", [])
            ]
            user_loss_values = [
                float(x.item()) for x in step_metrics.get("user_losses", [])
            ]
            step_total_loss = float(loss.item())

            step_log = {
                "step": step + 1,
                "loss": step_total_loss,
                "general_loss": g_loss_value,
                "user_loss": u_loss_value,
                "general_sample_losses": general_loss_values,
                "user_sample_losses": user_loss_values,
                "general_samples": len(general_samples),
                "user_samples": len(user_samples),
            }
            step_logs.append(step_log)

            self._append_consolidation_log(
                {
                    "kind": "consolidation_step",
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "run_id": run_id,
                    "step": step + 1,
                    "total_loss": step_total_loss,
                    "general_loss": g_loss_value,
                    "user_loss": u_loss_value,
                    "general_sample_losses": general_loss_values,
                    "user_sample_losses": user_loss_values,
                    "general_samples": len(general_samples),
                    "user_samples": len(user_samples),
                    "general_batch_preview": [
                        self._sample_record_preview(sample)
                        for sample in general_samples[: self.log_max_samples_per_step]
                    ],
                    "user_batch_preview": [
                        self._sample_record_preview(sample)
                        for sample in user_samples[: self.log_max_samples_per_step]
                    ],
                }
            )

        result = {
            "consolidated": True,
            "steps": self.consolidation_steps,
            "user_examples": len(session_pool),
            "ffn_linear_targets": self._num_ffn_targets,
            "run_id": run_id,
            "log_path": self.consolidation_log_path,
            "step_logs": step_logs,
        }
        self._append_consolidation_log(
            {
                "kind": "consolidation_end",
                "ts": datetime.now(timezone.utc).isoformat(),
                "run_id": run_id,
                "consolidated": True,
                "steps": self.consolidation_steps,
                "final_loss": step_logs[-1]["loss"] if step_logs else None,
            }
        )
        self.last_consolidation = copy.deepcopy(result)
        return result

    def update(self, info: Dict[str, Any]) -> None:
        stage = info.get("stage")
        event = info.get("event")
        if stage == "post_hooks" and event == "user_message":
            messages = info.get("messages", [])
            if (
                isinstance(messages, list)
                and len(messages) > 0
                and isinstance(messages[-1], dict)
            ):
                target = messages[-1]
                target_role = str(target.get("role", "user")).strip().lower()
                target_text = str(target.get("content", "")).strip()
                prefix_messages = copy.deepcopy(messages[:-1])
                if target_role == "user" and target_text:
                    self._session_examples.append(
                        {
                            "prefix_messages": prefix_messages,
                            "target_role": "user",
                            "target_text": target_text,
                        }
                    )
                    if len(self._session_examples) > self.max_session_examples:
                        self._session_examples = self._session_examples[
                            -self.max_session_examples :
                        ]

                info["consolidation_pending_examples"] = len(self._session_examples)
                info["consolidation_mode"] = "collect_only"
                info["update_status"] = "ok"

        self.info.update(copy.deepcopy(info))
        self.update_history.append(copy.deepcopy(info))

    def save(self, path: str) -> Dict[str, Any]:
        consolidation_result = {"consolidated": False, "reason": "not-run"}
        if self.consolidate_on_save and len(self._session_examples) > 0:
            try:
                consolidation_result = self._run_consolidation()
            except Exception as exc:
                consolidation_result = {"consolidated": False, "reason": str(exc)}

        if self.A is None or self.B is None or len(self.A) == 0 or len(self.B) == 0:
            return {
                "saved": False,
                "reason": "A/B not initialized",
                "consolidation": consolidation_result,
            }

        try:
            mx, _ = self._import_mlx()
            from mlx.utils import tree_flatten

            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)

            arrays: Dict[str, Any] = {}
            for key, value in self.A.items():
                arrays[f"A::{key}"] = value
            for key, value in self.B.items():
                arrays[f"B::{key}"] = value

            metadata = {
                "memory_module": self.name,
                "rank": str(self.rank),
                "optimizer_name": self.optimizer_name,
                "learning_rate": str(self.learning_rate),
                "weight_decay": str(self.weight_decay),
                "l2_regularization": str(self.l2_regularization),
                "consolidation_steps": str(self.consolidation_steps),
                "general_batch_size": str(self.general_batch_size),
                "user_batch_size": str(self.user_batch_size),
                "max_tokens_per_sample": str(self.max_tokens_per_sample),
                "user_alpha": str(self.user_alpha),
                "general_dataset_name": self.general_dataset_name,
                "general_split": self.general_split,
                "general_init_timeout_sec": str(self.general_init_timeout_sec),
                "general_next_timeout_sec": str(self.general_next_timeout_sec),
                "use_synthetic_general_fallback": str(
                    self.use_synthetic_general_fallback
                ),
                "num_ffn_layers": str(self._num_ffn_layers),
                "num_ffn_targets": str(self._num_ffn_targets),
                "target_keys": "|".join(self._target_keys),
                "consolidation_log_path": str(self.consolidation_log_path),
                "log_max_samples_per_step": str(self.log_max_samples_per_step),
                "log_text_max_chars": str(self.log_text_max_chars),
                "log_prefix_turns": str(self.log_prefix_turns),
            }

            if self._optimizer is not None:
                opt_flat = dict(tree_flatten(self._optimizer.state))
                for key, value in opt_flat.items():
                    arrays[f"optimizer_state::{key}"] = value

            mx.save_safetensors(str(out), arrays, metadata)
            self._session_examples = []
            return {
                "saved": True,
                "path": str(out),
                "consolidation": consolidation_result,
            }
        except Exception as exc:
            return {
                "saved": False,
                "reason": str(exc),
                "consolidation": consolidation_result,
            }

    def load(self, path: str) -> Dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {"loaded": False, "reason": f"file not found: {file_path}"}
        try:
            mx, _ = self._import_mlx()
            from mlx.utils import tree_unflatten

            arrays, metadata = mx.load(str(file_path), return_metadata=True)

            a_map: Dict[str, Any] = {}
            b_map: Dict[str, Any] = {}
            for key, value in arrays.items():
                if key.startswith("A::"):
                    a_map[key[len("A::") :]] = value
                elif key.startswith("B::"):
                    b_map[key[len("B::") :]] = value

            if len(a_map) == 0 or len(b_map) == 0:
                if "A" in arrays or "B" in arrays:
                    return {
                        "loaded": False,
                        "reason": "legacy consolidation checkpoint format is unsupported",
                    }
                return {"loaded": False, "reason": "missing A::/B:: tensors in file"}

            self.A = a_map
            self.B = b_map
            self._target_dims = {}
            for key, a_value in self.A.items():
                b_value = self.B.get(key)
                if b_value is None:
                    continue
                try:
                    self._target_dims[key] = (
                        int(a_value.shape[0]),
                        int(b_value.shape[1]),
                    )
                except Exception:
                    continue

            if "rank" in metadata:
                try:
                    self.rank = max(1, int(metadata["rank"]))
                except Exception:
                    pass
            if "optimizer_name" in metadata:
                self.optimizer_name = str(metadata["optimizer_name"]).strip().lower()
            if "learning_rate" in metadata:
                try:
                    self.learning_rate = float(metadata["learning_rate"])
                except Exception:
                    pass
            if "weight_decay" in metadata:
                try:
                    self.weight_decay = float(metadata["weight_decay"])
                except Exception:
                    pass
            if "l2_regularization" in metadata:
                try:
                    self.l2_regularization = max(
                        0.0, float(metadata["l2_regularization"])
                    )
                except Exception:
                    pass
            if "consolidation_steps" in metadata:
                try:
                    self.consolidation_steps = max(
                        1, int(metadata["consolidation_steps"])
                    )
                except Exception:
                    pass
            if "general_batch_size" in metadata:
                try:
                    self.general_batch_size = max(
                        0, int(metadata["general_batch_size"])
                    )
                except Exception:
                    pass
            if "user_batch_size" in metadata:
                try:
                    self.user_batch_size = max(1, int(metadata["user_batch_size"]))
                except Exception:
                    pass
            if "max_tokens_per_sample" in metadata:
                try:
                    self.max_tokens_per_sample = max(
                        8, int(metadata["max_tokens_per_sample"])
                    )
                except Exception:
                    pass
            if "user_alpha" in metadata:
                try:
                    self.user_alpha = min(1.0, max(0.0, float(metadata["user_alpha"])))
                except Exception:
                    pass
            if "general_dataset_name" in metadata:
                self.general_dataset_name = str(metadata["general_dataset_name"])
            if "general_split" in metadata:
                self.general_split = str(metadata["general_split"])
            if "general_init_timeout_sec" in metadata:
                try:
                    self.general_init_timeout_sec = max(
                        0.0, float(metadata["general_init_timeout_sec"])
                    )
                except Exception:
                    pass
            if "general_next_timeout_sec" in metadata:
                try:
                    self.general_next_timeout_sec = max(
                        0.0, float(metadata["general_next_timeout_sec"])
                    )
                except Exception:
                    pass
            if "use_synthetic_general_fallback" in metadata:
                self.use_synthetic_general_fallback = str(
                    metadata["use_synthetic_general_fallback"]
                ).strip().lower() in {"1", "true", "yes", "on"}
            if "consolidation_log_path" in metadata:
                self.consolidation_log_path = str(metadata["consolidation_log_path"])
            if "log_max_samples_per_step" in metadata:
                try:
                    self.log_max_samples_per_step = max(
                        1, int(metadata["log_max_samples_per_step"])
                    )
                except Exception:
                    pass
            if "log_text_max_chars" in metadata:
                try:
                    self.log_text_max_chars = max(
                        32, int(metadata["log_text_max_chars"])
                    )
                except Exception:
                    pass
            if "log_prefix_turns" in metadata:
                try:
                    self.log_prefix_turns = max(1, int(metadata["log_prefix_turns"]))
                except Exception:
                    pass
            if "num_ffn_layers" in metadata:
                try:
                    self._num_ffn_layers = max(0, int(metadata["num_ffn_layers"]))
                except Exception:
                    pass
            if "num_ffn_targets" in metadata:
                try:
                    self._num_ffn_targets = max(0, int(metadata["num_ffn_targets"]))
                except Exception:
                    pass
            if "target_keys" in metadata:
                raw = str(metadata["target_keys"]).strip()
                self._target_keys = [x for x in raw.split("|") if x]

            opt_items = []
            for key, value in arrays.items():
                if key.startswith("optimizer_state::"):
                    opt_items.append((key[len("optimizer_state::") :], value))
            if opt_items:
                self._ensure_optimizer()
                if self._optimizer is not None:
                    self._optimizer.state = tree_unflatten(opt_items)

            return {"loaded": True, "path": str(file_path)}
        except Exception as exc:
            return {"loaded": False, "reason": str(exc)}
