from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from lmm.memory.base import MemoryContext, MemoryModule


class HashGradientMemoryModule(MemoryModule):
    """
    Hash-based gradient memory.

    A conceptual matrix with 2^r rows and d_model columns stores correction
    vectors. For each hidden state vector h at the hooked layer:
      out = h + M[hash(h)]
    where hash(h) is built from r pairwise channel comparisons.

    On user updates, high-CE tokens are selected and each token updates exactly
    one hash row via a few gradient steps on that token's CE loss.
    """

    name = "hash_gradient"

    def __init__(
        self,
        hash_bits: int = 12,
        hash_seed: int = 0,
        token_update_steps: int = 3,
        token_max_update_steps: int = 20,
        token_target_loss: float = 1.0,
        token_learning_rate: float = 5e-2,
        token_optimizer: str = "adam",
        token_l2_regularization: float = 0.0,
        token_adam_betas: Tuple[float, float] = (0.9, 0.999),
        token_adam_eps: float = 1e-8,
        fallback_to_nearest: bool = True,
        min_shared_bits_ratio: float = 0.8,
        use_prefix_cache_for_loss: bool = True,
        max_tokens_to_update: int = 0,
    ) -> None:
        self.info: Dict[str, Any] = {}
        self.update_history: List[Dict[str, Any]] = []

        self.hash_bits = max(1, int(hash_bits))
        self.hash_seed = int(hash_seed)
        self.num_buckets = 1 << self.hash_bits
        self.token_update_steps = max(1, int(token_update_steps))
        self.token_max_update_steps = max(self.token_update_steps, int(token_max_update_steps))
        self.token_target_loss = max(0.0, float(token_target_loss))
        self.token_learning_rate = max(0.0, float(token_learning_rate))
        self.token_optimizer = token_optimizer.strip().lower()
        self.token_l2_regularization = max(0.0, float(token_l2_regularization))
        self.token_adam_betas = token_adam_betas
        self.token_adam_eps = max(1e-12, float(token_adam_eps))
        self.fallback_to_nearest = bool(fallback_to_nearest)
        self.min_shared_bits_ratio = min(1.0, max(0.0, float(min_shared_bits_ratio)))
        self.use_prefix_cache_for_loss = use_prefix_cache_for_loss
        self.max_tokens_to_update = max(0, int(max_tokens_to_update))

        self._d_model: Optional[int] = None
        self._hash_left: Optional[Any] = None
        self._hash_right: Optional[Any] = None
        self._hash_pairs: List[Tuple[int, int]] = []
        self._hash_weights: Optional[Any] = None

        # Sparse storage over the conceptual 2^r x d_model table.
        self._memory_rows: Dict[int, Any] = {}
        self._used_hashes: set[int] = set()
        self._zero_row: Optional[Any] = None

        self._model = None
        self._tokenizer = None
        self._prompt_builder: Optional[Callable[[list[dict[str, str]], bool], str]] = (
            None
        )
        self._prefix_cache_for_loss = None

        self._capture_hashes = False
        self._captured_raw_hashes: List[int] = []
        self._captured_effective_hashes: List[int] = []
        self._captured_fallback_flags: List[bool] = []
        self._captured_shared_bits: List[int] = []
        self._override_position: Optional[int] = None
        self._override_correction: Optional[Any] = None

        self.excluded_decoded_tokens = {
            "<start_of_turn>",
            "<end_of_turn>",
            "user",
            "assistant",
            "system",
        }
        self.last_loss: Optional[float] = None

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

    def set_prefix_cache(self, cache: Any) -> None:
        self._prefix_cache_for_loss = cache

    def _import_mlx(self):
        import mlx.core as mx
        import mlx.nn as nn

        return mx, nn

    def _infer_d_model(self) -> Optional[int]:
        candidates = [
            ("args", "hidden_size"),
            ("model", "args", "hidden_size"),
            ("model", "model", "args", "hidden_size"),
            ("language_model", "args", "hidden_size"),
            ("language_model", "model", "args", "hidden_size"),
        ]
        for path in candidates:
            obj = self._model
            ok = True
            for attr in path:
                if not hasattr(obj, attr):
                    ok = False
                    break
                obj = getattr(obj, attr)
            if ok:
                try:
                    return int(obj)
                except Exception:
                    continue
        return None

    def _ensure_hash_pairs(self, d_model: int) -> None:
        mx, _ = self._import_mlx()
        needs_init = False
        if self._hash_left is None or self._hash_right is None or self._hash_weights is None:
            needs_init = True
        elif len(self._hash_pairs) != self.hash_bits:
            needs_init = True
        elif any(a >= d_model or b >= d_model for a, b in self._hash_pairs):
            needs_init = True

        if not needs_init:
            return

        rng = np.random.default_rng(self.hash_seed)
        pairs: List[Tuple[int, int]] = []
        for _ in range(self.hash_bits):
            left = int(rng.integers(0, d_model))
            right = int(rng.integers(0, d_model))
            if d_model > 1:
                while right == left:
                    right = int(rng.integers(0, d_model))
            pairs.append((left, right))

        self._hash_pairs = pairs
        left_indices = [a for a, _ in pairs]
        right_indices = [b for _, b in pairs]
        self._hash_left = mx.array(left_indices, dtype=mx.int32)
        self._hash_right = mx.array(right_indices, dtype=mx.int32)
        self._hash_weights = mx.array(
            [1 << i for i in range(self.hash_bits)],
            dtype=mx.int32,
        )

    def _ensure_memory(self, d_model: int) -> None:
        mx, _ = self._import_mlx()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self._d_model is not None and self._d_model != d_model:
            # Different model width => clear incompatible rows.
            self._memory_rows = {}
            self._used_hashes = set()
        self._d_model = d_model
        if self._zero_row is None or int(self._zero_row.shape[0]) != d_model:
            self._zero_row = mx.zeros((d_model,), dtype=mx.float32)
        self._ensure_hash_pairs(d_model)

    def _encode_prompt(self, text: str) -> List[int]:
        if self._tokenizer is None:
            return []
        bos_token = getattr(self._tokenizer, "bos_token", None)
        add_special_tokens = bos_token is None or not text.startswith(bos_token)
        return list(
            self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        )

    def _decode_token_id(self, token_id: int) -> str:
        if self._tokenizer is None:
            return f"<{token_id}>"
        try:
            return str(self._tokenizer.decode([int(token_id)]))
        except Exception:
            try:
                return str(self._tokenizer.decode(int(token_id)))
            except Exception:
                return f"<{token_id}>"

    def _token_loss_dict(
        self, token_ids: List[int], token_losses: List[float]
    ) -> Dict[str, float]:
        decoded_counts: Dict[str, int] = {}
        out: Dict[str, float] = {}
        for token_id, loss in zip(token_ids, token_losses):
            decoded = self._decode_token_id(token_id)
            count = decoded_counts.get(decoded, 0) + 1
            decoded_counts[decoded] = count
            key = decoded if count == 1 else f"{decoded}#{count}"
            out[key] = float(loss)
        return out

    def _is_excluded_loss_token(self, token_id: int, decoded: str) -> bool:
        special_ids = set(getattr(self._tokenizer, "all_special_ids", []) or [])
        if token_id in special_ids:
            return True
        normalized = decoded.strip()
        if normalized == "":
            return True
        if normalized in self.excluded_decoded_tokens:
            return True
        return False

    def _select_high_ce_positions(
        self,
        kept_raw_positions: List[int],
        kept_losses: List[float],
    ) -> Tuple[List[int], float, bool]:
        if len(kept_losses) == 0:
            return [], 0.0, False
        avg = float(sum(kept_losses) / len(kept_losses))
        selected = [
            raw_pos
            for raw_pos, loss in zip(kept_raw_positions, kept_losses)
            if loss > avg
        ]
        fallback_used = False
        if len(selected) == 0:
            selected = list(kept_raw_positions)
            fallback_used = True
        if self.max_tokens_to_update > 0 and len(selected) > self.max_tokens_to_update:
            selected = selected[: self.max_tokens_to_update]
        return selected, avg, fallback_used

    def _min_shared_bits(self) -> int:
        return int(math.ceil(self.hash_bits * self.min_shared_bits_ratio))

    def _resolve_effective_hash(self, raw_hash: int) -> Tuple[int, bool, int]:
        if raw_hash in self._used_hashes:
            return raw_hash, False, self.hash_bits
        if not self.fallback_to_nearest:
            return raw_hash, False, 0
        if not self._used_hashes:
            return raw_hash, False, 0

        min_shared = self._min_shared_bits()
        best_hash = raw_hash
        best_shared = -1
        best_dist = self.hash_bits + 1
        for used_hash in self._used_hashes:
            dist = (raw_hash ^ used_hash).bit_count()
            shared = self.hash_bits - dist
            if shared < min_shared:
                continue
            if shared > best_shared or (shared == best_shared and dist < best_dist):
                best_hash = used_hash
                best_shared = shared
                best_dist = dist

        if best_shared < min_shared:
            return raw_hash, False, 0
        return best_hash, True, best_shared

    def _hash_to_binary(self, hash_index: int) -> str:
        masked = int(hash_index) & (self.num_buckets - 1)
        return format(masked, f"0{self.hash_bits}b")

    def _compute_hash_indices(self, hidden_states: Any) -> Any:
        mx, _ = self._import_mlx()
        if self._hash_left is None or self._hash_right is None or self._hash_weights is None:
            raise RuntimeError("Hash projection is not initialized.")
        left_vals = hidden_states[..., self._hash_left]
        right_vals = hidden_states[..., self._hash_right]
        bits = (left_vals > right_vals).astype(mx.int32)
        return mx.sum(bits * self._hash_weights, axis=-1).astype(mx.int32)

    def _lookup_corrections(
        self,
        raw_hash_flat: Sequence[int],
        out_dtype: Any,
    ) -> Tuple[Any, List[int], List[bool], List[int]]:
        mx, _ = self._import_mlx()
        if self._d_model is None:
            raise RuntimeError("d_model is not initialized.")
        if self._zero_row is None:
            self._zero_row = mx.zeros((self._d_model,), dtype=mx.float32)

        vectors: List[Any] = []
        effective_hashes: List[int] = []
        fallback_flags: List[bool] = []
        shared_bits_list: List[int] = []
        for raw_hash in raw_hash_flat:
            effective, fallback_used, shared_bits = self._resolve_effective_hash(raw_hash)
            row = self._memory_rows.get(effective, self._zero_row)
            vectors.append(row.astype(out_dtype))
            effective_hashes.append(effective)
            fallback_flags.append(fallback_used)
            shared_bits_list.append(shared_bits)

        if len(vectors) == 0:
            corr = mx.zeros((0, self._d_model), dtype=out_dtype)
        else:
            corr = mx.stack(vectors, axis=0)
        return corr, effective_hashes, fallback_flags, shared_bits_list

    def rewrite(self, hidden_states: Any, context: MemoryContext) -> Any:
        _ = context
        if not hasattr(hidden_states, "shape"):
            return hidden_states

        shape = getattr(hidden_states, "shape", None)
        if not shape or len(shape) < 2:
            return hidden_states
        batch = int(shape[0])
        seq_len = int(shape[1])
        d_model = int(shape[-1])
        if batch <= 0 or seq_len <= 0 or d_model <= 0:
            return hidden_states

        self._ensure_memory(d_model)
        mx, _ = self._import_mlx()

        raw_hash_arr = self._compute_hash_indices(hidden_states.astype(mx.float32))
        raw_hash_flat = [int(x) for x in raw_hash_arr.reshape(-1).tolist()]
        corr_flat, effective_hashes, fallback_flags, shared_bits_list = self._lookup_corrections(
            raw_hash_flat,
            hidden_states.dtype,
        )
        corrections = corr_flat.reshape((batch, seq_len, d_model))

        if self._capture_hashes:
            start = 0
            end = seq_len
            self._captured_raw_hashes = raw_hash_flat[start:end]
            self._captured_effective_hashes = effective_hashes[start:end]
            self._captured_fallback_flags = fallback_flags[start:end]
            self._captured_shared_bits = shared_bits_list[start:end]

        if self._override_position is not None and self._override_correction is not None:
            pos = int(self._override_position)
            if 0 <= pos < seq_len:
                pos_mask = (mx.arange(seq_len, dtype=mx.int32) == pos).astype(
                    hidden_states.dtype
                )[None, :, None]
                batch_mask = (mx.arange(batch, dtype=mx.int32) == 0).astype(
                    hidden_states.dtype
                )[:, None, None]
                mask = pos_mask * batch_mask
                override_vec = self._override_correction.astype(hidden_states.dtype)[
                    None, None, :
                ]
                corrections = corrections * (1.0 - mask) + override_vec * mask

        return hidden_states + corrections

    def _get_user_token_span(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[int], int, int]:
        if self._prompt_builder is None:
            return [], 0, 0
        if not messages:
            return [], 0, 0

        full_prompt = self._prompt_builder(messages, False)
        prefix_prompt = (
            self._prompt_builder(messages[:-1], False) if len(messages) > 1 else ""
        )

        full_tokens = self._encode_prompt(full_prompt)
        prefix_tokens = self._encode_prompt(prefix_prompt)
        user_start = min(len(prefix_tokens), len(full_tokens))
        return full_tokens, user_start, len(full_tokens)

    def _build_user_training_batch(
        self,
        full_tokens: List[int],
        user_start: int,
    ) -> Tuple[Any, Any, Any]:
        mx, _ = self._import_mlx()
        target_start = max(1, int(user_start))
        if target_start >= len(full_tokens):
            raise RuntimeError("User span has no predictable tokens.")

        prefix_tokens = full_tokens[: max(0, target_start - 1)]
        input_tokens = full_tokens[target_start - 1 : -1]
        target_tokens = full_tokens[target_start:]
        if len(input_tokens) != len(target_tokens):
            raise RuntimeError("Training batch alignment failed.")
        if len(target_tokens) == 0:
            raise RuntimeError("No target tokens available.")

        prefix_cache = None
        if self.use_prefix_cache_for_loss and self._prefix_cache_for_loss is not None:
            try:
                from mlx_lm.models import cache as cache_utils

                prefix_cache = copy.deepcopy(self._prefix_cache_for_loss)
                if target_start > 0 and cache_utils.can_trim_prompt_cache(prefix_cache):
                    cache_utils.trim_prompt_cache(prefix_cache, 1)
            except Exception:
                prefix_cache = None

        if self.use_prefix_cache_for_loss and prefix_cache is None:
            try:
                from mlx_lm.models import cache as cache_utils
            except Exception as exc:
                raise RuntimeError("Failed to import mlx_lm cache utilities.") from exc

            prefix_cache = cache_utils.make_prompt_cache(self._model)
            if len(prefix_tokens) > 0:
                prefix_arr = mx.array(prefix_tokens, dtype=mx.uint32)
                _ = self._model(prefix_arr[None], cache=prefix_cache)
                mx.eval([c.state for c in prefix_cache])

        return (
            mx.array(input_tokens, dtype=mx.uint32),
            mx.array(target_tokens, dtype=mx.uint32),
            prefix_cache,
        )

    def _forward_user_span(
        self,
        input_tokens: Any,
        target_tokens: Any,
        prefix_cache: Any,
        *,
        capture_hashes: bool = False,
        override_position: Optional[int] = None,
        override_correction: Optional[Any] = None,
    ) -> Tuple[Any, Any, List[int], List[int], List[bool], List[int]]:
        mx, nn = self._import_mlx()
        self._capture_hashes = bool(capture_hashes)
        self._captured_raw_hashes = []
        self._captured_effective_hashes = []
        self._captured_fallback_flags = []
        self._captured_shared_bits = []
        self._override_position = override_position
        self._override_correction = override_correction
        try:
            local_cache = copy.deepcopy(prefix_cache) if prefix_cache is not None else None
            logits = self._model(input_tokens[None], cache=local_cache)[0]
        finally:
            self._capture_hashes = False
            self._override_position = None
            self._override_correction = None
        selected_logits = logits.astype(mx.float32)
        selected_targets = target_tokens
        token_losses = nn.losses.cross_entropy(
            selected_logits,
            selected_targets,
            reduction="none",
        )
        return (
            selected_targets,
            token_losses,
            list(self._captured_raw_hashes),
            list(self._captured_effective_hashes),
            list(self._captured_fallback_flags),
            list(self._captured_shared_bits),
        )

    def _optimize_correction_for_position(
        self,
        *,
        raw_position: int,
        hash_index: int,
        input_tokens: Any,
        target_tokens: Any,
        prefix_cache: Any,
    ) -> Tuple[Any, List[float], str]:
        mx, _ = self._import_mlx()
        if self._d_model is None:
            raise RuntimeError("d_model is not initialized.")
        start_vec = self._memory_rows.get(hash_index, self._zero_row)
        if start_vec is None:
            start_vec = mx.zeros((self._d_model,), dtype=mx.float32)
        corr = start_vec.astype(mx.float32)
        step_losses: List[float] = []

        optimizer = self.token_optimizer
        if optimizer not in {"sgd", "adam"}:
            raise ValueError(
                f"Unsupported token_optimizer '{optimizer}'. Supported: sgd, adam"
            )
        beta1, beta2 = self.token_adam_betas
        m = mx.zeros_like(corr)
        v = mx.zeros_like(corr)

        def loss_fn(correction_vec):
            _, token_losses, _, _, _, _ = self._forward_user_span(
                input_tokens,
                target_tokens,
                prefix_cache,
                capture_hashes=False,
                override_position=raw_position,
                override_correction=correction_vec,
            )
            loss = token_losses[raw_position]
            if self.token_l2_regularization > 0.0:
                loss = loss + self.token_l2_regularization * mx.mean(
                    correction_vec * correction_vec
                )
            return loss

        stop_reason = "max_steps"
        for step in range(self.token_max_update_steps):
            loss, grad = mx.value_and_grad(loss_fn)(corr)
            mx.eval(loss, grad)
            loss_value = float(loss.item())
            step_losses.append(loss_value)
            reached_min_steps = (step + 1) >= self.token_update_steps
            if reached_min_steps and loss_value <= self.token_target_loss:
                stop_reason = "target_loss_reached"
                break
            if self.token_learning_rate <= 0.0:
                continue

            if optimizer == "sgd":
                corr = corr - self.token_learning_rate * grad
            else:
                # Ephemeral Adam on a single correction vector.
                m = beta1 * m + (1.0 - beta1) * grad
                v = beta2 * v + (1.0 - beta2) * (grad * grad)
                t = step + 1
                m_hat = m / (1.0 - (beta1**t))
                v_hat = v / (1.0 - (beta2**t))
                corr = corr - self.token_learning_rate * m_hat / (
                    mx.sqrt(v_hat) + self.token_adam_eps
                )

        return corr.astype(mx.float32), step_losses, stop_reason

    def update(self, info: Dict[str, Any]) -> None:
        stage = info.get("stage")
        event = info.get("event")
        if stage == "post_hooks" and event == "user_message":
            messages = info.get("messages", [])
            try:
                if self._model is None:
                    raise RuntimeError("Model runtime is not attached.")

                full_tokens, user_start, total_tokens = self._get_user_token_span(
                    messages
                )
                if len(full_tokens) < 2:
                    raise RuntimeError("Not enough tokens for next-token CE.")

                d_model = self._infer_d_model()
                if d_model is None:
                    raise RuntimeError("Could not infer d_model for hash memory.")
                self._ensure_memory(d_model)

                input_tokens, target_tokens, prefix_cache = self._build_user_training_batch(
                    full_tokens,
                    user_start,
                )
                (
                    _selected_targets,
                    token_losses,
                    raw_hashes,
                    effective_hashes,
                    fallback_flags,
                    shared_bits,
                ) = self._forward_user_span(
                    input_tokens,
                    target_tokens,
                    prefix_cache,
                    capture_hashes=True,
                )
                raw_token_ids = [int(x) for x in target_tokens.tolist()]
                raw_token_loss_values = [float(x) for x in token_losses.tolist()]
                if len(raw_hashes) != len(raw_token_ids):
                    raise RuntimeError(
                        "Captured hash count does not match token count."
                    )

                decoded_tokens = [self._decode_token_id(tid) for tid in raw_token_ids]
                keep_mask_list = [
                    0.0 if self._is_excluded_loss_token(tid, tok) else 1.0
                    for tid, tok in zip(raw_token_ids, decoded_tokens)
                ]
                kept_indices = [i for i, m in enumerate(keep_mask_list) if m > 0.0]
                user_token_count = len(kept_indices)
                if user_token_count <= 0:
                    raise RuntimeError(
                        "User span has no predictable content tokens after filtering."
                    )

                kept_token_ids = [raw_token_ids[i] for i in kept_indices]
                kept_token_losses = [raw_token_loss_values[i] for i in kept_indices]
                selected_positions, loss_threshold, high_loss_fallback_used = (
                    self._select_high_ce_positions(kept_indices, kept_token_losses)
                )
                token_hash_records: List[Dict[str, Any]] = []
                for raw_pos, token_id, token_loss in zip(
                    range(len(raw_token_ids)),
                    raw_token_ids,
                    raw_token_loss_values,
                ):
                    raw_hash = int(raw_hashes[raw_pos])
                    effective_hash = int(effective_hashes[raw_pos])
                    token_hash_records.append(
                        {
                            "raw_position": int(raw_pos),
                            "target_token_id": int(token_id),
                            "decoded_token": decoded_tokens[raw_pos],
                            "token_loss": float(token_loss),
                            "raw_hash": raw_hash,
                            "raw_hash_binary": self._hash_to_binary(raw_hash),
                            "effective_hash": effective_hash,
                            "effective_hash_binary": self._hash_to_binary(
                                effective_hash
                            ),
                            "fallback_used": bool(fallback_flags[raw_pos]),
                            "shared_bits_with_effective_hash": int(shared_bits[raw_pos]),
                        }
                    )

                per_token_updates: List[Dict[str, Any]] = []
                for raw_pos in selected_positions:
                    hash_index = int(raw_hashes[raw_pos])
                    updated_vec, step_losses, stop_reason = (
                        self._optimize_correction_for_position(
                        raw_position=raw_pos,
                        hash_index=hash_index,
                        input_tokens=input_tokens,
                        target_tokens=target_tokens,
                        prefix_cache=prefix_cache,
                        )
                    )
                    self._memory_rows[hash_index] = updated_vec
                    self._used_hashes.add(hash_index)
                    per_token_updates.append(
                        {
                            "raw_position": int(raw_pos),
                            "target_token_id": int(raw_token_ids[raw_pos]),
                            "decoded_token": self._decode_token_id(raw_token_ids[raw_pos]),
                            "hash": hash_index,
                            "hash_binary": self._hash_to_binary(hash_index),
                            "step_losses": step_losses,
                            "steps_taken": len(step_losses),
                            "stop_reason": stop_reason,
                            "target_loss": self.token_target_loss,
                            "fallback_used_in_forward": bool(fallback_flags[raw_pos]),
                            "resolved_hash_in_forward": int(effective_hashes[raw_pos]),
                            "resolved_hash_in_forward_binary": self._hash_to_binary(
                                int(effective_hashes[raw_pos])
                            ),
                            "shared_bits_with_resolved_hash": int(shared_bits[raw_pos]),
                            "matrix_row_updated": True,
                            "matrix_row_hash_updated": hash_index,
                            "matrix_row_hash_updated_binary": self._hash_to_binary(
                                hash_index
                            ),
                        }
                    )

                high_token_ids = [raw_token_ids[i] for i in selected_positions]
                high_token_losses = [raw_token_loss_values[i] for i in selected_positions]

                final_loss = (
                    float(sum(high_token_losses) / len(high_token_losses))
                    if len(high_token_losses) > 0
                    else float(sum(kept_token_losses) / len(kept_token_losses))
                )
                self.last_loss = final_loss

                info["user_response_loss"] = final_loss
                info["user_response_token_count"] = user_token_count
                info["context_token_count"] = max(0, user_start)
                info["total_token_count"] = total_tokens
                info["user_response_raw_token_count"] = len(raw_token_ids)
                info["user_response_raw_target_token_ids"] = raw_token_ids
                info["user_response_raw_token_losses"] = raw_token_loss_values
                info["user_response_target_token_ids"] = kept_token_ids
                info["user_response_token_losses"] = kept_token_losses
                info["user_response_token_loss_by_decoded_token"] = self._token_loss_dict(
                    kept_token_ids,
                    kept_token_losses,
                )
                info["user_response_high_ce_threshold"] = loss_threshold
                info["user_response_high_ce_target_token_ids"] = high_token_ids
                info["user_response_high_ce_token_losses"] = high_token_losses
                info["user_response_high_ce_token_loss_by_decoded_token"] = (
                    self._token_loss_dict(high_token_ids, high_token_losses)
                )
                info["user_response_high_ce_fallback_all_filtered"] = (
                    high_loss_fallback_used
                )

                info["hash_bits"] = self.hash_bits
                info["num_buckets"] = self.num_buckets
                info["fallback_to_nearest"] = self.fallback_to_nearest
                info["min_shared_bits_ratio"] = self.min_shared_bits_ratio
                info["token_update_steps"] = self.token_update_steps
                info["token_max_update_steps"] = self.token_max_update_steps
                info["token_target_loss"] = self.token_target_loss
                info["token_learning_rate"] = self.token_learning_rate
                info["token_optimizer"] = self.token_optimizer
                info["token_l2_regularization"] = self.token_l2_regularization
                info["memory_rows_used"] = len(self._used_hashes)
                info["captured_hashes_raw"] = raw_hashes
                info["captured_hashes_raw_binary"] = [
                    self._hash_to_binary(h) for h in raw_hashes
                ]
                info["captured_hashes_effective"] = effective_hashes
                info["captured_hashes_effective_binary"] = [
                    self._hash_to_binary(h) for h in effective_hashes
                ]
                info["captured_hashes_fallback_flags"] = fallback_flags
                info["captured_hashes_shared_bits"] = shared_bits
                info["token_hashes"] = token_hash_records
                info["selected_positions_for_update"] = selected_positions
                info["selected_hashes_for_update"] = [
                    int(raw_hashes[i]) for i in selected_positions
                ]
                info["selected_hashes_for_update_binary"] = [
                    self._hash_to_binary(int(raw_hashes[i])) for i in selected_positions
                ]
                info["updated_hash_rows"] = [
                    int(raw_hashes[i]) for i in selected_positions
                ]
                info["updated_hash_rows_binary"] = [
                    self._hash_to_binary(int(raw_hashes[i])) for i in selected_positions
                ]
                info["per_token_updates"] = per_token_updates
                info["update_status"] = "ok"
            except Exception as exc:
                info["update_status"] = "no_grad"
                info["grad_error"] = str(exc)

        self.info.update(copy.deepcopy(info))
        self.update_history.append(copy.deepcopy(info))

    def save(self, path: str) -> Dict[str, Any]:
        try:
            mx, _ = self._import_mlx()
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)

            if self._d_model is None:
                return {"saved": False, "reason": "d_model is not initialized"}

            used_indices = sorted(self._used_hashes)
            if len(used_indices) > 0:
                index_arr = mx.array(used_indices, dtype=mx.int32)
                value_arr = mx.stack(
                    [self._memory_rows[i].astype(mx.float32) for i in used_indices],
                    axis=0,
                )
            else:
                index_arr = mx.array([], dtype=mx.int32)
                value_arr = mx.zeros((0, self._d_model), dtype=mx.float32)

            left_arr = (
                self._hash_left
                if self._hash_left is not None
                else mx.array([], dtype=mx.int32)
            )
            right_arr = (
                self._hash_right
                if self._hash_right is not None
                else mx.array([], dtype=mx.int32)
            )
            arrays = {
                "memory_indices": index_arr,
                "memory_values": value_arr,
                "hash_left": left_arr,
                "hash_right": right_arr,
            }
            metadata = {
                "memory_module": self.name,
                "hash_bits": str(self.hash_bits),
                "hash_seed": str(self.hash_seed),
                "num_buckets": str(self.num_buckets),
                "d_model": str(self._d_model),
                "token_update_steps": str(self.token_update_steps),
                "token_max_update_steps": str(self.token_max_update_steps),
                "token_target_loss": str(self.token_target_loss),
                "token_learning_rate": str(self.token_learning_rate),
                "token_optimizer": self.token_optimizer,
                "token_l2_regularization": str(self.token_l2_regularization),
                "fallback_to_nearest": str(self.fallback_to_nearest),
                "min_shared_bits_ratio": str(self.min_shared_bits_ratio),
                "max_tokens_to_update": str(self.max_tokens_to_update),
            }
            mx.save_safetensors(str(out), arrays, metadata)
            return {"saved": True, "path": str(out)}
        except Exception as exc:
            return {"saved": False, "reason": str(exc)}

    def load(self, path: str) -> Dict[str, Any]:
        file_path = Path(path)
        if not file_path.exists():
            return {"loaded": False, "reason": f"file not found: {file_path}"}
        try:
            mx, _ = self._import_mlx()
            arrays, metadata = mx.load(str(file_path), return_metadata=True)
            if "memory_indices" not in arrays or "memory_values" not in arrays:
                return {
                    "loaded": False,
                    "reason": "missing memory_indices/memory_values in file",
                }

            if "hash_bits" in metadata:
                try:
                    self.hash_bits = max(1, int(metadata["hash_bits"]))
                    self.num_buckets = 1 << self.hash_bits
                except Exception:
                    pass
            if "hash_seed" in metadata:
                try:
                    self.hash_seed = int(metadata["hash_seed"])
                except Exception:
                    pass
            if "token_update_steps" in metadata:
                try:
                    self.token_update_steps = max(
                        1, int(metadata["token_update_steps"])
                    )
                except Exception:
                    pass
            if "token_max_update_steps" in metadata:
                try:
                    self.token_max_update_steps = max(
                        self.token_update_steps,
                        int(metadata["token_max_update_steps"]),
                    )
                except Exception:
                    pass
            if "token_target_loss" in metadata:
                try:
                    self.token_target_loss = max(
                        0.0, float(metadata["token_target_loss"])
                    )
                except Exception:
                    pass
            if "token_learning_rate" in metadata:
                try:
                    self.token_learning_rate = max(
                        0.0, float(metadata["token_learning_rate"])
                    )
                except Exception:
                    pass
            if "token_optimizer" in metadata:
                self.token_optimizer = str(metadata["token_optimizer"]).strip().lower()
            if "token_l2_regularization" in metadata:
                try:
                    self.token_l2_regularization = max(
                        0.0, float(metadata["token_l2_regularization"])
                    )
                except Exception:
                    pass
            if "fallback_to_nearest" in metadata:
                try:
                    self.fallback_to_nearest = (
                        str(metadata["fallback_to_nearest"]).strip().lower()
                        in {"1", "true", "yes", "on"}
                    )
                except Exception:
                    pass
            if "min_shared_bits_ratio" in metadata:
                try:
                    self.min_shared_bits_ratio = min(
                        1.0, max(0.0, float(metadata["min_shared_bits_ratio"]))
                    )
                except Exception:
                    pass
            if "max_tokens_to_update" in metadata:
                try:
                    self.max_tokens_to_update = max(
                        0, int(metadata["max_tokens_to_update"])
                    )
                except Exception:
                    pass

            memory_indices = [int(x) for x in arrays["memory_indices"].tolist()]
            memory_values = arrays["memory_values"]
            if len(memory_indices) != int(memory_values.shape[0]):
                return {
                    "loaded": False,
                    "reason": "memory_indices length does not match memory_values rows",
                }

            if "d_model" in metadata:
                try:
                    self._d_model = int(metadata["d_model"])
                except Exception:
                    self._d_model = None
            if self._d_model is None:
                self._d_model = int(memory_values.shape[1]) if len(memory_indices) > 0 else None

            self._memory_rows = {}
            self._used_hashes = set()
            for row_idx, hash_idx in enumerate(memory_indices):
                self._memory_rows[hash_idx] = memory_values[row_idx].astype(mx.float32)
                self._used_hashes.add(hash_idx)

            if self._d_model is not None:
                self._zero_row = mx.zeros((self._d_model,), dtype=mx.float32)

            if "hash_left" in arrays and "hash_right" in arrays:
                left_list = [int(x) for x in arrays["hash_left"].tolist()]
                right_list = [int(x) for x in arrays["hash_right"].tolist()]
                if len(left_list) == self.hash_bits and len(right_list) == self.hash_bits:
                    self._hash_pairs = list(zip(left_list, right_list))
                    self._hash_left = arrays["hash_left"].astype(mx.int32)
                    self._hash_right = arrays["hash_right"].astype(mx.int32)
                    self._hash_weights = mx.array(
                        [1 << i for i in range(self.hash_bits)],
                        dtype=mx.int32,
                    )
                else:
                    self._hash_pairs = []
                    self._hash_left = None
                    self._hash_right = None
                    self._hash_weights = None

            return {"loaded": True, "path": str(file_path)}
        except Exception as exc:
            return {"loaded": False, "reason": str(exc)}
