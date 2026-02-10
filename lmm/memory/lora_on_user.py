from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from lmm.memory.base import MemoryContext, MemoryModule


class LoraOnUserMemoryModule(MemoryModule):
    """
    Low-rank memory adapter applied on hidden states:
      hidden + (hidden @ A @ B)
    where A is [d_model, r] and B is [r, d_model].

    On user-message updates (post-hooks), compute CE loss on the latest user
    message tokens while conditioning on full prior chat context, and extract
    gradients wrt A and B.
    """

    name = "lora_on_user"

    def __init__(
        self,
        rank: int = 8,
        init_std: float = 0.02,
        optimizer_name: str = "adamw",
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        l2_regularization: float = 0.0,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        use_prefix_cache_for_loss: bool = True,
        optimization_steps: int = 1,
    ) -> None:
        self.info: Dict[str, Any] = {}
        self.update_history: List[Dict[str, Any]] = []
        self.rank = rank
        self.init_std = init_std
        self.optimizer_name = optimizer_name.strip().lower()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l2_regularization = max(0.0, float(l2_regularization))
        self.betas = betas
        self.eps = eps
        self.use_prefix_cache_for_loss = use_prefix_cache_for_loss
        self.optimization_steps = max(1, int(optimization_steps))

        self.A = None
        self.B = None
        self._active_params = None
        self._optimizer = None

        self._model = None
        self._tokenizer = None
        self._prompt_builder: Optional[Callable[[list[dict[str, str]], bool], str]] = (
            None
        )

        self.last_loss: Optional[float] = None
        self.last_user_token_count: int = 0
        self.last_grad_norms: Dict[str, float] = {}
        self.last_grads = None
        self._prefix_cache_for_loss = None
        self.excluded_decoded_tokens = {
            "<start_of_turn>",
            "<end_of_turn>",
            "user",
            "assistant",
            "system",
        }

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

    def _ensure_matrices(self, d_model: int, dtype: Any = None) -> None:
        mx, _ = self._import_mlx()
        if d_model <= 0:
            raise ValueError("d_model must be > 0")
        if self.A is not None and self.B is not None:
            if self.A.shape == (d_model, self.rank) and self.B.shape == (
                self.rank,
                d_model,
            ):
                return
        _ = dtype
        target_dtype = mx.float32
        self.A = (
            self.init_std
            * mx.random.normal((d_model, self.rank), dtype=mx.float32)
        ).astype(target_dtype)
        self.B = mx.zeros((self.rank, d_model), dtype=target_dtype)

    def _ensure_optimizer(self) -> None:
        if self._optimizer is not None:
            return

        _, _ = self._import_mlx()
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
            f"Unsupported optimizer '{self.optimizer_name}'. "
            "Supported: adamw, adam, sgd"
        )

    def _current_ab(self):
        if self._active_params is not None:
            return self._active_params["A"], self._active_params["B"]
        return self.A, self.B

    def rewrite(self, hidden_states: Any, context: MemoryContext) -> Any:
        _ = context
        if not hasattr(hidden_states, "shape"):
            return hidden_states

        shape = getattr(hidden_states, "shape", None)
        if not shape:
            return hidden_states

        d_model = int(shape[-1])
        dtype = getattr(hidden_states, "dtype", None)
        self._ensure_matrices(d_model, dtype=dtype)
        A, B = self._current_ab()
        if A is None or B is None:
            return hidden_states

        delta = (hidden_states @ A.astype(hidden_states.dtype)) @ B.astype(
            hidden_states.dtype
        )
        return hidden_states + delta

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

    def _select_high_ce_tokens(
        self,
        token_ids: List[int],
        token_losses: List[float],
    ) -> Tuple[List[int], List[float], float, bool]:
        if len(token_losses) == 0:
            return [], [], 0.0, False
        avg = float(sum(token_losses) / len(token_losses))
        # avg = 0.0
        indices = [i for i, v in enumerate(token_losses) if v > avg]
        fallback_used = False
        if len(indices) == 0:
            # Strictly-above-average can be empty (e.g., all values equal).
            # Fallback to all filtered content tokens to avoid zero-denominator updates.
            indices = list(range(len(token_losses)))
            fallback_used = True
        return (
            [token_ids[i] for i in indices],
            [token_losses[i] for i in indices],
            avg,
            fallback_used,
        )

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

    def _get_user_token_span(
        self, messages: List[Dict[str, str]]
    ) -> Tuple[List[int], int, int]:
        if self._prompt_builder is None:
            return [], 0, 0
        if not messages:
            return [], 0, 0

        full_prompt = self._prompt_builder(messages, False)
        prefix_prompt = self._prompt_builder(messages[:-1], False) if len(messages) > 1 else ""

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
        ab_params: Dict[str, Any],
        input_tokens: Any,
        target_tokens: Any,
        prefix_cache: Any,
    ) -> Tuple[Any, Any]:
        mx, nn = self._import_mlx()
        self._active_params = ab_params
        try:
            local_cache = copy.deepcopy(prefix_cache) if prefix_cache is not None else None
            logits = self._model(input_tokens[None], cache=local_cache)[0]
        finally:
            self._active_params = None
        selected_logits = logits.astype(mx.float32)
        selected_targets = target_tokens
        token_losses = nn.losses.cross_entropy(
            selected_logits,
            selected_targets,
            reduction="none",
        )
        return selected_targets, token_losses

    def _compute_loss_and_grads(
        self, full_tokens: List[int], user_start: int
    ) -> Tuple[
        float,
        int,
        Dict[str, float],
        Dict[str, Any],
        List[int],
        List[float],
        List[int],
        List[float],
        List[int],
        List[float],
        float,
        bool,
    ]:
        mx, _ = self._import_mlx()
        if self._model is None:
            raise RuntimeError("Model runtime is not attached.")
        if len(full_tokens) < 2:
            raise RuntimeError("Not enough tokens for next-token CE.")

        if self.A is None or self.B is None:
            d_model = self._infer_d_model()
            if d_model is None:
                raise RuntimeError("Could not infer d_model to initialize A/B.")
            self._ensure_matrices(d_model, dtype=mx.float32)

        input_tokens, target_tokens, prefix_cache = self._build_user_training_batch(
            full_tokens,
            user_start,
        )
        raw_token_ids = [int(x) for x in target_tokens.tolist()]
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
        keep_mask = mx.array(keep_mask_list, dtype=mx.float32)

        params = {"A": self.A, "B": self.B}

        def loss_fn(ab_params, in_tokens, tgt_tokens):
            _, token_losses = self._forward_user_span(
                ab_params,
                in_tokens,
                tgt_tokens,
                prefix_cache,
            )
            mask_kept = keep_mask.astype(token_losses.dtype)
            kept_mean = mx.sum(token_losses * mask_kept) / mx.sum(mask_kept)
            high_mask = mask_kept * (token_losses > kept_mean).astype(token_losses.dtype)
            high_count = mx.sum(high_mask)
            effective_mask = mx.where(high_count > 0, high_mask, mask_kept)
            ce_loss = mx.sum(token_losses * effective_mask) / mx.sum(effective_mask)
            if self.l2_regularization <= 0.0:
                return ce_loss
            l2_penalty = self.l2_regularization * (
                mx.sum(ab_params["A"] * ab_params["A"])
                + mx.sum(ab_params["B"] * ab_params["B"])
            )
            return ce_loss + l2_penalty

        loss, grads = mx.value_and_grad(loss_fn)(params, input_tokens, target_tokens)
        selected_targets, token_losses = self._forward_user_span(
            params,
            input_tokens,
            target_tokens,
            prefix_cache,
        )
        mx.eval(loss, grads["A"], grads["B"], selected_targets, token_losses)

        grad_a_norm = float(
            mx.sqrt(mx.sum(grads["A"] * grads["A"])).item()
        )
        grad_b_norm = float(
            mx.sqrt(mx.sum(grads["B"] * grads["B"])).item()
        )
        raw_token_loss_values = [float(x) for x in token_losses.tolist()]
        token_ids = [raw_token_ids[i] for i in kept_indices]
        token_loss_values = [raw_token_loss_values[i] for i in kept_indices]
        (
            high_token_ids,
            high_token_losses,
            loss_threshold,
            high_loss_fallback_used,
        ) = self._select_high_ce_tokens(token_ids, token_loss_values)
        self.last_grads = grads
        self.last_loss = float(loss.item())
        self.last_user_token_count = user_token_count
        self.last_grad_norms = {"A": grad_a_norm, "B": grad_b_norm}
        return (
            self.last_loss,
            user_token_count,
            self.last_grad_norms,
            grads,
            token_ids,
            token_loss_values,
            raw_token_ids,
            raw_token_loss_values,
            high_token_ids,
            high_token_losses,
            loss_threshold,
            high_loss_fallback_used,
        )

    def _apply_optimizer_step(self, grads: Dict[str, Any]) -> Dict[str, float]:
        mx, _ = self._import_mlx()
        self._ensure_optimizer()
        if self.A is None or self.B is None:
            raise RuntimeError("A/B are not initialized.")
        if self._optimizer is None:
            raise RuntimeError("Optimizer is not initialized.")

        params = {"A": self.A, "B": self.B}
        new_params = self._optimizer.apply_gradients(grads, params)
        self.A = new_params["A"]
        self.B = new_params["B"]
        mx.eval(self.A, self.B)

        a_norm = float(mx.sqrt(mx.sum(self.A * self.A)).item())
        b_norm = float(mx.sqrt(mx.sum(self.B * self.B)).item())
        return {"A": a_norm, "B": b_norm}

    def update(self, info: Dict[str, Any]) -> None:
        stage = info.get("stage")
        event = info.get("event")
        if stage == "post_hooks" and event == "user_message":
            messages = info.get("messages", [])
            try:
                full_tokens, user_start, total_tokens = self._get_user_token_span(
                    messages
                )
                optimization_losses: List[float] = []
                last_values = None
                for _ in range(self.optimization_steps):
                    last_values = self._compute_loss_and_grads(
                        full_tokens,
                        user_start,
                    )
                    (
                        loss,
                        num_user_tokens,
                        grad_norms,
                        grads,
                        token_ids,
                        token_losses,
                        raw_token_ids,
                        raw_token_losses,
                        high_token_ids,
                        high_token_losses,
                        loss_threshold,
                        high_loss_fallback_used,
                    ) = last_values
                    optimization_losses.append(loss)
                    param_norms = self._apply_optimizer_step(grads)

                if last_values is None:
                    raise RuntimeError("No optimization step executed.")

                info["user_response_loss"] = loss
                info["user_response_token_count"] = num_user_tokens
                info["context_token_count"] = max(0, user_start)
                info["total_token_count"] = total_tokens
                info["user_response_raw_token_count"] = len(raw_token_ids)
                info["user_response_raw_target_token_ids"] = raw_token_ids
                info["user_response_raw_token_losses"] = raw_token_losses
                info["user_response_target_token_ids"] = token_ids
                info["user_response_token_losses"] = token_losses
                info["user_response_token_loss_by_decoded_token"] = (
                    self._token_loss_dict(token_ids, token_losses)
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
                info["grad_A_norm"] = grad_norms["A"]
                info["grad_B_norm"] = grad_norms["B"]
                info["param_A_norm"] = param_norms["A"]
                info["param_B_norm"] = param_norms["B"]
                info["optimizer"] = self.optimizer_name
                info["learning_rate"] = self.learning_rate
                info["l2_regularization"] = self.l2_regularization
                info["lora_rank"] = self.rank
                info["optimization_steps"] = self.optimization_steps
                info["optimization_step_losses"] = optimization_losses
                if self._optimizer is not None:
                    info["optimizer_step"] = int(self._optimizer.step.item())
                info["update_status"] = "ok"
            except Exception as exc:
                info["update_status"] = "no_grad"
                info["grad_error"] = str(exc)

        self.info.update(copy.deepcopy(info))
        self.update_history.append(copy.deepcopy(info))

    def save(self, path: str) -> Dict[str, Any]:
        if self.A is None or self.B is None:
            return {"saved": False, "reason": "A/B not initialized"}
        try:
            mx, _ = self._import_mlx()
            from mlx.utils import tree_flatten

            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            arrays = {"A": self.A, "B": self.B}
            metadata = {
                "memory_module": self.name,
                "rank": str(self.rank),
                "optimizer_name": self.optimizer_name,
                "learning_rate": str(self.learning_rate),
                "weight_decay": str(self.weight_decay),
                "l2_regularization": str(self.l2_regularization),
            }
            if self._optimizer is not None:
                opt_flat = dict(tree_flatten(self._optimizer.state))
                for k, v in opt_flat.items():
                    arrays[f"optimizer_state::{k}"] = v
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
            from mlx.utils import tree_unflatten

            arrays, metadata = mx.load(str(file_path), return_metadata=True)
            if "A" not in arrays or "B" not in arrays:
                return {"loaded": False, "reason": "missing A/B in file"}
            self.A = arrays["A"]
            self.B = arrays["B"]
            if "rank" in metadata:
                try:
                    self.rank = int(metadata["rank"])
                except Exception:
                    pass
            if "l2_regularization" in metadata:
                try:
                    self.l2_regularization = max(
                        0.0, float(metadata["l2_regularization"])
                    )
                except Exception:
                    pass

            opt_items = []
            for k, v in arrays.items():
                if k.startswith("optimizer_state::"):
                    opt_items.append((k[len("optimizer_state::") :], v))
            if opt_items:
                self._ensure_optimizer()
                if self._optimizer is not None:
                    self._optimizer.state = tree_unflatten(opt_items)

            return {"loaded": True, "path": str(file_path)}
        except Exception as exc:
            return {"loaded": False, "reason": str(exc)}
