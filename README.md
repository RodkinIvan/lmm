# lmm

Local Memory Module (LMM) playground.

## Current status

- Step 1 implemented: CLI chat interface for an `mlx_lm` model.
- Step 2 implemented: pluggable memory module hook at middle hidden layer.
- Modules:
  - `identity`: returns hidden states unchanged.
  - `lora_on_user`: applies `hidden + hidden @ A @ B` where
    `A in R[d_model x r]` (normal init) and `B in R[r x d_model]` (zero init).
    Because `B` starts as zeros, initial behavior is identity.
  - `hash_gradient`: hash-based correction memory. A conceptual `2^r x d_model`
    table stores correction rows indexed by a binary hash of middle-layer hidden
    states. On user updates, high-CE tokens update their hashed row by a few
    gradient steps on that token loss.

## Run (conda `ai` env)

```bash
conda run -n ai python -m lmm.chat \
  --backend mlx_lm \
  --model google/gemma-3-1b-it \
  --memory identity
```

Try hash-based memory:

```bash
conda run -n ai python -m lmm.chat \
  --backend mlx_lm \
  --model google/gemma-3-1b-it \
  --memory hash_gradient
```

Then type messages in the prompt. Type `exit` or `quit` to stop.

The app starts model loading at startup (before the first user message).
To persist memory-module state between runs, pass separate load/save paths:

```bash
conda run -n ai python -m lmm.chat \
  --backend mlx_lm \
  --model google/gemma-3-1b-it \
  --memory lora_on_user \
  --mm-load-path artifacts/lora_mm_prev.safetensors \
  --mm-save-path artifacts/lora_mm_next.safetensors
```

If `--mm-load-path` does not exist, load is skipped.
If `--mm-save-path` is omitted, save path defaults to:

- `artifacts/module_name_<YYYYMMDD_HHMMSS_microseconds>.safetensors`

`--mm-path` is still accepted as a legacy alias and sets both load/save paths.
Save behavior on exit:

- normal `exit` / `quit`: save
- `Ctrl+D` (EOF): save
- `Ctrl+C` (KeyboardInterrupt): skip save

Decoding defaults to deterministic. You can control it via:

- `--deterministic` / `--non-deterministic`
- `--decoding-strategy auto|greedy|beam|sample`

In the current `mlx_lm` API used here, beam search is not exposed, so `beam`
falls back to greedy (deterministic mode) or sampling (non-deterministic mode).

Use `--local-files-only` if model files are already cached and you want no network use:

```bash
conda run -n ai python -m lmm.chat \
  --backend mlx_lm \
  --model google/gemma-3-1b-it \
  --memory identity \
  --local-files-only
```

Memory module activation logging is enabled by default and written as JSONL to:

- `logs/memory_activations.jsonl`

Each hook call logs input/output slices for at least `hidden_states[:, -1, -5:]`.
Override file path with `--activation-log-path`.

Update callbacks are logged to the same JSONL file with `kind="update"`:

- `stage="pre_hooks"` before user-message callbacks.
- `stage="post_hooks"` after callbacks mutate `info`
  (including `info["last_user_response"]` from the default hook).

For `lora_on_user`, `post_hooks` also attempts to compute cross-entropy loss and
gradients (`grad_A_norm`, `grad_B_norm`) on the latest user message tokens while
conditioning on the full previous chat context, then runs exactly one optimizer
step to update `A` and `B` (default optimizer: `AdamW`).
For speed, this uses a frozen-prefix KV cache from past generation when
available, and computes loss/gradients on the user segment on top of that cache
(fallback: rebuild prefix cache).

It also logs token-wise CE values before averaging:

- `user_response_target_token_ids` (filtered content tokens used in loss)
- `user_response_token_losses` (filtered content-token losses)
- `user_response_token_loss_by_decoded_token` (dictionary form; repeated decoded
  tokens get `#2`, `#3`, ... suffixes)
- `user_response_raw_target_token_ids` / `user_response_raw_token_losses`
  (before filtering scaffold tokens)

Optimization objective uses only filtered content tokens with CE strictly above
the filtered-token average for that response.

## Memorization Smoke Test

Use the included script to run a simple two-session memorization check over
10 names (teach/save, then load/query):

```bash
conda run -n ai python -m scripts.memorization_smoke \
  --backend mlx_lm \
  --model google/gemma-3-1b-it
```

## Project layout

- `lmm/chat.py`: CLI REPL.
- `lmm/config.py`: runtime config.
- `lmm/engine.py`: orchestration between adapter and memory module.
- `lmm/adapters/`: model backends.
- `lmm/memory/`: memory module interface + implementations.

## Add a new memory module

1. Implement `MemoryModule.rewrite(...)` in `lmm/memory/`.
2. Register it in `lmm/memory/registry.py`.
3. Run with `--memory <name>`.

## Add a new model backend

1. Implement `ModelAdapter` in `lmm/adapters/`.
2. Register it in `lmm/adapters/registry.py`.
3. Run with `--backend <name>`.
