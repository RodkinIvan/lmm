from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import List

from lmm.adapters.registry import create_adapter
from lmm.engine import ChatEngine
from lmm.memory.lora_on_user import LoraOnUserMemoryModule


DEFAULT_NAMES: List[str] = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eva",
    "Ivan",
    "Yuri",
    "Aydar",
    "Mikhail",
    "Matvey",
]

QUESTION_PROMPT = (
    'What is my name? If you think you don\'t know, just try starting and may be '
    'you will guess: \"User\'s name is...\"'
)


def make_engine(
    *,
    model_id: str,
    mm_load_path: str,
    mm_save_path: str,
    max_tokens: int,
    local_files_only: bool,
    backend: str,
    lora_rank: int,
    optimizer: str,
    learning_rate: float,
    optimization_steps: int,
    l2_regularization: float,
) -> tuple[ChatEngine, object]:
    memory_module = LoraOnUserMemoryModule(
        rank=lora_rank,
        optimizer_name=optimizer,
        learning_rate=learning_rate,
        optimization_steps=optimization_steps,
        l2_regularization=l2_regularization,
    )
    adapter = create_adapter(
        backend=backend,
        model_id=model_id,
        memory_module=memory_module,
        local_files_only=local_files_only,
        verbose=False,
        activation_log_path="logs/memory_activations.jsonl",
        deterministic=True,
        decoding_strategy="greedy",
    )
    adapter.preload()
    if mm_load_path:
        result = adapter.load_memory(mm_load_path)
        if not result.get("loaded", False):
            print(f"[warn] memory load failed/skipped: {result}")
    engine = ChatEngine(
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return engine, adapter


def run_name_trial(
    *,
    name: str,
    model_id: str,
    backend: str,
    max_tokens: int,
    local_files_only: bool,
    workdir: Path,
    lora_rank: int,
    optimizer: str,
    learning_rate: float,
    optimization_steps: int,
    l2_regularization: float,
) -> bool:
    mm_path = str(workdir / f"{name.lower()}_memory.safetensors")

    # Session 1: teach and save.
    engine1, adapter1 = make_engine(
        model_id=model_id,
        mm_load_path="",
        mm_save_path=mm_path,
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        backend=backend,
        lora_rank=lora_rank,
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimization_steps=optimization_steps,
        l2_regularization=l2_regularization,
    )
    _ = engine1.ask(f"I am {name}.")
    save_result = adapter1.save_memory(mm_path)
    if not save_result.get("saved", False):
        print(f"[warn] {name}: save failed: {save_result}")
        return False

    # Session 2: load and query.
    engine2, _ = make_engine(
        model_id=model_id,
        mm_load_path=mm_path,
        mm_save_path="",
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        backend=backend,
        lora_rank=lora_rank,
        optimizer=optimizer,
        learning_rate=learning_rate,
        optimization_steps=optimization_steps,
        l2_regularization=l2_regularization,
    )
    answer = engine2.ask(QUESTION_PROMPT)
    success = name.lower() in answer.lower()
    verdict = "PASS" if success else "FAIL"
    print(f"{verdict} | name={name} | answer={answer!r}")
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple 2-session memorization smoke test for lora_on_user memory module."
    )
    parser.add_argument(
        "--backend",
        default="mlx_lm",
        help="Model backend.",
    )
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="Model id or local path.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per reply.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve HF model from local cache only.",
    )
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_NAMES),
        help="Comma-separated names to test.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for lora_on_user optimizer.",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        choices=["adamw", "adam", "sgd"],
        help="Optimizer for lora_on_user.",
    )
    parser.add_argument(
        "--lora-rank",
        type=int,
        default=8,
        help="LoRA rank for lora_on_user.",
    )
    parser.add_argument(
        "--optimization-steps",
        type=int,
        default=1,
        help="Number of optimization steps per user message.",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=0.0,
        help="L2 penalty coefficient added to CE loss for A and B.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if not names:
        raise SystemExit("No names provided.")

    successes = 0
    with tempfile.TemporaryDirectory(prefix="lmm_mem_test_") as tmp:
        workdir = Path(tmp)
        for name in names:
            try:
                ok = run_name_trial(
                    name=name,
                    model_id=args.model,
                    backend=args.backend,
                    max_tokens=args.max_tokens,
                    local_files_only=args.local_files_only,
                    workdir=workdir,
                    lora_rank=args.lora_rank,
                    optimizer=args.optimizer,
                    learning_rate=args.learning_rate,
                    optimization_steps=args.optimization_steps,
                    l2_regularization=args.l2_regularization,
                )
            except Exception as exc:
                ok = False
                print(f"FAIL | name={name} | error={exc}")
            successes += 1 if ok else 0

    total = len(names)
    rate = 100.0 * successes / total
    print(f"Success rate: {successes}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
