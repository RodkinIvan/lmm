from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any, List

from lmm.adapters.registry import create_adapter
from lmm.engine import ChatEngine
from lmm.memory.registry import create_memory_module


DEFAULT_NAMES: List[str] = [
    "Alice",
    "Bob",
    "Carol",
    "David",
    "Eva",
    "Ivan",
]

SYSTEM_PROMPT = (
    "You are talking with the same user across sessions. "
    "When asked about earlier user-provided personal facts, answer directly."
)

QUESTION_PROMPT = "In our previous session, what is my name? Reply with only the name."
SMOKE_QUESTION_PROMPT = (
    "What is my name? If you think you don't know, just try starting and may be "
    'you will guess: "User\'s name is..."'
)
TEACH_SIMPLE = "My name is {name}."
TEACH_SMOKE = "I am {name}."


def make_engine(
    *,
    model_id: str,
    backend: str,
    memory_name: str,
    max_tokens: int,
    local_files_only: bool,
    mm_load_path: str,
    system_prompt: str,
) -> tuple[ChatEngine, Any]:
    memory_module = create_memory_module(memory_name)
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
        _ = adapter.load_memory(mm_load_path)
    engine = ChatEngine(
        adapter=adapter,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return engine, adapter


def run_trial(
    *,
    name: str,
    model_id: str,
    backend: str,
    memory_name: str,
    max_tokens: int,
    local_files_only: bool,
    workdir: Path,
    question_prompt: str,
    system_prompt: str,
    teach_template: str,
) -> bool:
    mm_path = str(workdir / f"{memory_name}_{name.lower()}_state.safetensors")

    engine1, adapter1 = make_engine(
        model_id=model_id,
        backend=backend,
        memory_name=memory_name,
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        mm_load_path="",
        system_prompt=system_prompt,
    )
    _ = engine1.ask(teach_template.format(name=name))
    save_result = adapter1.save_memory(mm_path)
    if not save_result.get("saved", False):
        print(f"FAIL | name={name} | save={save_result}")
        return False

    engine2, _ = make_engine(
        model_id=model_id,
        backend=backend,
        memory_name=memory_name,
        max_tokens=max_tokens,
        local_files_only=local_files_only,
        mm_load_path=mm_path,
        system_prompt=system_prompt,
    )
    answer = engine2.ask(question_prompt)
    ok = name.lower() in answer.lower()
    verdict = "PASS" if ok else "FAIL"
    print(f"{verdict} | name={name} | answer={answer!r}")
    return ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple 2-session one-fact name benchmark for any memory module."
    )
    parser.add_argument("--backend", default="mlx_lm", help="Model backend.")
    parser.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model id or local path.",
    )
    parser.add_argument(
        "--memory",
        default="lora_on_user",
        help="Memory module name from lmm.memory.registry.",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32, help="Max generated tokens."
    )
    parser.add_argument(
        "--names",
        default=",".join(DEFAULT_NAMES),
        help="Comma-separated names.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve model from local cache only.",
    )
    parser.add_argument(
        "--question-style",
        default="smoke",
        choices=["strict", "smoke"],
        help="Question prompt style.",
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="Enable benchmark system prompt (disabled by default for memorization_smoke parity).",
    )
    parser.add_argument(
        "--teach-style",
        default="smoke",
        choices=["simple", "smoke"],
        help="Teaching utterance style.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = [n.strip() for n in args.names.split(",") if n.strip()]
    if not names:
        raise SystemExit("No names provided")

    question_prompt = (
        QUESTION_PROMPT if args.question_style == "strict" else SMOKE_QUESTION_PROMPT
    )
    system_prompt = SYSTEM_PROMPT if args.use_system_prompt else ""
    teach_template = TEACH_SMOKE if args.teach_style == "smoke" else TEACH_SIMPLE

    successes = 0
    with tempfile.TemporaryDirectory(prefix="lmm_single_fact_name_") as tmp:
        workdir = Path(tmp)
        for name in names:
            try:
                ok = run_trial(
                    name=name,
                    model_id=args.model,
                    backend=args.backend,
                    memory_name=args.memory,
                    max_tokens=args.max_tokens,
                    local_files_only=args.local_files_only,
                    workdir=workdir,
                    question_prompt=question_prompt,
                    system_prompt=system_prompt,
                    teach_template=teach_template,
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
