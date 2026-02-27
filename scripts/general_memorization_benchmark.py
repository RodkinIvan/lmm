from __future__ import annotations

import argparse
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from lmm.adapters.registry import create_adapter
from lmm.engine import ChatEngine
from lmm.memory.registry import create_memory_module


@dataclass
class Scenario:
    name: str
    teach: str
    query: str
    expected_keywords: Sequence[str]


DEFAULT_SCENARIOS: List[Scenario] = [
    Scenario(
        name="identity_location",
        teach="My name is Elena and I live in Porto.",
        query="What city do I live in?",
        expected_keywords=["porto"],
    ),
    Scenario(
        name="work",
        teach="I work as a robotics engineer at Delta Dynamics.",
        query="What is my profession?",
        expected_keywords=["robotics", "engineer"],
    ),
    Scenario(
        name="constraints",
        teach="I am vegetarian and allergic to peanuts.",
        query="What dietary constraints did I mention?",
        expected_keywords=["vegetarian", "peanut"],
    ),
    Scenario(
        name="preference",
        teach="I prefer concise bullet-point answers and dislike long introductions.",
        query="How should you format your answers for me?",
        expected_keywords=["concise", "bullet"],
    ),
    Scenario(
        name="project_plan",
        teach="I need to finish the memory benchmark by Friday and share results with my team.",
        query="What deadline did I mention?",
        expected_keywords=["friday"],
    ),
]


DISTRACTORS = [
    "Summarize quicksort in one sentence.",
    "Give two pros and two cons of microservices.",
    "How does caching reduce latency?",
    "Explain what a race condition is.",
    "What does idempotency mean in APIs?",
]


def make_engine(
    *,
    backend: str,
    model_id: str,
    memory_name: str,
    local_files_only: bool,
    max_tokens: int,
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
    engine = ChatEngine(
        adapter=adapter,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return engine, adapter


def run_scenario(
    *,
    scenario: Scenario,
    backend: str,
    model_id: str,
    memory_name: str,
    local_files_only: bool,
    max_tokens: int,
    distractor_turns: int,
    workdir: Path,
) -> bool:
    state_path = str(workdir / f"{memory_name}_{scenario.name}.safetensors")

    engine_1, adapter_1 = make_engine(
        backend=backend,
        model_id=model_id,
        memory_name=memory_name,
        local_files_only=local_files_only,
        max_tokens=max_tokens,
    )
    _ = engine_1.ask(scenario.teach)
    for idx in range(max(0, distractor_turns)):
        _ = engine_1.ask(DISTRACTORS[idx % len(DISTRACTORS)])

    save_result = adapter_1.save_memory(state_path)
    if not save_result.get("saved", False):
        print(f"FAIL | {scenario.name} | save={save_result}")
        return False

    engine_2, adapter_2 = make_engine(
        backend=backend,
        model_id=model_id,
        memory_name=memory_name,
        local_files_only=local_files_only,
        max_tokens=max_tokens,
    )
    load_result = adapter_2.load_memory(state_path)
    if not load_result.get("loaded", False):
        print(f"FAIL | {scenario.name} | load={load_result}")
        return False

    answer = engine_2.ask(scenario.query)
    normalized = answer.lower()
    success = any(k.lower() in normalized for k in scenario.expected_keywords)
    verdict = "PASS" if success else "FAIL"
    print(f"{verdict} | {scenario.name} | query={scenario.query!r} | answer={answer!r}")
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="General two-session memorization benchmark for memory modules."
    )
    parser.add_argument("--backend", default="mlx_lm", help="Model backend.")
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="Model id or local path.",
    )
    parser.add_argument(
        "--memory",
        default="lora_on_user",
        help="Memory module name from lmm.memory.registry.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Maximum generated tokens per reply.",
    )
    parser.add_argument(
        "--distractor-turns",
        type=int,
        default=2,
        help="Number of unrelated turns inserted after teaching.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve model from local cache only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    successes = 0
    with tempfile.TemporaryDirectory(prefix="lmm_general_mem_bench_") as tmp:
        workdir = Path(tmp)
        for scenario in DEFAULT_SCENARIOS:
            try:
                ok = run_scenario(
                    scenario=scenario,
                    backend=args.backend,
                    model_id=args.model,
                    memory_name=args.memory,
                    local_files_only=args.local_files_only,
                    max_tokens=args.max_tokens,
                    distractor_turns=args.distractor_turns,
                    workdir=workdir,
                )
            except Exception as exc:
                ok = False
                print(f"FAIL | {scenario.name} | error={exc}")
            successes += 1 if ok else 0

    total = len(DEFAULT_SCENARIOS)
    rate = 100.0 * successes / total
    print(f"Success rate: {successes}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
