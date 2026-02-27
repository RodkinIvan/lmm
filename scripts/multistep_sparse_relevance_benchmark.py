from __future__ import annotations

import argparse
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Sequence

from lmm.adapters.registry import create_adapter
from lmm.engine import ChatEngine
from lmm.memory.registry import create_memory_module


@dataclass
class SparseScenario:
    name: str
    fact_text: str
    query_text: str
    expected_keywords: Sequence[str]


SCENARIOS: List[SparseScenario] = [
    SparseScenario(
        name="location",
        fact_text="I moved to Porto last year and now I live there.",
        query_text="In our previous session, what city did I say I live in? Reply with only the city.",
        expected_keywords=["porto"],
    ),
    SparseScenario(
        name="work",
        fact_text="I work as a robotics engineer at Delta Dynamics.",
        query_text="In our previous session, what job did I tell you I have? Reply concisely.",
        expected_keywords=["robotics", "engineer"],
    ),
    SparseScenario(
        name="constraint",
        fact_text="I am allergic to peanuts and I avoid them completely.",
        query_text="In our previous session, what allergy did I mention? Reply with only the allergy.",
        expected_keywords=["peanut"],
    ),
    SparseScenario(
        name="preference",
        fact_text="I prefer concise bullet-point responses.",
        query_text="In our previous session, how did I ask you to format responses? Reply concisely.",
        expected_keywords=["concise", "bullet"],
    ),
    SparseScenario(
        name="deadline",
        fact_text="My deadline for the benchmark report is Friday.",
        query_text="In our previous session, what deadline did I mention? Reply with only the deadline.",
        expected_keywords=["friday"],
    ),
]


DISTRACTOR_MESSAGES = [
    "Can you explain what memoization is in dynamic programming?",
    "Give me one short summary of TCP vs UDP.",
    "What are common causes of flaky tests?",
    "How does top-p sampling differ from top-k sampling?",
    "List two ways to reduce API latency.",
    "What is the difference between threads and async?",
    "How do hash tables handle collisions?",
    "Why are idempotent endpoints useful?",
    "What is eventual consistency in distributed systems?",
]

SYSTEM_PROMPT = (
    "You are talking with the same user across sessions. "
    "When the user asks about previously stated personal facts, answer directly "
    "and concisely from remembered user information. "
    "Do not answer about your own identity as an AI model."
)


def make_engine(
    *,
    backend: str,
    model_id: str,
    memory_name: str,
    local_files_only: bool,
    max_tokens: int,
    init_memory_path: str,
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
    if init_memory_path:
        _ = adapter.load_memory(init_memory_path)
    engine = ChatEngine(
        adapter=adapter,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=max_tokens,
        temperature=0.0,
        top_p=1.0,
    )
    return engine, adapter


def make_session_messages(
    *,
    fact_text: str,
    total_turns: int,
    rng: random.Random,
) -> tuple[List[str], int]:
    turns = max(2, int(total_turns))
    insert_at = rng.randrange(turns)
    out: List[str] = []
    for idx in range(turns):
        if idx == insert_at:
            out.append(fact_text)
        else:
            out.append(rng.choice(DISTRACTOR_MESSAGES))
    return out, insert_at


def run_case(
    *,
    scenario: SparseScenario,
    backend: str,
    model_id: str,
    memory_name: str,
    local_files_only: bool,
    max_tokens: int,
    session1_turns: int,
    rng: random.Random,
    workdir: Path,
    init_memory_path: str,
) -> bool:
    case_state = str(workdir / f"sparse_{memory_name}_{scenario.name}.safetensors")

    session_messages, insert_at = make_session_messages(
        fact_text=scenario.fact_text,
        total_turns=session1_turns,
        rng=rng,
    )

    engine_1, adapter_1 = make_engine(
        backend=backend,
        model_id=model_id,
        memory_name=memory_name,
        local_files_only=local_files_only,
        max_tokens=max_tokens,
        init_memory_path=init_memory_path,
    )
    for user_text in session_messages:
        _ = engine_1.ask(user_text)

    save_result = adapter_1.save_memory(case_state)
    if not save_result.get("saved", False):
        print(f"FAIL | {scenario.name} | save={save_result}")
        return False

    engine_2, adapter_2 = make_engine(
        backend=backend,
        model_id=model_id,
        memory_name=memory_name,
        local_files_only=local_files_only,
        max_tokens=max_tokens,
        init_memory_path=init_memory_path,
    )
    load_result = adapter_2.load_memory(case_state)
    if not load_result.get("loaded", False):
        print(f"FAIL | {scenario.name} | load={load_result}")
        return False

    answer = engine_2.ask(scenario.query_text)
    normalized = answer.lower()
    success = any(k.lower() in normalized for k in scenario.expected_keywords)
    verdict = "PASS" if success else "FAIL"
    print(
        f"{verdict} | {scenario.name} | fact_turn={insert_at + 1}/{len(session_messages)} | "
        f"query={scenario.query_text!r} | answer={answer!r}"
    )
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark #1: multi-step sparse relevance. One relevant user fact is "
            "embedded at a random turn among distractors, then queried in a new session."
        )
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
        "--session1-turns",
        type=int,
        default=6,
        help="Number of user-assistant cycles in session 1.",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=5,
        help="How many scenarios to run from the built-in set.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve model from local cache only.",
    )
    parser.add_argument(
        "--init-memory",
        default="",
        help="Optional path to preloaded memory state before each case.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    scenarios = SCENARIOS[: max(1, min(args.max_cases, len(SCENARIOS)))]

    successes = 0
    with tempfile.TemporaryDirectory(prefix="lmm_sparse_multistep_") as tmp:
        workdir = Path(tmp)
        for scenario in scenarios:
            try:
                ok = run_case(
                    scenario=scenario,
                    backend=args.backend,
                    model_id=args.model,
                    memory_name=args.memory,
                    local_files_only=args.local_files_only,
                    max_tokens=args.max_tokens,
                    session1_turns=args.session1_turns,
                    rng=rng,
                    workdir=workdir,
                    init_memory_path=args.init_memory,
                )
            except Exception as exc:
                ok = False
                print(f"FAIL | {scenario.name} | error={exc}")
            successes += 1 if ok else 0

    total = len(scenarios)
    rate = 100.0 * successes / total
    print(f"Success rate: {successes}/{total} ({rate:.1f}%)")


if __name__ == "__main__":
    main()
