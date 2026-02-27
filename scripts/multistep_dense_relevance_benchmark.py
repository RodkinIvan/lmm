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
class RelevantFact:
    text: str
    query: str
    expected_keywords: Sequence[str]


@dataclass
class DenseScenario:
    name: str
    facts: Sequence[RelevantFact]


SCENARIOS: List[DenseScenario] = [
    DenseScenario(
        name="profile_a",
        facts=[
            RelevantFact(
                text="My name is Elena.",
                query="In our previous session, what name did I tell you? Reply with only the name.",
                expected_keywords=["elena"],
            ),
            RelevantFact(
                text="I live in Porto.",
                query="In our previous session, what city did I say I live in? Reply with only the city.",
                expected_keywords=["porto"],
            ),
            RelevantFact(
                text="I work as a robotics engineer.",
                query="In our previous session, what profession did I mention? Reply concisely.",
                expected_keywords=["robotics", "engineer"],
            ),
            RelevantFact(
                text="I am allergic to peanuts.",
                query="In our previous session, what allergy did I mention? Reply with only the allergy.",
                expected_keywords=["peanut"],
            ),
            RelevantFact(
                text="My deadline is Friday.",
                query="In our previous session, what deadline did I mention? Reply with only the deadline.",
                expected_keywords=["friday"],
            ),
        ],
    ),
    DenseScenario(
        name="profile_b",
        facts=[
            RelevantFact(
                text="My name is Amir.",
                query="In our previous session, what name did I tell you? Reply with only the name.",
                expected_keywords=["amir"],
            ),
            RelevantFact(
                text="I recently moved to Berlin.",
                query="In our previous session, where did I say I moved? Reply with only the city.",
                expected_keywords=["berlin"],
            ),
            RelevantFact(
                text="I prefer concise bullet-point replies.",
                query="In our previous session, how did I ask you to format answers? Reply concisely.",
                expected_keywords=["concise", "bullet"],
            ),
            RelevantFact(
                text="I am vegetarian.",
                query="In our previous session, what diet did I mention? Reply with only the diet.",
                expected_keywords=["vegetarian"],
            ),
            RelevantFact(
                text="I enjoy trail running on weekends.",
                query="In our previous session, what hobby did I mention? Reply concisely.",
                expected_keywords=["running", "trail"],
            ),
        ],
    ),
]

SYSTEM_PROMPT = (
    "You are talking with the same user across sessions. "
    "When asked about earlier user-provided facts, answer directly and concisely "
    "from remembered user information. "
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


def run_case(
    *,
    scenario: DenseScenario,
    backend: str,
    model_id: str,
    memory_name: str,
    local_files_only: bool,
    max_tokens: int,
    rng: random.Random,
    workdir: Path,
    init_memory_path: str,
) -> bool:
    case_state = str(workdir / f"dense_{memory_name}_{scenario.name}.safetensors")

    ordered_facts = list(scenario.facts)
    rng.shuffle(ordered_facts)
    target_fact = ordered_facts[rng.randrange(len(ordered_facts))]

    engine_1, adapter_1 = make_engine(
        backend=backend,
        model_id=model_id,
        memory_name=memory_name,
        local_files_only=local_files_only,
        max_tokens=max_tokens,
        init_memory_path=init_memory_path,
    )
    for fact in ordered_facts:
        _ = engine_1.ask(fact.text)

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

    answer = engine_2.ask(target_fact.query)
    normalized = answer.lower()
    success = any(k.lower() in normalized for k in target_fact.expected_keywords)
    verdict = "PASS" if success else "FAIL"
    print(
        f"{verdict} | {scenario.name} | target_query={target_fact.query!r} | "
        f"answer={answer!r}"
    )
    return success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark #2: multi-step dense relevance. All user messages in session 1 "
            "are relevant, then session 2 asks about a random prior user message."
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
        "--max-cases",
        type=int,
        default=2,
        help="How many built-in scenarios to run.",
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
    with tempfile.TemporaryDirectory(prefix="lmm_dense_multistep_") as tmp:
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
