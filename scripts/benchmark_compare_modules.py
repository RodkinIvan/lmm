from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence


SUCCESS_RE = re.compile(r"Success rate:\s+(\d+)/(\d+)\s+\(([^)]+)%\)")


@dataclass
class Score:
    module: str
    sparse: float
    dense: float


def _parse_success_rate(output: str) -> float:
    m = SUCCESS_RE.search(output)
    if not m:
        raise RuntimeError("Could not parse success rate from output")
    return float(m.group(3))


def _run_cmd(cmd: Sequence[str], repo_root: Path) -> str:
    proc = subprocess.run(
        list(cmd),
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}"
        )
    return proc.stdout


def _run_sparse(args: argparse.Namespace, module: str, repo_root: Path) -> float:
    out = _run_cmd(
        [
            args.python,
            "-m",
            "scripts.multistep_sparse_relevance_benchmark",
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--memory",
            module,
            "--max-cases",
            str(args.sparse_cases),
            "--session1-turns",
            str(args.session1_turns),
            "--max-tokens",
            str(args.max_tokens),
            "--seed",
            str(args.seed),
        ],
        repo_root,
    )
    return _parse_success_rate(out)


def _run_dense(args: argparse.Namespace, module: str, repo_root: Path) -> float:
    out = _run_cmd(
        [
            args.python,
            "-m",
            "scripts.multistep_dense_relevance_benchmark",
            "--backend",
            args.backend,
            "--model",
            args.model,
            "--memory",
            module,
            "--max-cases",
            str(args.dense_cases),
            "--max-tokens",
            str(args.max_tokens),
            "--seed",
            str(args.seed),
        ],
        repo_root,
    )
    return _parse_success_rate(out)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Compare memory modules on sparse+dense multistep benchmarks."
    )
    p.add_argument("--python", default=sys.executable, help="Python executable")
    p.add_argument("--backend", default="mlx_lm", help="Backend")
    p.add_argument(
        "--model",
        default="mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        help="Model id/path",
    )
    p.add_argument(
        "--modules",
        default="identity,lora_on_user,hash_gradient",
        help="Comma-separated memory modules",
    )
    p.add_argument("--sparse-cases", type=int, default=5, help="Sparse scenarios count")
    p.add_argument("--dense-cases", type=int, default=2, help="Dense scenarios count")
    p.add_argument(
        "--session1-turns", type=int, default=6, help="Sparse session 1 turns"
    )
    p.add_argument("--max-tokens", type=int, default=64, help="Max generated tokens")
    p.add_argument("--seed", type=int, default=7, help="Seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    modules = [m.strip() for m in args.modules.split(",") if m.strip()]
    scores: List[Score] = []

    for module in modules:
        print(f"Running module={module}")
        sparse = _run_sparse(args, module, repo_root)
        dense = _run_dense(args, module, repo_root)
        scores.append(Score(module=module, sparse=sparse, dense=dense))

    print("\nResults:")
    print("module\tsparse\tdense\tavg")
    for score in scores:
        avg = 0.5 * (score.sparse + score.dense)
        print(f"{score.module}\t{score.sparse:.1f}\t{score.dense:.1f}\t{avg:.1f}")

    best = max(scores, key=lambda s: 0.5 * (s.sparse + s.dense))
    print(f"\nBest module: {best.module}")


if __name__ == "__main__":
    main()
