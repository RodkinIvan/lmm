from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from lmm.adapters.registry import create_adapter
from lmm.config import ChatConfig
from lmm.engine import ChatEngine
from lmm.memory.registry import create_memory_module


def parse_args() -> ChatConfig:
    parser = argparse.ArgumentParser(description="LMM CLI chat")
    parser.add_argument("--backend", default="mlx_lm", help="Model backend.")
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="Model id or local path.",
    )
    parser.add_argument(
        "--memory",
        default="identity",
        help="Memory module name.",
    )
    parser.add_argument(
        "--mm-load-path",
        default="",
        help="Path to load memory-module state.",
    )
    parser.add_argument(
        "--mm-save-path",
        default="",
        help="Path to save memory-module state on chat exit.",
    )
    parser.add_argument(
        "--mm-path",
        default="",
        help="Legacy alias: sets both --mm-load-path and --mm-save-path.",
    )
    parser.add_argument(
        "--activation-log-path",
        default="logs/memory_activations.jsonl",
        help="Path to JSONL file with memory activation slices and update events.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum generated tokens per reply.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Resolve HF model from local cache only.",
    )
    parser.add_argument(
        "--system-prompt",
        default="",
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose generation output from backend.",
    )
    args = parser.parse_args()
    mm_load_path = args.mm_load_path
    mm_save_path = args.mm_save_path
    if args.mm_path:
        if not mm_load_path:
            mm_load_path = args.mm_path
        if not mm_save_path:
            mm_save_path = args.mm_path
    return ChatConfig(
        backend=args.backend,
        model_id=args.model,
        memory_module=args.memory,
        mm_load_path=mm_load_path,
        mm_save_path=mm_save_path,
        activation_log_path=args.activation_log_path,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        local_files_only=args.local_files_only,
        system_prompt=args.system_prompt,
        verbose=args.verbose,
    )


def _default_mm_save_path(module_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{module_name}_{ts}.safetensors"


def run_chat(config: ChatConfig) -> None:
    memory_module = create_memory_module(config.memory_module)
    adapter = create_adapter(
        backend=config.backend,
        model_id=config.model_id,
        memory_module=memory_module,
        local_files_only=config.local_files_only,
        verbose=config.verbose,
        activation_log_path=config.activation_log_path,
    )
    engine = ChatEngine(
        adapter=adapter,
        system_prompt=config.system_prompt,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        top_p=config.top_p,
    )

    print("Loading model backend...")
    try:
        adapter.preload()
        print("Model backend ready.")
    except Exception as exc:
        print(f"Model preload failed: {exc}")
        print("Continuing. The app will retry loading on the first request.")
    if config.mm_load_path:
        load_path = Path(config.mm_load_path)
        if not load_path.exists():
            print(f"Memory load skipped: file not found ({load_path})")
        else:
            try:
                result = adapter.load_memory(str(load_path))
                print(f"Memory load: {result}")
            except Exception as exc:
                print(f"Memory load failed: {exc}")
    elif config.mm_save_path:
        print("Memory load skipped: no --mm-load-path provided")

    mm_save_path = config.mm_save_path.strip()
    if not mm_save_path:
        mm_save_path = _default_mm_save_path(config.memory_module)
        print(f"Memory save path defaulted to: {mm_save_path}")

    print("LMM chat started. Type 'exit' or 'quit' to stop.")
    print(
        f"backend={config.backend} model={config.model_id} memory={config.memory_module}"
    )

    exit_requested = False
    save_on_exit = True
    while True:
        try:
            user_text = input("you> ").strip()
        except EOFError:
            print("")
            exit_requested = True
            break
        except KeyboardInterrupt:
            print("")
            exit_requested = True
            save_on_exit = False
            break
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            exit_requested = True
            break
        try:
            reply = engine.ask(user_text)
        except Exception as exc:
            print(f"error> {exc}")
            continue
        print(f"assistant> {reply}")

    if save_on_exit:
        try:
            result = adapter.save_memory(mm_save_path)
            print(f"Memory save: {result}")
        except Exception as exc:
            print(f"Memory save failed: {exc}")
    else:
        print("Memory save skipped (KeyboardInterrupt).")
    if exit_requested:
        print("bye")


def main() -> None:
    config = parse_args()
    run_chat(config)


if __name__ == "__main__":
    main()
