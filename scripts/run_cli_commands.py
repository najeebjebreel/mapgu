#!/usr/bin/env python3
"""
Run the documented commands in cli_commands.txt sequentially.

Usage:
  python scripts/run_cli_commands.py
  python scripts/run_cli_commands.py --dry-run
  python scripts/run_cli_commands.py --continue-on-error
  python scripts/run_cli_commands.py --match certified_sp --dry-run
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CommandSpec:
    line_no: int
    text: str


def _parse_python_command(command: str) -> List[str]:
    last_error: Exception | None = None
    for posix in (True, False):
        try:
            argv = shlex.split(command, posix=posix)
        except ValueError as exc:
            last_error = exc
            continue
        if argv and argv[0] == "python":
            argv[0] = sys.executable
            return argv
    raise ValueError(f"Unsupported command format: {command!r}") from last_error


def load_commands(path: Path) -> List[CommandSpec]:
    commands: List[CommandSpec] = []
    buffer: List[str] = []
    start_line: int | None = None

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if start_line is None:
            start_line = line_no

        continued = stripped.endswith("\\")
        piece = stripped[:-1].rstrip() if continued else stripped
        buffer.append(piece)

        if continued:
            continue

        command = " ".join(buffer).strip()
        buffer.clear()

        if command.startswith("python "):
            commands.append(CommandSpec(line_no=start_line, text=command))

        start_line = None

    if buffer:
        raise ValueError(f"Unterminated continued command starting at line {start_line}.")

    return commands


def filter_commands(commands: List[CommandSpec], matches: List[str]) -> List[CommandSpec]:
    if not matches:
        return commands
    needles = [m.lower() for m in matches]
    return [cmd for cmd in commands if any(needle in cmd.text.lower() for needle in needles)]


def build_parser() -> argparse.ArgumentParser:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Run the commands listed in cli_commands.txt in sequence."
    )
    parser.add_argument(
        "--commands-file",
        type=Path,
        default=repo_root / "cli_commands.txt",
        help="Path to the command list to execute.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Keep going after a command fails.",
    )
    parser.add_argument(
        "--match",
        action="append",
        default=[],
        help="Only run commands containing this substring. Repeatable.",
    )
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    commands_file = args.commands_file.resolve()
    repo_root = Path(__file__).resolve().parents[1]
    commands = filter_commands(load_commands(commands_file), args.match)

    if not commands:
        print("No commands matched.", file=sys.stderr)
        return 1

    print(f"Loaded {len(commands)} commands from {commands_file}")

    failures = 0
    for idx, spec in enumerate(commands, start=1):
        print(f"\n[{idx}/{len(commands)}] line {spec.line_no}")
        print(spec.text)

        if args.dry_run:
            continue

        argv_cmd = _parse_python_command(spec.text)
        completed = subprocess.run(argv_cmd, cwd=repo_root)
        if completed.returncode == 0:
            continue

        failures += 1
        print(
            f"Command failed with exit code {completed.returncode}: line {spec.line_no}",
            file=sys.stderr,
        )
        if not args.continue_on_error:
            return completed.returncode

    if failures:
        print(f"\nCompleted with {failures} failing command(s).", file=sys.stderr)
        return 1

    print("\nAll commands completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
