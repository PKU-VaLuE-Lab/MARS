#!/usr/bin/env python3
"""Tee a protected FastWAM run to its durable result log."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import TextIO


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "experiments/libero/eval_libero_all.py"


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(stream.isatty() for stream in self.streams)

    def fileno(self) -> int:
        return self.streams[0].fileno()

    @property
    def encoding(self) -> str:
        return self.streams[0].encoding


def main() -> None:
    log_path = Path(os.environ["FASTWAM_RESULT_LOG"]).resolve()
    log_path.relative_to(ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("w", encoding="utf-8", buffering=1) as result_stream:
        sys.stdout = _Tee(original_stdout, result_stream)
        sys.stderr = _Tee(original_stderr, result_stream)
        try:
            runpy.run_path(str(ENTRY), run_name="__main__")
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
