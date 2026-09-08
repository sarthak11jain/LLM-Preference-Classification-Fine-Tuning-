"""Fail when common private artifacts appear in the publishable tree."""

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", "_kaggle_cli_audit", "docs", "notebooks"}
SKIP_FILES = {Path("experiments/artifact-inventory.csv")}
PATTERNS = [
    re.compile(r"(?i)kaggle\.json|api[_-]?token|access[_-]?token"),
    re.compile(r"(?i)[A-Z]:\\[^\r\n]{2,}\\Users\\"),
    re.compile(r"(?i)/kaggle/(input|working)"),
]
FORBIDDEN_SUFFIXES = {".safetensors", ".ckpt", ".bin", ".pt", ".pth"}


def files_to_scan():
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.relative_to(ROOT) in SKIP_FILES:
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            yield path, "forbidden model artifact"
            continue
        if path.stat().st_size > 25 * 1024 * 1024:
            yield path, "file exceeds 25 MiB publication limit"
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in PATTERNS:
            if pattern.search(text):
                yield path, f"matched {pattern.pattern}"
                break


def main() -> int:
    findings = list(files_to_scan())
    if findings:
        for path, reason in findings:
            print(f"{path.relative_to(ROOT)}: {reason}")
        return 1
    print("Public repository audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
