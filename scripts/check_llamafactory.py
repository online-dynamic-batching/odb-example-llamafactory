#!/usr/bin/env python3
"""Check whether a LLaMA-Factory checkout is ready for this ODB example."""

from __future__ import annotations

import argparse
from pathlib import Path


def _contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamafactory-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.llamafactory_root.expanduser().resolve()
    checks = [
        (
            "LLaMA-Factory launcher",
            root / "src" / "llamafactory" / "launcher.py",
            None,
        ),
        (
            "TMDB writer",
            root / "src" / "llamafactory" / "data" / "qmdb" / "tmdb.py",
            None,
        ),
        (
            "lazy tensor loader",
            root / "src" / "llamafactory" / "data" / "lazy" / "loader.py",
            None,
        ),
        (
            "TMDB dataset parser",
            root / "src" / "llamafactory" / "data" / "parser.py",
            "tmdb_file",
        ),
        (
            "ODB enable hook",
            root / "src" / "llamafactory" / "train" / "sft" / "workflow.py",
            "odb.integrations.llamafactory import enable_odb",
        ),
    ]

    failures: list[str] = []
    for label, path, needle in checks:
        if needle is None:
            ok = path.exists()
        else:
            ok = _contains(path, needle)
        status = "OK" if ok else "MISSING"
        print(f"[{status}] {label}: {path}")
        if not ok:
            failures.append(label)

    if failures:
        raise SystemExit(
            "\nThis checkout is not ready for odb-example-llamafactory. "
            "Use an ODB-enabled LLaMA-Factory checkout, or port the integration "
            "shown in patches/llamafactory_enable_odb.patch."
        )

    print("\nLLaMA-Factory checkout is ready for this ODB example.")


if __name__ == "__main__":
    main()
