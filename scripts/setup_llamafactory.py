#!/usr/bin/env python3
"""Prepare a pinned ODB-enabled LLaMA-Factory checkout for this example."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "https://github.com/hiyouga/LLaMA-Factory.git"
PINNED_COMMIT = "ca50f22c38a77e72a4a21ef177ce4aa8f29d6930"
PINNED_PATCH = ROOT / "patches" / "llamafactory_odb_integration_ca50f22c.patch"
EXTRA_TREE = ROOT / "patches" / "llamafactory_extra"


def _run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd is not None else None, check=True)


def _contains(path: Path, needle: str) -> bool:
    if not path.exists():
        return False
    return needle in path.read_text(encoding="utf-8", errors="ignore")


def _has_main_hooks(root: Path) -> bool:
    return all(
        [
            (root / "src" / "llamafactory" / "launcher.py").exists(),
            (root / "src" / "llamafactory" / "data" / "qmdb" / "tmdb.py").exists(),
            (root / "src" / "llamafactory" / "data" / "lazy" / "loader.py").exists(),
            _contains(
                root / "src" / "llamafactory" / "data" / "parser.py",
                "tmdb_file",
            ),
            _contains(
                root / "src" / "llamafactory" / "train" / "sft" / "workflow.py",
                "odb.integrations.llamafactory import enable_odb",
            ),
        ]
    )


def _has_extra_files(root: Path) -> bool:
    return all(
        [
            (root / "src" / "llamafactory" / "data" / "dataloaders" / "__init__.py").exists(),
            _contains(
                root / "src" / "llamafactory" / "data" / "dataloaders" / "__init__.py",
                "def apply_odb",
            ),
        ]
    )


def _is_ready(root: Path) -> bool:
    return _has_main_hooks(root) and _has_extra_files(root)


def _ensure_checkout(target: Path, repo: str) -> None:
    if target.exists() and not (target / ".git").exists():
        raise RuntimeError(f"{target} exists but is not a git checkout")
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", repo, str(target)])
    _run(["git", "fetch", "--all", "--tags"], cwd=target)
    _run(["git", "checkout", PINNED_COMMIT], cwd=target)


def _copy_extra_files(target: Path) -> None:
    if not EXTRA_TREE.exists():
        return
    for source in EXTRA_TREE.rglob("*"):
        if source.is_dir():
            continue
        relative = source.relative_to(EXTRA_TREE)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _apply_patch(target: Path) -> None:
    if _is_ready(target):
        print(f"LLaMA-Factory checkout already has ODB hooks: {target}", flush=True)
        return
    if _has_main_hooks(target):
        _copy_extra_files(target)
        if _is_ready(target):
            print(f"LLaMA-Factory checkout had ODB hooks; copied extra files: {target}", flush=True)
            return
    if not PINNED_PATCH.exists():
        raise FileNotFoundError(PINNED_PATCH)
    _run(["git", "apply", "--check", str(PINNED_PATCH)], cwd=target)
    _run(["git", "apply", str(PINNED_PATCH)], cwd=target)
    _copy_extra_files(target)
    if not _is_ready(target):
        raise RuntimeError("patch applied, but ODB integration hooks were not detected")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        default=ROOT / ".deps" / "LLaMA-Factory-odb",
        help="Where to create the compatible LLaMA-Factory checkout.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Upstream LLaMA-Factory repository to clone.",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Run pip install -e on the prepared checkout.",
    )
    args = parser.parse_args()

    target = args.target.expanduser().resolve()
    _ensure_checkout(target, args.repo)
    _apply_patch(target)
    if args.install:
        _run([sys.executable, "-m", "pip", "install", "-e", str(target)])

    print("\nODB-enabled LLaMA-Factory checkout is ready.")
    print(f"export LLAMAFACTORY_ROOT={target}")


if __name__ == "__main__":
    main()
