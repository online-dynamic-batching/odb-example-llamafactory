#!/usr/bin/env python3
"""Apply the LLaMA-Factory hook patch required by this example."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PATCH = ROOT / "patches" / "llamafactory_enable_odb.patch"


def _workflow_has_hook(llamafactory_root: Path) -> bool:
    workflow = (
        llamafactory_root / "src" / "llamafactory" / "train" / "sft" / "workflow.py"
    )
    if not workflow.exists():
        return False
    text = workflow.read_text(encoding="utf-8")
    return (
        "odb.integrations.llamafactory import enable_odb" in text
        and "enable_odb(" in text
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llamafactory-root", type=Path, required=True)
    args = parser.parse_args()

    root = args.llamafactory_root.resolve()
    if _workflow_has_hook(root):
        print(f"enable_odb hook already present in {root}")
        return
    if not PATCH.exists():
        raise SystemExit(f"patch file not found: {PATCH}")

    check = subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=root)
    if check.returncode != 0:
        raise SystemExit(
            "Patch does not apply cleanly. Use an ODB/TMDB-capable LLaMA-Factory checkout "
            "or inspect patches/llamafactory_enable_odb.patch."
        )

    subprocess.run(["git", "apply", str(PATCH)], cwd=root, check=True)
    if not _workflow_has_hook(root):
        raise SystemExit("patch applied, but enable_odb hook was not detected")
    print(f"Applied enable_odb hook patch to {root}")


if __name__ == "__main__":
    main()
