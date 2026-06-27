#!/usr/bin/env python3
"""Evaluate LLaMA-Factory MM-Mix checkpoints.

The validation-loss path uses LLaMA-Factory itself. The MMMU-MC path uses this
repository's built-in choice-likelihood evaluator by default.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = Path(
    os.environ.get("ODB_MM_MIX_RUN_DIR", ROOT / "data/llamafactory-mm-mix")
)
DEFAULT_TRAIN_ROOT = Path(
    os.environ.get("ODB_MM_MIX_TRAIN_ROOT", ROOT / "outputs/llamafactory-mm-mix")
)
DEFAULT_PREFIX = os.environ.get("ODB_MM_MIX_PROJECT_PREFIX", "public_mmmix_lf")
DEFAULT_MMMU_SCRIPT = ROOT / "scripts" / "eval_benchmark.py"


def _raise_fd_limit() -> None:
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def _setup_env(*, lf_root: Path, cuda_visible_devices: str) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", cuda_visible_devices)
    env.setdefault("DISABLE_VERSION_CHECK", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MMMU_EVAL_SINGLE_DEVICE_LOAD", "1")
    env.setdefault("PYTHONFAULTHANDLER", "1")

    lf_src = lf_root / "src"
    pythonpath = [str(lf_src)]
    existing = env.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"expected a YAML mapping: {path}")
    return data


def _target_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        return args.checkpoint
    return args.train_root / f"{args.project_prefix}_{args.target}"


def _target_train_config(args: argparse.Namespace) -> Path:
    if args.train_config is not None:
        return args.train_config
    return args.run_dir / "configs" / f"{args.target}.yaml"


def _write_eval_config(
    *, checkpoint: Path, train_config: Path, save_dir: Path, val_size: float
) -> Path:
    cfg = _load_yaml(train_config)
    for key in list(cfg):
        if key.startswith("odb_") or key in {"use_odb", "odb_version"}:
            cfg.pop(key, None)
    cfg.update(
        {
            "model_name_or_path": str(checkpoint),
            "stage": "sft",
            "do_train": False,
            "do_eval": True,
            "finetuning_type": "full",
            "output_dir": str(save_dir),
            "overwrite_output_dir": True,
            "report_to": "none",
            "prediction_loss_only": True,
            "per_device_eval_batch_size": 1,
            "val_size": val_size,
            "dataloader_num_workers": 2,
            "bf16": True,
        }
    )
    for key in ("deepspeed", "max_steps", "save_strategy"):
        cfg.pop(key, None)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint / ".odb_public_valloss_eval.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


def _has_eval_loss(save_dir: Path) -> Path | None:
    for candidate in (save_dir / "all_results.json", save_dir / "eval_results.json"):
        if candidate.is_file():
            data = json.loads(candidate.read_text(encoding="utf-8"))
            if data.get("eval_loss") is not None:
                return candidate
    return None


def _run_valloss(
    *, args: argparse.Namespace, checkpoint: Path, train_config: Path, env: dict[str, str]
) -> Path | None:
    save_dir = checkpoint / f"eval_out_{args.output_tag}_valloss"
    existing = _has_eval_loss(save_dir)
    if existing is not None:
        print(f"[odb-lf-eval] valloss exists: {existing}", flush=True)
        return existing

    config_path = _write_eval_config(
        checkpoint=checkpoint,
        train_config=train_config,
        save_dir=save_dir,
        val_size=args.val_size,
    )
    cmd = [
        sys.executable,
        "-u",
        "-c",
        (
            "import sys; "
            f"sys.argv = ['llamafactory-cli', 'train', {str(config_path)!r}]; "
            "from llamafactory.cli import main; "
            "main()"
        ),
    ]
    print(f"[odb-lf-eval] valloss: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(args.lf_root), env=env, check=True)
    result = _has_eval_loss(save_dir)
    if result is not None:
        data = json.loads(result.read_text(encoding="utf-8"))
        print(
            f"[odb-lf-eval] eval_loss={float(data['eval_loss']):.6f} result={result}",
            flush=True,
        )
    return result


def _resolve_mmmu_script(args: argparse.Namespace) -> Path | None:
    if args.mmmu_script is not None:
        return args.mmmu_script
    env = os.environ.get("ODB_MMMU_EVAL_SCRIPT")
    if env:
        return Path(env)
    return DEFAULT_MMMU_SCRIPT


def _run_mmmu(
    *, args: argparse.Namespace, checkpoint: Path, env: dict[str, str]
) -> Path | None:
    script = _resolve_mmmu_script(args)
    if script is None or not script.exists():
        raise FileNotFoundError(
            "MMMU-MC evaluator was not found. The default script should be "
            f"{DEFAULT_MMMU_SCRIPT}; rerun with --skip-mmmu to skip benchmark eval."
        )

    save_dir = checkpoint / f"mmmu_mc_likelihood_{args.output_tag}"
    result = save_dir / "mmmu_mc_likelihood_results.json"
    if result.is_file():
        print(f"[odb-lf-eval] MMMU-MC exists: {result}", flush=True)
        return result

    cmd = [
        sys.executable,
        "-u",
        str(script),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(save_dir),
    ]
    print(f"[odb-lf-eval] MMMU-MC: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(script.parent), env=env, check=True)
    if result.is_file():
        data = json.loads(result.read_text(encoding="utf-8"))
        acc = data.get("overall_accuracy")
        if acc is not None:
            acc_pct = float(acc) * 100.0 if float(acc) <= 1.0 else float(acc)
            print(
                f"[odb-lf-eval] MMMU-MC overall_accuracy={acc_pct:.2f}% result={result}",
                flush=True,
            )
        return result
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        choices=["odb", "standard"],
        default=os.environ.get("ODB_MM_MIX_EVAL_TARGET", "odb"),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--train-config", type=Path)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--project-prefix", default=DEFAULT_PREFIX)
    parser.add_argument(
        "--lf-root",
        type=Path,
        default=Path(os.environ.get("LLAMAFACTORY_ROOT", "LLaMA-Factory")),
    )
    parser.add_argument(
        "--mmmu-script",
        type=Path,
        help="Optional evaluator override. Defaults to scripts/eval_benchmark.py.",
    )
    parser.add_argument(
        "--output-tag", default=os.environ.get("ODB_MM_MIX_EVAL_TAG", "public_lf")
    )
    parser.add_argument("--skip-valloss", action="store_true")
    parser.add_argument("--skip-mmmu", action="store_true")
    parser.add_argument(
        "--val-size",
        type=float,
        default=float(os.environ.get("ODB_MM_MIX_VAL_SIZE", "0.05")),
        help="Validation split ratio used by LLaMA-Factory valloss eval.",
    )
    parser.add_argument(
        "--cuda-visible-devices", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _raise_fd_limit()
    args.lf_root = args.lf_root.resolve()
    checkpoint = _target_checkpoint(args).resolve()
    train_config = _target_train_config(args).resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")
    if not train_config.exists():
        raise FileNotFoundError(f"train config not found: {train_config}")

    env = _setup_env(
        lf_root=args.lf_root, cuda_visible_devices=args.cuda_visible_devices
    )
    print(f"[odb-lf-eval] target={args.target}", flush=True)
    print(f"[odb-lf-eval] checkpoint={checkpoint}", flush=True)
    print(f"[odb-lf-eval] train_config={train_config}", flush=True)
    if not args.skip_valloss:
        _run_valloss(
            args=args, checkpoint=checkpoint, train_config=train_config, env=env
        )
    if not args.skip_mmmu:
        _run_mmmu(args=args, checkpoint=checkpoint, env=env)


if __name__ == "__main__":
    main()
