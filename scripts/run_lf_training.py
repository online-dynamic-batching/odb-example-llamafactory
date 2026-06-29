#!/usr/bin/env python3
"""Launch public MM-Mix training through a LLaMA-Factory checkout.

This runner starts from the generated LLaMA-Factory run directory, rewrites only
the small config files into a work directory, and launches training through a
checkout that calls ``odb.integrations.llamafactory.enable_odb(...)``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ENABLE_PATCH_SCRIPT = ROOT / "scripts" / "apply_enable_odb_patch.py"
DEFAULT_RUN_DIR = Path("data/llamafactory-mm-mix")
DEFAULT_OUTPUT_ROOT = Path("outputs/llamafactory-mm-mix")
DS_Z2 = {
    "fp16": {
        "enabled": "auto",
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1,
    },
    "bf16": {"enabled": "auto"},
    "zero_optimization": {
        "stage": 2,
        "contiguous_gradients": True,
        "overlap_comm": False,
        "allgather_bucket_size": 536870912,
        "reduce_bucket_size": 536870912,
    },
    "train_micro_batch_size_per_gpu": "auto",
    "train_batch_size": "auto",
    "gradient_accumulation_steps": "auto",
    "gradient_clipping": "auto",
}


def _default_llamafactory_root() -> Path:
    env = os.environ.get("LLAMAFACTORY_ROOT")
    if env:
        return Path(env)
    return Path("LLaMA-Factory")


def _default_odb_src() -> Path | None:
    env = os.environ.get("ODB_SRC")
    if env:
        return Path(env)
    return None


def _has_enable_hook(llamafactory_root: Path) -> bool:
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


def _ensure_hook(llamafactory_root: Path, *, apply_patch: bool) -> None:
    if _has_enable_hook(llamafactory_root):
        print(
            f"[odb-lf-training] enable_odb hook already present in {llamafactory_root}",
            flush=True,
        )
        return
    if not apply_patch:
        raise RuntimeError(
            f"{llamafactory_root} does not contain the enable_odb(...) hook. "
            "Run scripts/apply_enable_odb_patch.py or pass --apply-hook-patch."
        )
    subprocess.run(
        [
            sys.executable,
            str(ENABLE_PATCH_SCRIPT),
            "--llamafactory-root",
            str(llamafactory_root),
        ],
        check=True,
    )
    if not _has_enable_hook(llamafactory_root):
        raise RuntimeError(
            f"enable_odb(...) hook still missing after patching {llamafactory_root}"
        )


def _prepare_config(args: argparse.Namespace) -> Path:
    source_config = args.run_dir / "configs" / f"{args.loader}.yaml"
    source_info = args.run_dir / "dataset_info.json"
    if not source_config.exists():
        raise FileNotFoundError(source_config)
    if not source_info.exists():
        raise FileNotFoundError(source_info)

    work_dir = args.work_dir / args.project
    work_dir.mkdir(parents=True, exist_ok=True)
    dataset_info_path = work_dir / "dataset_info.json"
    config_path = work_dir / f"{args.loader}.yaml"
    ds_path = work_dir / "ds_z2.json"

    info = json.loads(source_info.read_text(encoding="utf-8"))
    dataset_info_path.write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid YAML config: {source_config}")
    cfg["dataset_dir"] = str(work_dir)
    cfg["output_dir"] = str(args.output_root / args.project)
    if args.max_steps > 0:
        cfg["max_steps"] = args.max_steps
    else:
        cfg.pop("max_steps", None)

    if args.loader == "odb":
        cfg["use_odb"] = True
        cfg["odb_version"] = 51
        cfg["odb_join_mode"] = True
        cfg["odb_loss_scaling_mode"] = "token_exact"
        cfg["odb_no_warmup"] = True
        # Some LLaMA-Factory variants expose this option in YAML. The prepared
        # checkout sets the worker sharing strategy in code, so keep the
        # generated YAML compatible with the current hparams parser.
        cfg.pop("odb_mp_sharing_strategy", None)

    ds_path.write_text(json.dumps(DS_Z2, indent=2) + "\n", encoding="utf-8")
    cfg["deepspeed"] = str(ds_path)

    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "project": args.project,
                "loader": args.loader,
                "run_dir": str(args.run_dir),
                "config": str(config_path),
                "dataset_info": str(dataset_info_path),
                "output_dir": cfg["output_dir"],
                "max_steps": cfg.get("max_steps"),
            },
            indent=2,
        ),
        flush=True,
    )
    return config_path


def _configure_env(llamafactory_root: Path, odb_src: Path | None) -> None:
    src = llamafactory_root / "src"
    pythonpath = [str(src)]
    if odb_src is not None and odb_src.exists():
        pythonpath.insert(0, str(odb_src))
        sys.path.insert(0, str(odb_src))
    sys.path.insert(0, str(src))
    existing = os.environ.get("PYTHONPATH")
    if existing:
        pythonpath.append(existing)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath)

    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PROFILE_DATALOADER", "1")
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    if os.environ.get("NCCL_SOCKET_IFNAME"):
        os.environ.setdefault("GLOO_SOCKET_IFNAME", os.environ["NCCL_SOCKET_IFNAME"])
    topo = os.environ.get("NCCL_TOPO_FILE")
    if topo is not None and (not topo.strip() or not Path(topo).exists()):
        os.environ.pop("NCCL_TOPO_FILE", None)


def _launch(llamafactory_root: Path, config_path: Path) -> None:
    import odb
    from odb.integrations.llamafactory import enable_odb

    print(f"[odb-lf-training] odb={Path(odb.__file__).resolve()}", flush=True)
    print(f"[odb-lf-training] enable_odb={enable_odb}", flush=True)

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    nproc = len([item for item in visible.split(",") if item.strip()]) if visible else 8
    os.environ.setdefault("NPROC_PER_NODE", str(nproc))
    os.environ.setdefault("NNODES", "1")
    os.environ.setdefault("NODE_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(hard, 65536)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))

    cmd = [
        "torchrun",
        "--nnodes",
        os.environ["NNODES"],
        "--node_rank",
        os.environ["NODE_RANK"],
        "--nproc_per_node",
        os.environ["NPROC_PER_NODE"],
        "--master_addr",
        os.environ["MASTER_ADDR"],
        "--master_port",
        os.environ["MASTER_PORT"],
        str(llamafactory_root / "src" / "llamafactory" / "launcher.py"),
        str(config_path),
    ]
    print("[odb-lf-training] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(llamafactory_root), env=dict(os.environ), check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--loader",
        choices=["odb", "standard"],
        default=os.environ.get("ODB_LOADER", "odb"),
    )
    parser.add_argument(
        "--project", default=os.environ.get("ODB_PROJECT", "public_mmmix_odb")
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=int(os.environ.get("ODB_LF_MAX_STEPS", "20")),
        help="Optimizer-step cap for a short run. Use 0 for a full-epoch run.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(os.environ.get("ODB_LF_RUN_DIR", DEFAULT_RUN_DIR)),
        help="Generated run directory from scripts/prepare_lf_training.py.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(os.environ.get("ODB_WORK_DIR", ".odb-work")),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ODB_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)),
    )
    parser.add_argument(
        "--llamafactory-root", type=Path, default=_default_llamafactory_root()
    )
    parser.add_argument("--odb-src", type=Path, default=_default_odb_src())
    parser.add_argument(
        "--apply-hook-patch",
        action="store_true",
        default=os.environ.get("ODB_APPLY_HOOK_PATCH") == "1",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir = args.run_dir.resolve()
    args.work_dir = args.work_dir.resolve()
    args.output_root = args.output_root.resolve()
    args.llamafactory_root = args.llamafactory_root.resolve()
    if args.odb_src is not None:
        args.odb_src = args.odb_src.resolve()

    _ensure_hook(args.llamafactory_root, apply_patch=args.apply_hook_patch)
    config_path = _prepare_config(args)
    _configure_env(args.llamafactory_root, args.odb_src)
    _launch(args.llamafactory_root, config_path)


if __name__ == "__main__":
    main()
