#!/usr/bin/env python3
"""Prepare LLaMA-Factory training files for public MM-Mix.

This script uses:

- ``odb_mm_mix`` from the public MM-Mix recipe package for reading portable
  JSONL TMDB records.
- ``online-dynamic-batching`` from PyPI at training time through
  ``odb.integrations.llamafactory.enable_odb(...)``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
ENABLE_ODB_PATCH = ROOT / "patches" / "llamafactory_enable_odb.patch"


def _maybe_add_llamafactory_src(path: str | None) -> None:
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    env_path = os.environ.get("LLAMAFACTORY_SRC") or os.environ.get("LLAMAFACTORY_ROOT")
    if env_path:
        candidates.append(Path(env_path))
    for candidate in candidates:
        src_dir = (
            candidate if (candidate / "llamafactory").exists() else candidate / "src"
        )
        if src_dir.exists() and str(src_dir) not in sys.path:
            sys.path.insert(0, str(src_dir))


def _load_lf_tmdb_writer(llamafactory_src: str | None):
    _maybe_add_llamafactory_src(llamafactory_src)
    try:
        from llamafactory.data.qmdb.tmdb import TMDB as LLaMAFactoryTMDB
    except Exception as exc:
        raise RuntimeError(
            "Could not import LLaMA-Factory TMDB writer. Install LLaMA-Factory "
            "or pass --llamafactory-src /path/to/LLaMA-Factory."
        ) from exc
    return LLaMAFactoryTMDB


def _lf_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "conversation": record.get("conversations") or record.get("conversation") or [],
        "images": record.get("images") or [],
        "source": record.get("source", "unknown"),
        "upstream": record.get("upstream") or {},
    }


def export_lf_tmdb(args: argparse.Namespace) -> dict[str, Path]:
    from odb_mm_mix import TMDB

    lf_tmdb = _load_lf_tmdb_writer(args.llamafactory_src)
    records_by_source: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    store = TMDB(args.data)
    for index, record in enumerate(store.records()):
        source = str(record.get("source") or "unknown").replace("/", "_")
        key = str(record.get("id") or f"{source}_{index:08d}")
        records_by_source[source][key] = _lf_record(record)
    if not records_by_source:
        raise ValueError(f"No records found in {args.data}")

    tmdb_root = args.output / "tmdb"
    if args.overwrite and tmdb_root.exists():
        shutil.rmtree(tmdb_root)
    tmdb_root.mkdir(parents=True, exist_ok=True)

    exported: dict[str, Path] = {}
    for source, rows in sorted(records_by_source.items()):
        target = tmdb_root / source / "tmdb"
        if target.exists() and args.overwrite:
            shutil.rmtree(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        lf_tmdb.dump(rows, str(target), readonly=False, commit_size=args.commit_size)
        exported[source] = target
    return exported


def write_dataset_info(
    args: argparse.Namespace, exported: dict[str, Path]
) -> list[str]:
    dataset_names: list[str] = []
    info: dict[str, Any] = {}
    for source, tmdb_path in sorted(exported.items()):
        name = f"{args.dataset_prefix}_{source}"
        dataset_names.append(name)
        info[name] = {
            "tmdb_file": str(tmdb_path),
            "image_dir": str(args.data),
            "formatting": "sharegpt",
            "weight": 1.0,
            "columns": {"messages": "conversation", "images": "images"},
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            },
        }
    (args.output / "dataset_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dataset_names


def _base_config(args: argparse.Namespace, dataset_names: list[str]) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "model_name_or_path": args.model,
        "trust_remote_code": True,
        "image_max_pixels": args.image_max_pixels,
        "stage": "sft",
        "do_train": True,
        "do_eval": False,
        "finetuning_type": args.finetuning_type,
        "dataset_dir": str(args.output),
        "dataset": ",".join(dataset_names),
        "template": args.template,
        "cutoff_len": args.cutoff_len,
        "preprocessing_num_workers": args.preprocessing_workers,
        "dataloader_num_workers": args.dataloader_workers,
        "dataloader_prefetch_factor": args.prefetch_factor,
        "save_strategy": args.save_strategy,
        "overwrite_output_dir": True,
        "report_to": args.report_to,
        "seed": args.seed,
        "val_size": args.val_size,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": args.warmup_ratio,
        "bf16": True,
        "ddp_timeout": 180000000,
    }
    if args.deepspeed:
        cfg["deepspeed"] = args.deepspeed
    if args.max_steps > 0:
        cfg["max_steps"] = args.max_steps
    return cfg


def _lf_loss_scaling_mode(value: str) -> str:
    mode = value.strip().lower().replace("-", "_")
    aliases = {
        "none": "sample",
        "off": "sample",
        "false": "sample",
        "sample": "sample",
        "approx": "token_approx",
        "token": "token_approx",
        "token_approx": "token_approx",
        "exact": "token_exact",
        "token_exact": "token_exact",
        "true": "token_exact",
    }
    if mode not in aliases:
        raise ValueError(
            "--loss-scaling must be one of sample, approx/token_approx, or exact/token_exact; "
            f"got {value!r}"
        )
    return aliases[mode]


def write_train_configs(
    args: argparse.Namespace, dataset_names: list[str]
) -> dict[str, Path]:
    config_dir = args.output / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    base = _base_config(args, dataset_names)

    standard = dict(base)
    standard["output_dir"] = str(args.train_output / f"{args.project_prefix}_standard")
    standard["dataloader_prefetch_factor"] = args.standard_prefetch_factor

    odb_cfg = dict(base)
    odb_cfg.update(
        {
            "output_dir": str(args.train_output / f"{args.project_prefix}_odb"),
            "use_odb": True,
            "odb_max_input_length": args.token_budget,
            "odb_version": 51,
            "odb_loss_scaling_mode": _lf_loss_scaling_mode(args.loss_scaling),
            "odb_join_mode": True,
            "odb_buffer_size": args.buffer_size,
        }
    )

    paths = {"standard": config_dir / "standard.yaml", "odb": config_dir / "odb.yaml"}
    paths["standard"].write_text(
        yaml.safe_dump(standard, sort_keys=False), encoding="utf-8"
    )
    paths["odb"].write_text(yaml.safe_dump(odb_cfg, sort_keys=False), encoding="utf-8")
    return paths


def write_integration_files(args: argparse.Namespace) -> dict[str, Path]:
    patch_dir = args.output / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    patch_path = patch_dir / ENABLE_ODB_PATCH.name
    if ENABLE_ODB_PATCH.exists():
        shutil.copy2(ENABLE_ODB_PATCH, patch_path)

    hook_note = args.output / "ENABLE_ODB_HOOK.md"
    hook_note.write_text(
        "\n".join(
            [
                "# LLaMA-Factory enable_odb Hook",
                "",
                "This example expects the ODB training path to call:",
                "",
                "```python",
                "from odb.integrations.llamafactory import enable_odb",
                "",
                "enable_odb(",
                "    trainer=trainer,",
                "    train_dataloader=dataset_module['_lazy_dataloader'],",
                "    training_args=training_args,",
                "    data_args=data_args,",
                "    train_dataset=dataset_module['train_dataset'],",
                "    token_budget=data_args.odb_max_input_length,",
                "    loss_scaling='exact',",
                "    join=data_args.odb_join_mode,",
                "    trainer_integration='framework',",
                ")",
                "```",
                "",
                "For normal use, prepare the compatible checkout from the repository root:",
                "",
                "```bash",
                "./run.sh setup-lf",
                "```",
                "",
                f"`patches/{ENABLE_ODB_PATCH.name}` is an advanced patch file for",
                "custom LLaMA-Factory checkouts that already expose the same lazy dataset hooks.",
                "",
                "The generated `run_odb.py` wrapper fails fast when the hook is missing,",
                "so ODB training runs cannot silently fall back to the old in-tree path.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    result = {"hook_note": hook_note}
    if patch_path.exists():
        result["patch"] = patch_path
    return result


def write_wrappers(
    args: argparse.Namespace, config_paths: dict[str, Path]
) -> dict[str, Path]:
    wrapper_dir = args.output / "wrappers"
    wrapper_dir.mkdir(parents=True, exist_ok=True)
    wrappers: dict[str, Path] = {}
    for name, config_path in config_paths.items():
        path = wrapper_dir / f"run_{name}.py"
        require_enable_odb = name == "odb"
        path.write_text(
            "\n".join(
                [
                    "#!/usr/bin/env python3",
                    "from pathlib import Path",
                    "import os",
                    "import resource",
                    "import subprocess",
                    "import sys",
                    "",
                    f"REPO_ROOT = Path({str(args.llamafactory_root)!r})",
                    f"CONFIG = Path({str(config_path)!r})",
                    f"REQUIRE_ENABLE_ODB = {require_enable_odb!r}",
                    f"ENABLE_ODB_PATCH = Path({str(args.output / 'patches' / ENABLE_ODB_PATCH.name)!r})",
                    "src = REPO_ROOT / 'src'",
                    "pythonpath = [str(src)]",
                    "existing_pythonpath = os.environ.get('PYTHONPATH')",
                    "if existing_pythonpath:",
                    "    pythonpath.append(existing_pythonpath)",
                    "os.environ['PYTHONPATH'] = os.pathsep.join(pythonpath)",
                    "import odb  # noqa: F401 - ensure the pip package is installed",
                    "if REQUIRE_ENABLE_ODB:",
                    "    from odb.integrations.llamafactory import enable_odb",
                    "    workflow = REPO_ROOT / 'src' / 'llamafactory' / 'train' / 'sft' / 'workflow.py'",
                    "    try:",
                    "        workflow_text = workflow.read_text(encoding='utf-8')",
                    "    except FileNotFoundError as exc:",
                    "        raise RuntimeError(f'LLaMA-Factory workflow not found: {workflow}') from exc",
                    "    has_hook = 'odb.integrations.llamafactory import enable_odb' in workflow_text and 'enable_odb(' in workflow_text",
                    "    if not has_hook:",
                    "        raise RuntimeError(",
                    "            'This ODB example requires the LLaMA-Factory enable_odb(...) hook. '",
                    "            'Run ./run.sh setup-lf from the example repository to prepare the pinned '",
                    "            'compatible checkout, or set LLAMAFACTORY_ROOT to an equivalent checkout that calls '",
                    "            'odb.integrations.llamafactory.enable_odb(...) after Trainer/DataLoader construction. '",
                    "            f'Advanced patch file: {ENABLE_ODB_PATCH}'",
                    "        )",
                    "    print(f'[odb-lf-example] enable_odb hook verified: {enable_odb}', flush=True)",
                    "os.environ.setdefault('DISABLE_VERSION_CHECK', '1')",
                    "os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')",
                    "os.environ.setdefault('PROFILE_DATALOADER', '1')",
                    "if os.environ.get('NCCL_SOCKET_IFNAME'):",
                    "    os.environ.setdefault('GLOO_SOCKET_IFNAME', os.environ['NCCL_SOCKET_IFNAME'])",
                    "os.environ.setdefault('NCCL_DEBUG', 'WARN')",
                    "visible = os.environ.get('CUDA_VISIBLE_DEVICES')",
                    "nproc = len([x for x in visible.split(',') if x.strip()]) if visible else 1",
                    "os.environ.setdefault('NPROC_PER_NODE', str(nproc))",
                    "os.environ.setdefault('MASTER_ADDR', '127.0.0.1')",
                    "os.environ.setdefault('MASTER_PORT', '29500')",
                    "soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)",
                    "target = min(hard, 65536)",
                    "if soft < target:",
                    "    resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))",
                    "cmd = [",
                    "    'torchrun', '--nnodes', os.environ.get('NNODES', '1'),",
                    "    '--node_rank', os.environ.get('NODE_RANK', '0'),",
                    "    '--nproc_per_node', os.environ['NPROC_PER_NODE'],",
                    "    '--master_addr', os.environ['MASTER_ADDR'],",
                    "    '--master_port', os.environ['MASTER_PORT'],",
                    "    str(REPO_ROOT / 'src/llamafactory/launcher.py'), str(CONFIG),",
                    "]",
                    "print('[odb-lf-example] ' + ' '.join(cmd), flush=True)",
                    "subprocess.run(cmd, cwd=str(REPO_ROOT), env=dict(os.environ), check=True)",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        path.chmod(0o755)
        wrappers[name] = path
    return wrappers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Portable public MM-Mix TMDB directory.",
    )
    parser.add_argument("--output", type=Path, default=Path("data/llamafactory-mm-mix"))
    parser.add_argument("--llamafactory-src", default=None)
    parser.add_argument(
        "--llamafactory-root",
        type=Path,
        default=Path(os.environ.get("LLAMAFACTORY_ROOT", "LLaMA-Factory")),
    )
    parser.add_argument("--dataset-prefix", default="public_mmmix")
    parser.add_argument("--project-prefix", default="public_mmmix_lf")
    parser.add_argument(
        "--train-output",
        type=Path,
        default=Path(
            os.environ.get("ODB_MM_MIX_TRAIN_OUTPUT", "outputs/llamafactory-mm-mix")
        ),
    )
    parser.add_argument("--model", default=os.environ.get("ODB_MM_MIX_MODEL"))
    parser.add_argument("--template", default="qwen3_vl_nothink")
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--finetuning-type", default="full")
    parser.add_argument("--cutoff-len", type=int, default=16384)
    parser.add_argument(
        "--image-max-pixels",
        type=int,
        default=int(os.environ.get("ODB_MM_MIX_IMAGE_MAX_PIXELS", "589824")),
    )
    parser.add_argument("--token-budget", type=int, default=12288)
    parser.add_argument("--buffer-size", type=int, default=1024)
    parser.add_argument("--loss-scaling", default="token_exact")
    parser.add_argument("--dataloader-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=512)
    parser.add_argument("--standard-prefetch-factor", type=int, default=2)
    parser.add_argument("--preprocessing-workers", type=int, default=16)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.05)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--save-strategy", default="no")
    parser.add_argument("--report-to", default="none")
    parser.add_argument("--commit-size", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("pass --model or set ODB_MM_MIX_MODEL")
    return args


def main() -> None:
    args = parse_args()
    args.data = args.data.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.llamafactory_root = args.llamafactory_root.expanduser().resolve()
    args.train_output = args.train_output.expanduser().resolve()
    if args.llamafactory_src is not None:
        args.llamafactory_src = str(Path(args.llamafactory_src).expanduser().resolve())
    if args.deepspeed:
        args.deepspeed = str(Path(args.deepspeed).expanduser().resolve())

    args.output.mkdir(parents=True, exist_ok=True)
    exported = export_lf_tmdb(args)
    dataset_names = write_dataset_info(args, exported)
    config_paths = write_train_configs(args, dataset_names)
    integration_files = write_integration_files(args)
    wrappers = write_wrappers(args, config_paths)
    print(
        json.dumps(
            {
                "dataset_info": str(args.output / "dataset_info.json"),
                "configs": {k: str(v) for k, v in config_paths.items()},
                "integration": {k: str(v) for k, v in integration_files.items()},
                "wrappers": {k: str(v) for k, v in wrappers.items()},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
