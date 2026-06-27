# Online Dynamic Batching for LLaMA-Factory

Train and evaluate a public multimodal fine-tuning example with
[Online Dynamic Batching](https://github.com/online-dynamic-batching/online-dynamic-batching)
and LLaMA-Factory.

This repository is a runnable integration example. It is not a reproduction
package for the paper's experimental numbers; throughput and quality metrics
can differ with hardware, storage, dataset composition, model checkpoints, and
software versions.

## Prerequisites

- A Python environment with PyTorch and NVIDIA GPU support.
- A local Qwen3-VL-2B-Instruct checkpoint, provided through
  `ODB_MM_MIX_MODEL`.
- Network access to GitHub and the public data/model sources, or equivalent
  local mirrors.
- Enough disk space for the generated public TMDB data, checkpoints, validation
  outputs, and MMMU-MC benchmark outputs.

## Run ODB

Use a Python environment with PyTorch/GPU support, then run:

```bash
export ODB_MM_MIX_MODEL=/path/to/Qwen3-VL-2B-Instruct
./run.sh all-odb
```

This installs the example dependencies, prepares a compatible LLaMA-Factory
code directory, builds the public data, trains ODB, and runs validation loss plus
MMMU-MC evaluation.

By default training uses all GPUs in `CUDA_VISIBLE_DEVICES`, or GPUs
`0,1,2,3,4,5,6,7` when `CUDA_VISIBLE_DEVICES` is unset.

## Verified Path

The verified commands cover:

- `./run.sh all-odb`: data build, ODB training, validation loss, and MMMU-MC.
- `./run.sh train-standard` plus `./run.sh eval-standard`: fixed-batch
  baseline training and evaluation.

The records under [results/](results/) are public integration reference runs.
They are useful for checking that the example behaves sensibly, but they should
not be read as paper-number reproduction results.

## Run Step By Step

```bash
# Install ODB and the helper dependencies for this example.
./run.sh install

# Check out the tested LLaMA-Factory code into .deps/LLaMA-Factory-odb.
./run.sh setup-lf

# Download/build the public multimodal TMDB training data.
./run.sh data

# Generate LLaMA-Factory configs and ODB/Standard launch wrappers.
./run.sh prepare

# Train the ODB run.
./run.sh train-odb

# Compute validation loss and MMMU-MC for the ODB checkpoint.
./run.sh eval-odb
```

The default paths are:

- LLaMA-Factory code checked out into: `.deps/LLaMA-Factory-odb`
- Public data: `data/mm-mix-tmdb`
- Generated LLaMA-Factory run files: `data/llamafactory-mm-mix`
- Checkpoints and eval outputs: `outputs/llamafactory-mm-mix`

## Run Standard

After `./run.sh setup-lf`, `./run.sh data`, and `./run.sh prepare`, run the
fixed-batch baseline:

```bash
./run.sh train-standard
./run.sh eval-standard
```

## Common Options

```bash
# Use a subset of GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh train-odb

# Use a custom LLaMA-Factory code directory.
LLAMAFACTORY_ROOT=/path/to/llamafactory-code ./run.sh check

# Skip MMMU-MC or validation loss during eval.
./run.sh eval-odb --skip-mmmu
./run.sh eval-odb --skip-valloss
```

## Outputs

Default model directories:

| Target | Directory |
| --- | --- |
| ODB | `outputs/llamafactory-mm-mix/public_mmmix_lf_odb` |
| Standard | `outputs/llamafactory-mm-mix/public_mmmix_lf_standard` |

`--target` selects the saved model directory and matching generated training
config. Use `--target odb` for `public_mmmix_lf_odb`; use `--target standard`
for `public_mmmix_lf_standard`.

Validation-loss outputs are written under the evaluated checkpoint directory as
`eval_out_public_lf_valloss`.

MMMU-MC outputs are written under the evaluated checkpoint directory as
`mmmu_mc_likelihood_public_lf` and include:

- `mmmu_mc_likelihood_results.json`
- `predictions.jsonl`
- `excluded.jsonl`
- `score_audit.json`

Machine-readable validation records are kept under [results/](results/).

## Commands

| Command | Purpose |
| --- | --- |
| `./run.sh install` | Install Python dependencies for this example. |
| `./run.sh setup-lf` | Check out the tested LLaMA-Factory code into `.deps/LLaMA-Factory-odb`. |
| `./run.sh data` | Build the public TMDB data. |
| `./run.sh prepare` | Generate LLaMA-Factory configs and wrappers. |
| `./run.sh train-odb` | Train with ODB. |
| `./run.sh eval-odb` | Evaluate the ODB checkpoint. |
| `./run.sh train-standard` | Train the fixed-batch baseline. |
| `./run.sh eval-standard` | Evaluate the Standard checkpoint. |
| `./run.sh all-odb` | Run the complete ODB path. |

## Manual Training Launcher

Use `scripts/run_lf_training.py` to override `--max-steps`, `--output-root`, or
`--project` without editing generated YAML files:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python scripts/run_lf_training.py \
  --loader odb \
  --project public_mmmix_odb \
  --run-dir data/llamafactory-mm-mix \
  --output-root outputs/llamafactory-mm-mix \
  --llamafactory-root "$LLAMAFACTORY_ROOT" \
  --max-steps 20
```

Set `--max-steps 0` for a full-epoch run. Use `--loader standard` for the
fixed-batch baseline.

For multi-node runs, set `NNODES`, `NODE_RANK`, `MASTER_ADDR`, `MASTER_PORT`,
and your site-specific NCCL/RDMA environment variables before launching.

## LLaMA-Factory Checkout

`./run.sh setup-lf` prepares a tested LLaMA-Factory checkout with the ODB hooks
needed by this example.

This example keeps model-specific preprocessing inside LLaMA-Factory. ODB is
enabled after LLaMA-Factory has produced single-sample tensor dictionaries.

## Related Examples

- Shared public data recipe: [odb-mm-mix-example](https://github.com/online-dynamic-batching/odb-mm-mix-example)
- HF Trainer native example: [odb-example-hf-trainer](https://github.com/online-dynamic-batching/odb-example-hf-trainer)
- Accelerate example: [odb-mm-mix-accelerate](https://github.com/online-dynamic-batching/odb-mm-mix-accelerate)
- Lightning example: [odb-mm-mix-lightning](https://github.com/online-dynamic-batching/odb-mm-mix-lightning)
