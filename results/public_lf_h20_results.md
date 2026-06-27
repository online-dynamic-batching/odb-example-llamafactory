# Public LLaMA-Factory MM-Mix H20 Results

These are integration/reference results for the public MM-Mix LLaMA-Factory
example. They are not exact reproductions of the paper's MM-Mix data or
paper-number reproduction results.
The row-level data table is stored in
[`public_lf_h20_results.csv`](public_lf_h20_results.csv).

All rows use the public LLaMA-Factory MM-Mix recipe with the same seven dataset
entries: `public_mmmix_aokvqa`,
`public_mmmix_chinese_text_recognition`, `public_mmmix_iiit5k`,
`public_mmmix_image_textualization`, `public_mmmix_orand_car_a`,
`public_mmmix_sharegpt4o`, and `public_mmmix_vqa_like`. The 1-node and 2-node
rows use different storage setups, so they should not be read as a scaling
study.

The throughput column reports global emitted training samples per second across
all ranks, computed as `emitted_samples / train_runtime`. In ODB runs, LLaMA-Factory's raw
`train_samples_per_second` field is not used because Trainer step accounting
does not equal emitted-sample accounting after dynamic grouping.

| Setup | Loader | Runtime (s) | Global emitted samples/s | Real tok/s | Train loss | Val loss | MMMU-MC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 node x 8 H20 | Standard | 14442.20 | 13.59 | 9217.29 | 1.0814 | 1.0297 | 42.71 |
| 1 node x 8 H20 | ODB | 4784.12 | 41.01 | 31122.29 | 1.2724 | 1.0604 | 47.06 |
| 2 node x 8 H20 | Standard | 12578.46 | 15.60 | 12942.43 | 1.0754 | 1.0309 | 43.88 |
| 2 node x 8 H20 | ODB | 2920.73 | 67.17 | 51148.06 | 1.2906 | 1.0799 | 47.29 |

Speedups:

- 1-node ODB vs Standard: 3.02x emitted samples/s, 3.38x real tok/s.
- 2-node ODB vs Standard: 4.31x emitted samples/s, 3.95x real tok/s.

They are kept together only to document available public LLaMA-Factory
integration reference results.

For the 2-node rows, the per-rank average emitted-sample rate is roughly the
global rate divided by 16 ranks: 0.97 samples/s/rank for Standard and
4.20 samples/s/rank for ODB.

Checkpoint artifact paths are intentionally omitted from this public summary.
The row-level metrics in the CSV are sufficient for checking the example's
expected behavior.
