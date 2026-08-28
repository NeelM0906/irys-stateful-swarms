# verification/

Data and code behind the two-judge figures in
[`../BENCHMARK_VERIFICATION.md`](../BENCHMARK_VERIFICATION.md).

| File | Contents |
|---|---|
| `results.jsonl` | Per-task results for all 210 sampled tasks. One JSON object per task: task id, stratum, the reported single-judge result, and for each of `claude-sonnet-4-6` and `gpt-5.5` the criteria count, criteria passed, `all_pass`, and the count of criteria whose deliverable did not resolve. Also `dual_all_pass_rate` and `all_pass_strict`. |
| `aggregate.py` | Stratified estimator and confidence intervals. Reproduces every number in the verification document from `results.jsonl`, offline, with no API access. |
| `score_sample.py` | The scoring run that produced `results.jsonl`. Requires the release archives, a `harvey-labs` checkout, and judge credentials. |

## Reproduce the aggregates (offline, no credentials)

```bash
python aggregate.py results.jsonl
```

Expected:

```
dual_all_pass_rate (LAB headline)        14.42%  95% CI [11.13%, 17.70%]  ~290 tasks
all_pass, both judges (LAB aggregate)     7.38%  95% CI [ 5.22%,  9.54%]  ~148 tasks
claude-sonnet-4-6 alone                  17.43%  95% CI [14.33%, 20.54%]  ~350 tasks
gpt-5.5 alone                            11.40%  95% CI [ 8.05%, 14.75%]  ~229 tasks
```

The 210 sampled task ids are the `tid` field of `results.jsonl`:

```bash
jq -r .tid results.jsonl | sort
```

## Reproduce the scoring itself

```bash
python score_sample.py --harvey-labs /path/to/harvey-labs \
                       --release /path/to/extracted-release \
                       --nA 120 --nB1 60 --nB2 30 --seed 20260827 \
                       --out results.jsonl
```

`--release` is a directory of the extracted per-family release archives, i.e.
`<release>/<family>/<family>/<task-path>/{scores.json,output/}`.

Sampling is deterministic given `--seed`, and nested: raising `--nA` keeps every task already
drawn at a lower `--nA`, so the sample can be extended without discarding work. The run is
resumable — task ids already present in `--out` are skipped.

## Sampling design

Strata are defined by the **reported** single-judge result, so stratum sizes are fixed by the
release rather than by anything measured here:

| Stratum | Definition | N |
|---|---|---:|
| A | reported all-pass | 636 |
| B1 | reported missing 1–3 criteria | 813 |
| B2 | reported missing ≥4 criteria | 561 |
| | | **2010** |

Stratum A measures how many reported all-passes survive the two-judge profile; B1 and B2
measure the opposite direction — tasks not reported all-pass that the official judges pass.
Sampling only stratum A would bias the estimate downward, which is why B1 and B2 are included.

`aggregate.py` documents the estimator and the finite-population correction inline.
