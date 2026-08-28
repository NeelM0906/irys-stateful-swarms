# Independent verification of the LAB benchmark run

An independent re-scoring of the outputs published in
[release v2.0](https://github.com/dl1683/irys-stateful-swarms/releases), using the released
outputs unmodified and the official
[`harveyai/harvey-labs`](https://github.com/harveyai/harvey-labs) evaluator.

Every claim below is checkable against the published artifacts and the official evaluator.

## Summary

Strict all-pass is reported using a **single** LLM judge (`gemini-3.5-flash-lite`). LAB's
standard scoring profile uses **two** judges. Re-scoring the released outputs under the
official dual-judge profile gives **14.42%** (95% CI 11.13–17.70%), a factor of 2.19 below
the single-judge figure for that run.

### Which run this measures

This measures **release v2.0**, the `636 / 2,010 = 31.6%` run — the only benchmark outputs
published to date. The README headline has since moved to `654 / 2,010 = 32.5%` from a later
run ("routing optimization"); those outputs have not been released, so the same correction
cannot be measured on that run by anyone outside the project. The scoring-profile issue is
unchanged between the two runs — the judge configuration in `src/scoring.py` is the same — but
the magnitude quoted here belongs to the run that was actually scored.

## The scoring configuration in the release

Every one of the 2,010 `scores.json` files in release v2.0 records:

```json
"judge_model": "gemini-3.5-flash-lite",
"scorer_type": "harvey"
```

The release contains **zero** `scores_dual.json` files.

```bash
find <extracted-release> -name scores.json -exec jq -r .judge_model {} \; | sort | uniq -c
#  2010 gemini-3.5-flash-lite
find <extracted-release> -name scores_dual.json | wc -l
#     0
```

Per [`harvey-labs/docs/eval-strategies.md`](https://github.com/harveyai/harvey-labs/blob/main/docs/eval-strategies.md),
the standard profile is two judges (`claude-sonnet-4-6` and `gpt-5.5`):

```
dual_criterion_pass = mean(per-judge criterion-pass fractions)
dual_all_pass_rate  = mean(per-judge task all-pass values)
```

and `evaluation/run_eval.py:251` sets `"all_pass": dual_ap == 1.0` — both judges must all-pass. `src/scoring.py`
defaults `judge_model` to a single Gemini flash-lite model, so the published figure is a
single-judge score placed in a table against dual-judge published results.

## Method

- **Inputs:** the released `output/` directories, unmodified.
- **Scorer:** `evaluation.scoring.score_rubric` from `harveyai/harvey-labs`, unmodified.
- **Judges:** `claude-sonnet-4-6` and `gpt-5.5` (the standard profile).
- **Rubrics:** the official `task.json` criteria.
- **Sample:** stratified, n=210 of 2,010, seed `20260827` — 120 drawn from the 636 tasks
  reported all-pass, 60 from tasks missing 1–3 criteria, 30 from tasks missing ≥4. Stratum
  definitions and sizes, the estimator and the finite-population correction are implemented in
  [`verification/aggregate.py`](verification/aggregate.py); the sampling is in
  [`verification/score_sample.py`](verification/score_sample.py) and is deterministic given the
  seed. The sampled task ids are the `tid` field of
  [`verification/results.jsonl`](verification/results.jsonl).
- **Volume:** 11,184 criteria per judge (22,368 judge calls), 0 errors.

Only transport-level prompt caching was added. Model, prompt text, and temperature are
unchanged; cached and uncached runs produced identical verdicts on overlapping tasks.

## Results

| Metric | Result | 95% CI |
|---|---:|---:|
| **`dual_all_pass_rate`** (official headline metric) | **14.42%** | 11.13–17.70% |
| Strict `all_pass` (both judges) | 7.38% | 5.22–9.54% |
| `claude-sonnet-4-6` alone | 17.43% | 14.33–20.54% |
| `gpt-5.5` alone | 11.40% | 8.05–14.75% |
| Published (`gemini-3.5-flash-lite`, single judge) | 31.64% | — |

Of the tasks reported as all-pass, 50.8% survive `claude-sonnet-4-6`, 27.5% survive
`gpt-5.5`, and 23.3% survive both. Tasks *not* reported as all-pass are rarely rescued by
the official judges (3–7% for tasks missing 1–3 criteria, 0% for tasks missing ≥4), so the
correction is not an artifact of sampling only the passing tasks.

## Why the criteria-level numbers are broadly unaffected

The criteria-level results largely hold. Measured on the **same 210 sampled tasks**
(these are not comparable to the published 91.35% macro, which covers all 2,010 tasks;
the meaningful comparison is judge-against-judge on identical tasks):

| Judge | Criteria pass rate on the sample |
|---|---:|
| `gemini-3.5-flash-lite` | 97.26% |
| `claude-sonnet-4-6` | 94.13% |
| `gpt-5.5` | 92.01% |

A 3–5pp per-criterion difference is modest. But all-pass requires **every** criterion in a
task — 53 on average here — to pass at once, so that gap compounds multiplicatively into a
2.2× difference at the task level. The two official judges agreed with each other on
task-level all-pass only 79% of the time, which is the reason the protocol averages two
judges rather than relying on one.

## What we checked and found correct

These were verified against the published artifacts and are **not** in dispute:

- The aggregation arithmetic reconciles exactly: 636 all-pass, micro 91.96%, macro 91.35%,
  with zero discrepancies against `release_manifest.json` across all 27 families.
- All 2,010 tasks map to real LAB tasks. None missing, none fabricated.
- Criteria counts match the official `task.json` files exactly — 114,437 vs 114,437, with
  zero per-task mismatches. The rubrics are unmodified.
- The rubric-leakage protections (`_GENERATION_METADATA_BLOCKLIST` in `src/runner.py`,
  `src/swarm/prompt_audit.py`) are real and appear effective.
- Deliverable filename resolution is not a confound: 0.00% / 0.01% of criteria hit a
  missing file under the two judges.

The harness and measurement pipeline are correct. The judge configuration is the sole
discrepancy found.

## On comparability

Harvey's published all-pass rates are measured on their ~1,200-task private holdout, not
the public set. A corrected public-set figure is still not directly comparable to them.
That caveat applies symmetrically to any public-set result, our own included.

## Reproducing this

```bash
# Re-score any released task with both official judges
python -c "
from evaluation.judge import Judge
from evaluation.scoring import score_rubric
import json
td = json.load(open('tasks/<family>/<task>/task.json', encoding='utf-8-sig'))
for m in ['claude-sonnet-4-6', 'gpt-5.5']:
    r = score_rubric(criteria=td['criteria'], run_dir='<release-task-dir>',
                     judge=Judge(model=m), task_desc=td['title'], parallel=8)
    n = len(r.criteria_results)
    p = sum(1 for c in r.criteria_results if c['verdict'] == 'pass')
    print(m, f'{p}/{n}', 'all_pass=', p == n)
"
```

The full per-task result set for all 210 sampled tasks, the aggregation script that produces
every figure above, and the scoring script that produced the data are committed in
[`verification/`](verification/). The aggregates reproduce offline with no API access:

```bash
cd verification && python aggregate.py results.jsonl
```
