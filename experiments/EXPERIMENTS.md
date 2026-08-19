# Experiments Log

## ⚠ Era boundary: benchmark updated (2026-08-19)

Harvey LAB was updated upstream (v1.0): 2,010 tasks across 27 families (was 1,251/24), including a new 250-task enterprise-search family over a shared 9,288-document corpus, expanded contracts/diligence sets, and rubric revisions. **All results below this section were measured on the pre-update benchmark and are not comparable to post-update runs.** A fresh randomized 54-task baseline (2 per family) opens the new era; the smoke ladder is now 27/54/270.

Model routing for the new era is fixed as a two-tier deployment configuration (values in the untracked `.env`, per repo publication policy) so that all cross-cycle deltas attribute to structural changes only.

---

## Pre-update era summary (v30–v34, June 2026)

Narrative entries for v30–v34 were logged in `ledger.jsonl` but never summarized here. Arc: provenance-architecture iterations (blackboard-as-index with source-span hydration) peaked at 91.7% criteria macro (v31); subsequent routing ablations traded a few points of macro for large cost reductions, and v34 (87.8%) was locked as the final pre-update production config. v35–v36 runs exist under `results/` but were not ledger-logged (hygiene gap noted 2026-08-19).

---

## v29d: GPT-5.5 Synthesis (2026-06-15)
**Config**: `SWARM_SYNTHESIS_MODEL=gpt-5.5` + flash-lite worker + flash smart | **Commit**: `372304a` | **Status**: completed

| Metric | Value |
|--------|-------|
| Criteria macro | **88.8%** (2623/2954) |
| Full-pass | 0/48 |
| vs v29 (flash synth) | **+9.7pp** |
| vs v10 baseline | **+19.1pp** |
| Improved tasks | 44 |
| Regressed tasks | 3 |
| Cost | $120.37 total ($2.51/task avg) |

**What we learned**: Synthesis model quality is the dominant lever. Flash-lite extraction (20K chunks + completeness) produces enough claims (801/task avg). Flash smart tier handles control/analyze adequately. But flash can't assemble 800+ claims into quality output — GPT-5.5 and opus both gain ~10pp by doing synthesis better. GPT-5.5 completed all 48 tasks with zero failures at comparable quality to opus (88.9% on 34 tasks). Three minor regressions: healthcare subpoena (-6.5pp), IRS analysis (-4.2pp), immigration PERM (-1.6pp).

**Top improvements**: LPA scenario-18 (+43.4pp to 98.1%), clinical trial (+33.3pp to 92.2%), capital markets registration (+30.0pp to 83.3%), LPA scenario-20 (+15.8pp to 97.9%), NPDES permit (+12.7pp to 90.9%).

**Decision**: GPT-5.5 is the production synthesis model. Next target: close the remaining ~6pp gap to 95%.

---

## exp-002: Synthesis-Fix Ablation (2026-06-08)
**Config**: `.env.phase0_synthesis_fix` | **Commit**: `1ac5dcf` | **Status**: completed

| Metric | Value |
|--------|-------|
| Criteria rate | 78.1% (2178/2789) |
| All-pass | 0/48 |
| vs baseline-match | +0.4pp (wash) |
| Improved tasks | 18 |
| Regressed tasks | 19 |

**What we learned**: Synthesis-side fixes (completeness criteria active + 2000-char summaries) are a wash. Completeness criteria presence doesn't correlate with task score (77.8% with vs 77.0% without). Strategy entry count correlates better. The 7pp gap is driven by missing domain-specific legal knowledge, not synthesis quality. Net 198 criteria lost vs criteria-aided baseline, concentrated in specific identification (95), calculation (49), and legal citation (21) categories.

**Decision**: Abandon synthesis-side ablations. Implement Domain Lens (exp-003).

---

## exp-001: Baseline-Match (2026-06-08)
**Config**: `.env.phase0_baseline_match` | **Commit**: `1ac5dcf` | **Status**: completed

| Metric | Value |
|--------|-------|
| Criteria rate | 77.9% |
| All-pass | 0/48 |
| vs cycle27 (criteria-aided) | -7.3pp |

**What we learned**: Removing rubric criteria from synthesis (criteria=[]) drops rate from 85.2% to 77.9%. This 7.3pp gap is the criteria-aided advantage. The gap is NOT due to synthesis quality — it's due to the system not knowing what domain-specific items to look for without rubric hints.

---

## exp-003: Domain Lens (2026-06-08)
**Config**: `.env.phase0_domain_lens` | **Commit**: `ae0a228` | **Status**: completed — NEGATIVE RESULT

| Metric | Value |
|--------|-------|
| Criteria rate | 75.3% (2106/2789) |
| All-pass | 0/48 |
| vs baseline-match | **-1.8pp (regression)** |
| Improved tasks | 20 |
| Regressed tasks | 23 |
| Bottom-10 avg delta | +2.1pp (need +6pp) |
| Top-10 avg delta | **-5.4pp (need <1.5pp)** |
| Lens generation success | 33/48 (69%) |
| Lens avg entries when rich | 29.3 |

**Go gate: FAILS ALL 4 CRITERIA.**

**What we learned**: Domain lens is net harmful. Even tasks with rich lens data (29+ entries) regressed by -2.0pp avg. Empty-lens tasks also regressed -1.3pp (suggesting code change confounders between commits). Catastrophic regressions on extraction tasks: extract-psa-key-terms -32pp, extract-assets -34pp, draft-subscription -28pp. The ~29 strategy entries from the lens dilute extraction focus on high-performing tasks without compensating benefit. Some tasks improved massively (+30pp draft-antitrust, +17pp draft-privacy-impact, +15pp identify-settlement-proposal) but these gains are overwhelmed by regressions elsewhere.

**Root cause hypothesis**: Injecting 29 strategy entries into the blackboard overwhelms the synthesis stage for tasks that were already performing well. The extraction workers may be spending iterations chasing lens hypotheses instead of doing careful document extraction. The lens helps some specific task types (compliance training, privacy impact) but hurts others (extraction, key-term analysis).

**Decision**: Abandon domain lens approach. Revert to SWARM_DOMAIN_LENS=0. The criteria-free gap must be closed through extraction depth and blackboard evolution quality, not by injecting domain scaffolding.
