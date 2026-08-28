#!/usr/bin/env python3
"""Re-score released irys-stateful-swarms outputs with LAB's standard two-judge profile.

Judges: claude-sonnet-4-6 + gpt-5.5 (harvey-labs STANDARD_DUAL_JUDGE_PROFILE).
`evaluation.scoring.score_rubric` is used unmodified; only transport-level prompt
caching is added (identical prompt text, model and temperature -- every criterion in a
task shares one prompt prefix). Cached and uncached runs produced identical verdicts.

Usage:
  python score_sample.py --harvey-labs /path/to/harvey-labs \
                         --release /path/to/extracted-release \
                         --nA 120 --nB1 60 --nB2 30 --out results.jsonl

--release is a directory of extracted per-family release archives, i.e.
  <release>/<family>/<family>/<task-path>/{scores.json,output/}

Credentials are read by harvey-labs' own evaluation.run_eval._load_env().
Resumable: rerunning skips task ids already present in --out.
Sampling is nested -- raising --nA keeps every task already drawn at a lower --nA.
"""
import argparse, json, os, random, sys, threading, time
from pathlib import Path

JUDGES = ["claude-sonnet-4-6", "gpt-5.5"]
SPLIT = "## Criterion"
U = {j: {"calls": 0, "in": 0, "out": 0, "cw": 0, "cr": 0} for j in JUDGES}
_l = threading.Lock()


def instrument():
    """Cache the shared prompt prefix. Does not alter prompt text or sampling params."""
    from openai.resources.responses import Responses
    import anthropic.resources.messages as AM

    o = Responses.create
    def ow(self, **kw):
        r = o(self, **kw); m = kw.get("model", "gpt-5.5")
        try:
            u = r.usage; d = getattr(u, "input_tokens_details", None)
            with _l:
                s = U.setdefault(m, {"calls": 0, "in": 0, "out": 0, "cw": 0, "cr": 0})
                s["calls"] += 1; s["in"] += u.input_tokens; s["out"] += u.output_tokens
                s["cr"] += getattr(d, "cached_tokens", 0) or 0
        except Exception: pass
        return r
    Responses.create = ow

    a = AM.Messages.create
    def aw(self, **kw):
        try:
            msgs = kw.get("messages")
            if msgs and len(msgs) == 1 and isinstance(msgs[0].get("content"), str):
                txt = msgs[0]["content"]; i = txt.find(SPLIT)
                if i > 2000:
                    kw["messages"] = [{"role": "user", "content": [
                        {"type": "text", "text": txt[:i], "cache_control": {"type": "ephemeral"}},
                        {"type": "text", "text": txt[i:]}]}]
        except Exception: pass
        r = a(self, **kw); m = kw.get("model", "claude-sonnet-4-6")
        try:
            u = r.usage
            with _l:
                s = U.setdefault(m, {"calls": 0, "in": 0, "out": 0, "cw": 0, "cr": 0})
                s["calls"] += 1; s["in"] += u.input_tokens; s["out"] += u.output_tokens
                s["cw"] += getattr(u, "cache_creation_input_tokens", 0) or 0
                s["cr"] += getattr(u, "cache_read_input_tokens", 0) or 0
        except Exception: pass
        return r
    AM.Messages.create = aw


def build_index(release: Path, hl: Path):
    """Map each released task to its official task.json and its reported result."""
    idx = []
    for sp in release.rglob("scores.json"):
        p = sp.relative_to(release).parts          # (family, family, *task-path, scores.json)
        fam, rest = p[0], p[2:-1]
        tj = hl / "tasks" / fam / Path(*rest) / "task.json"
        d = json.load(open(sp))
        idx.append(dict(
            tid=f"{fam}/{'/'.join(rest)}", run_dir=str(sp.parent), task_json=str(tj),
            exists=tj.exists(), irys_all_pass=bool(d.get("all_pass")),
            irys_nc=d.get("n_criteria", 0), irys_np=d.get("n_passed", 0),
            miss=d.get("n_criteria", 0) - d.get("n_passed", 0)))
    return idx


def stratify(idx, nA, nB1, nB2, seed):
    """A = reported all-pass; B1 = missing 1-3 criteria; B2 = missing >=4."""
    A  = [t for t in idx if t["miss"] == 0]
    B1 = [t for t in idx if 1 <= t["miss"] <= 3]
    B2 = [t for t in idx if t["miss"] >= 4]
    out = []
    for nm, pool, n in (("A", A, nA), ("B1", B1, nB1), ("B2", B2, nB2)):
        pl = sorted(pool, key=lambda x: x["tid"])
        random.Random(seed).shuffle(pl)            # nested across values of n
        for t in pl[:min(n, len(pl))]:
            out.append({**t, "stratum": nm})
    return out, dict(A=len(A), B1=len(B1), B2=len(B2))


def score_cached(criteria, run_dir, judge, task_desc, parallel, score_rubric):
    """Warm the shared prefix with one call, then parallelise the remainder."""
    if len(criteria) <= 2:
        return score_rubric(criteria=criteria, run_dir=run_dir, judge=judge,
                            task_desc=task_desc, parallel=1).criteria_results
    first = score_rubric(criteria=criteria[:1], run_dir=run_dir, judge=judge,
                         task_desc=task_desc, parallel=1)
    rest = score_rubric(criteria=criteria[1:], run_dir=run_dir, judge=judge,
                        task_desc=task_desc, parallel=parallel)
    return first.criteria_results + rest.criteria_results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvey-labs", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default="results.jsonl")
    ap.add_argument("--nA", type=int, default=120)
    ap.add_argument("--nB1", type=int, default=60)
    ap.add_argument("--nB2", type=int, default=30)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--parallel", type=int, default=8)
    ap.add_argument("--tasks-parallel", type=int, default=6)
    a = ap.parse_args()

    hl = Path(a.harvey_labs).expanduser().resolve()
    sys.path.insert(0, str(hl)); os.chdir(hl)
    from evaluation.run_eval import _load_env; _load_env()

    idx = build_index(Path(a.release).expanduser().resolve(), hl)
    unmapped = [t for t in idx if not t["exists"]]
    print(f"released tasks: {len(idx)}  unmapped: {len(unmapped)}")
    for t in unmapped[:10]: print("  unmapped:", t["tid"])

    sample, sizes = stratify(idx, a.nA, a.nB1, a.nB2, a.seed)
    print(f"strata: {sizes}  sampled: {len(sample)}  seed: {a.seed}", flush=True)

    instrument()
    from evaluation.judge import Judge
    from evaluation.scoring import score_rubric
    from concurrent.futures import ThreadPoolExecutor

    outp = Path(a.out); done = set()
    if outp.exists():
        for ln in open(outp):
            try: done.add(json.loads(ln)["tid"])
            except Exception: pass
    todo = [t for t in sample if t["tid"] not in done]
    print(f"already scored: {len(done)}  to score: {len(todo)}", flush=True)
    lk = threading.Lock(); t0 = time.time()

    def run_one(t):
        rec = dict(tid=t["tid"], stratum=t["stratum"], irys_all_pass=t["irys_all_pass"],
                   irys_nc=t["irys_nc"], irys_np=t["irys_np"], per_judge={})
        try:
            td = json.load(open(t["task_json"], encoding="utf-8-sig"))
            for jm in JUDGES:
                crs = score_cached(td["criteria"], Path(t["run_dir"]), Judge(model=jm),
                                   td["title"], a.parallel, score_rubric)
                nc = len(crs); npass = sum(1 for c in crs if c["verdict"] == "pass")
                rec["per_judge"][jm] = dict(
                    n_criteria=nc, n_passed=npass, all_pass=bool(nc > 0 and npass == nc),
                    notfound=sum(1 for c in crs if "not found" in json.dumps(c).lower()))
            aps = [v["all_pass"] for v in rec["per_judge"].values()]
            rec["dual_all_pass_rate"] = sum(aps) / len(aps)   # LAB dual_all_pass_rate
            rec["all_pass_strict"] = all(aps)                 # LAB aggregate all_pass
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        with lk:
            open(outp, "a").write(json.dumps(rec) + "\n")
            pj = rec.get("per_judge", {})
            print(f"[{time.time()-t0:7.1f}s] {rec['tid'][:52]:<52} {rec['stratum']:<2} "
                  f"reported={rec['irys_all_pass']!s:<5} "
                  f"son={pj.get('claude-sonnet-4-6', {}).get('all_pass')!s:<5} "
                  f"gpt={pj.get('gpt-5.5', {}).get('all_pass')!s:<5} "
                  f"both={rec.get('all_pass_strict')!s:<5}"
                  + (f" ERR={rec['error'][:70]}" if "error" in rec else ""), flush=True)

    with ThreadPoolExecutor(max_workers=a.tasks_parallel) as ex:
        list(ex.map(run_one, todo))

    print("\n=== token usage ===")
    for m, s in U.items():
        if s["calls"]:
            print(f"  {m}: calls={s['calls']} in={s['in']:,} out={s['out']:,} "
                  f"cache_write={s['cw']:,} cache_read={s['cr']:,}")
    print(f"wall={time.time()-t0:.1f}s  tasks={len(todo)}")


if __name__ == "__main__":
    main()
