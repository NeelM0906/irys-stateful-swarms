#!/usr/bin/env python3
"""Aggregate score_sample.py output into LAB's dual-judge metrics over all 2,010 tasks.

Stratified estimator. Strata are defined by the *reported* (single-judge) result, so
stratum sizes come from the release itself:

    A  = tasks reported all-pass                    N = 636
    B1 = tasks reported missing 1-3 criteria        N = 813
    B2 = tasks reported missing >=4 criteria        N = 561
                                                  total 2010

For a metric p measured on a sample of n_s tasks from stratum s:

    estimate = sum_s (N_s * p_s) / N_total
    var      = sum_s (N_s^2 * p_s(1-p_s)/n_s * (N_s-n_s)/(N_s-1)) / N_total^2

(the trailing factor is the finite-population correction). The 95% interval is
estimate +/- 1.96*sqrt(var). Per-stratum intervals use the Wilson score interval.

Usage:  python aggregate.py results.jsonl
"""
import json, math, sys

N = {"A": 636, "B1": 813, "B2": 561}
NTOT = 2010
REPORTED = 0.3164          # 636/2010, single gemini-3.5-flash-lite judge
JUDGES = ["claude-sonnet-4-6", "gpt-5.5"]


def wilson(k, n, z=1.96):
    if n == 0: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ok = [r for r in rows if "error" not in r]
    print(f"records={len(rows)}  scored={len(ok)}  errors={len(rows)-len(ok)}")
    for r in rows:
        if "error" in r: print("  ERROR", r["tid"], r["error"][:110])

    est = {}
    print(f"\n{'stratum':<9}{'n':>5}{'N':>6}{'both':>9}{'sonnet':>9}{'gpt-5.5':>9}{'dual':>9}")
    for s in ("A", "B1", "B2"):
        g = [r for r in ok if r["stratum"] == s]; n = len(g)
        if not n: continue
        st = sum(r["all_pass_strict"] for r in g)
        so = sum(r["per_judge"][JUDGES[0]]["all_pass"] for r in g)
        gp = sum(r["per_judge"][JUDGES[1]]["all_pass"] for r in g)
        dm = sum(r["dual_all_pass_rate"] for r in g) / n
        est[s] = dict(n=n, st=st / n, so=so / n, gp=gp / n, dm=dm)
        print(f"{s:<9}{n:>5}{N[s]:>6}{st/n:>9.1%}{so/n:>9.1%}{gp/n:>9.1%}{dm:>9.1%}")
        lo, hi = wilson(st, n)
        print(f"{'':<9}{'':>5}{'':>6}  both-judges Wilson 95% CI [{lo:.1%}, {hi:.1%}]")

    covered = sum(N[s] for s in est)
    if covered != NTOT:
        print(f"\nWARNING: strata covered = {covered}/{NTOT}; uncovered treated as 0.")

    def total(key):
        num = var = 0.0
        for s, e in est.items():
            p, n, Ns = e[key], e["n"], N[s]
            num += Ns * p
            fpc = max(0.0, (Ns - n) / (Ns - 1)) if Ns > 1 else 0.0
            var += (Ns ** 2) * (p * (1 - p) / n) * fpc
        return num / NTOT, math.sqrt(var) / NTOT

    print("\n=== extrapolated to all 2,010 tasks ===")
    for k, lab in (("dm", "dual_all_pass_rate (LAB headline)"),
                   ("st", "all_pass, both judges (LAB aggregate)"),
                   ("so", "claude-sonnet-4-6 alone"),
                   ("gp", "gpt-5.5 alone")):
        m, se = total(k)
        print(f"  {lab:<40}{m:7.2%}  95% CI [{max(0,m-1.96*se):.2%}, {m+1.96*se:.2%}]"
              f"  ~{m*NTOT:.0f} tasks")

    m, _ = total("dm")
    print(f"\n  reported (single gemini-3.5-flash-lite judge): {REPORTED:.2%}")
    if m > 0:
        print(f"  ratio reported/dual = {REPORTED/m:.2f}x")

    print("\n=== deliverable resolution (fairness check) ===")
    for jm in JUDGES:
        nf = sum(r["per_judge"][jm]["notfound"] for r in ok)
        tc = sum(r["per_judge"][jm]["n_criteria"] for r in ok)
        aff = sum(1 for r in ok if r["per_judge"][jm]["notfound"] > 0)
        print(f"  {jm:<22} {nf}/{tc} criteria ({nf/max(tc,1):.2%}) across {aff}/{len(ok)} tasks")

    print("\n=== criteria-level pass rate on the sampled tasks ===")
    print("  (not comparable to the published 91.35% macro over all 2,010 tasks;")
    print("   the valid comparison is judge against judge on identical tasks)")
    ic = sum(r["irys_nc"] for r in ok); ip = sum(r["irys_np"] for r in ok)
    print(f"  {'gemini-3.5-flash-lite':<22} {ip}/{ic} = {ip/max(ic,1):.2%}")
    for jm in JUDGES:
        c = sum(r["per_judge"][jm]["n_criteria"] for r in ok)
        p = sum(r["per_judge"][jm]["n_passed"] for r in ok)
        print(f"  {jm:<22} {p}/{c} = {p/max(c,1):.2%}")

    agree = sum(1 for r in ok
                if r["per_judge"][JUDGES[0]]["all_pass"] == r["per_judge"][JUDGES[1]]["all_pass"])
    print(f"\n  judge agreement on task all_pass: {agree}/{len(ok)} = {agree/max(len(ok),1):.1%}")


if __name__ == "__main__":
    main()
