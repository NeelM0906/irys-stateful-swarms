"""Materialize paired candidate result directories for V3 shadow scoring.

Reads candidate markdown from loop/candidate_*.md (produced by synthesis.py
when LOOP_SYNTHESIS_VERIFICATION_SHADOW=1) and writes candidate deliverable
files to output_candidate/ using the same docx/xlsx conversion as production.

Usage:
    python .review_tmp/cycle9_v3_pair.py <results_dir>

Produces output_candidate/ alongside output/ in each task directory that has
shadow corrections. Scoring: run `python -m src.cli score` once on the control
output/ tree and once on the candidate tree (after swapping directories).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def materialize_task(task_dir: Path) -> dict | None:
    """Materialize candidate output for one task. Returns summary or None."""
    loop_dir = task_dir / "loop"
    manifest_path = loop_dir / "candidate_manifest.json"
    if not manifest_path.exists():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not any(v.get("has_corrections") for v in manifest.values()):
        return {"task": str(task_dir), "status": "no_corrections"}

    candidate_dir = task_dir / "output_candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    # Copy non-candidate files from output/ as baseline
    output_dir = task_dir / "output"
    if output_dir.exists():
        for f in output_dir.iterdir():
            if f.is_file() and f.name not in manifest:
                shutil.copy2(f, candidate_dir / f.name)

    from src.runner import _write_docx, _write_xlsx

    materialized = []
    for filename, meta in manifest.items():
        candidate_md_path = loop_dir / f"candidate_{_safe(filename)}.md"
        if not candidate_md_path.exists():
            continue

        candidate_text = candidate_md_path.read_text(encoding="utf-8-sig")
        out_path = candidate_dir / filename

        if filename.lower().endswith(".xlsx"):
            _write_xlsx(out_path, candidate_text)
        elif filename.lower().endswith(".docx"):
            _write_docx(out_path, candidate_text)
        else:
            out_path.write_text(candidate_text, encoding="utf-8")

        materialized.append({
            "filename": filename,
            "has_corrections": meta.get("has_corrections", False),
            "candidate_hash": meta.get("candidate_hash"),
            "chars": meta.get("chars", 0),
        })

    summary = {
        "task": str(task_dir),
        "status": "materialized",
        "files": materialized,
        "corrected_files": sum(
            1 for m in materialized if m["has_corrections"]),
    }
    (loop_dir / "paired_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def _safe(filename: str) -> str:
    return filename.replace("/", "_").replace("\\", "_")


def main():
    if len(sys.argv) < 2:
        print("Usage: python .review_tmp/cycle9_v3_pair.py <results_dir>")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    task_dirs = sorted({
        p.parent.parent for p in results_dir.rglob("candidate_manifest.json")
    })
    print(f"Materializing candidates for {len(task_dirs)} tasks")

    total = 0
    corrected = 0
    for td in task_dirs:
        result = materialize_task(td)
        if result:
            total += 1
            status = result.get("status", "?")
            if status == "materialized":
                n = result.get("corrected_files", 0)
                corrected += n
                print(f"  {td.name}: {n} corrected files materialized")
            else:
                print(f"  {td.name}: {status}")

    print(f"\nDone: {total} tasks processed, {corrected} corrected files")


if __name__ == "__main__":
    main()
