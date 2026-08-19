"""Source triage — decide what to read from metadata alone.

For a 1,000-document corpus, not every document is relevant to the
question. Before reading anything, score each source against the targets
using ONLY metadata: file name, directory path, size, type, dates in
names. No text is materialized here — that is the whole point.

The result is a live reading plan, not a contract: sources marked
"unlikely" can be pulled in later if evidence demands it (the controller
sees the full catalog every iteration).

Large corpora (more than the controller's catalog window) additionally
build a target-stratified retrieval frontier: triage returns the best
metadata candidates per open target, and the controller's catalog view
round-robins across targets instead of showing a globally sorted prefix.
The frontier prioritizes; it never trims — every source stays on the
board, resolvable and readable.
"""
from __future__ import annotations

from .llm import call_json
from .state import Board, Source

_BATCH = 150            # catalog lines per triage call
_WINDOW = 60            # controller catalog window; frontier activates above this
_PER_TARGET_CAP = 3     # candidates per target per batch (prompt-size control)
_PRIORITY_RANK = {"definite": 0, "maybe": 1}


def triage_sources(smart_caller, board: Board) -> None:
    """Score every source's relevance to the open targets. Metadata only."""
    docs = [s for s in board.sources if s.kind == "document"]
    if not docs:
        return
    # Small corpora: everything is worth reading; skip the LLM call.
    if len(docs) <= 8:
        for s in docs:
            s.relevance = "definite"
            s.relevance_reason = "small corpus — read everything"
        board.log("triage", f"{len(docs)} sources, small corpus: all definite")
        return
    if len(docs) > _WINDOW:
        _triage_frontier(smart_caller, board, docs)
        return

    targets_text = "\n".join(
        f"- [{t.materiality}] {t.need}" for t in board.open_targets()[:20]
    )

    scored = 0
    for i in range(0, len(docs), _BATCH):
        batch = docs[i:i + _BATCH]
        catalog = "\n".join(
            f"{s.id} | {s.path_hint}/{s.name} | {s.size_bytes // 1024}KB"
            for s in batch
        )
        prompt = f"""You are triaging a document corpus for a research task. You see ONLY metadata: file paths, names, and sizes. Directory names, file names, dates and form types embedded in names carry strong signal (e.g. "sec/10-K/2025" vs "ir/news-releases/2019").

TASK:
{board.instruction[:2000]}

WHAT WE NEED TO ANSWER:
{targets_text}

SOURCES (id | path/name | size):
{catalog}

For each source, judge how likely it is to contain evidence relevant to the task. Be decisive — the cost of marking everything "maybe" is reading everything.

Return JSON:
{{"sources": [{{"id": "...", "relevance": "definite|maybe|unlikely", "reason": "<10 words>"}}]}}
Include EVERY source id listed."""

        parsed = call_json(
            smart_caller, board, prompt, kind="triage", max_tokens=16384,
        )
        if not isinstance(parsed, dict):
            continue
        for item in parsed.get("sources", []):
            if not isinstance(item, dict):
                continue
            src = board.find_source(str(item.get("id", "")))
            if src is None:
                continue
            rel = str(item.get("relevance", "")).lower()
            if rel in ("definite", "maybe", "unlikely"):
                src.relevance = rel
                src.relevance_reason = str(item.get("reason", ""))[:120]
                scored += 1

    # Anything the model skipped stays readable, just deprioritized.
    for s in docs:
        if s.relevance == "unknown":
            s.relevance = "maybe"
            s.relevance_reason = "not scored — default maybe"

    counts: dict[str, int] = {}
    for s in docs:
        counts[s.relevance] = counts.get(s.relevance, 0) + 1
    board.log("triage", f"scored {scored}/{len(docs)} sources: {counts}",
              detail=counts)


def _triage_frontier(smart_caller, board: Board, docs: list[Source]) -> None:
    """Large-corpus triage: best candidates per open target, not a row per file."""
    targets = board.open_targets()[:20]
    targets_text = "\n".join(
        f"- {t.id} [{t.materiality}] {t.need}" for t in targets
    )
    target_ids = {t.id for t in targets}

    # frontier[target_id] -> ordered candidate records
    frontier: dict[str, list[dict]] = {t.id: [] for t in targets}
    malformed_batches = 0
    batch_count = 0

    for batch_index, i in enumerate(range(0, len(docs), _BATCH)):
        batch = docs[i:i + _BATCH]
        batch_count += 1
        batch_ids = {s.id for s in batch}
        catalog = "\n".join(
            f"{s.id} | {s.path_hint}/{s.name} | {s.size_bytes // 1024}KB"
            for s in batch
        )
        prompt = f"""You are triaging one batch of a large document corpus for a research task. You see ONLY metadata: file paths, names, and sizes. Directory names, file names, dates and form types embedded in names carry strong signal.

TASK:
{board.instruction[:2000]}

WHAT WE NEED TO ANSWER (target id, materiality, need):
{targets_text}

SOURCES IN THIS BATCH (id | path/name | size):
{catalog}

For each target, pick the sources from THIS BATCH whose metadata most strongly suggests decisive evidence for that target. At most {_PER_TARGET_CAP} sources per target. A source may serve several targets. Skip targets this batch cannot serve — do not pad.

Return JSON:
{{"candidates": [{{"id": "<source id>", "target_ids": ["<target id>"], "priority": "definite|maybe", "reason": "<10 words>"}}]}}"""

        parsed = call_json(
            smart_caller, board, prompt, kind="triage", max_tokens=16384,
        )
        if not isinstance(parsed, dict) or not isinstance(parsed.get("candidates"), list):
            malformed_batches += 1
            continue

        per_target_new: dict[str, int] = {}
        for rank, item in enumerate(parsed["candidates"]):
            if not isinstance(item, dict):
                continue
            sid = str(item.get("id", ""))
            if sid not in batch_ids:
                continue
            priority = str(item.get("priority", "")).lower()
            if priority not in _PRIORITY_RANK:
                continue
            reason = str(item.get("reason", ""))[:120]
            raw_tids = item.get("target_ids", [])
            if not isinstance(raw_tids, list):
                continue
            for tid in raw_tids:
                tid = str(tid)
                if tid not in target_ids:
                    continue
                existing = next(
                    (c for c in frontier[tid] if c["source_id"] == sid), None,
                )
                if existing is not None:
                    # Duplicate row: keep the strongest priority; keep the
                    # earliest batch/rank for stable ordering.
                    if _PRIORITY_RANK[priority] < _PRIORITY_RANK[existing["priority"]]:
                        existing["priority"] = priority
                        existing["reason"] = reason
                    continue
                if per_target_new.get(tid, 0) >= _PER_TARGET_CAP:
                    continue
                per_target_new[tid] = per_target_new.get(tid, 0) + 1
                frontier[tid].append({
                    "source_id": sid, "priority": priority, "reason": reason,
                    "batch_index": batch_index, "response_rank": rank,
                })

    # Order each target's candidates: priority, then batch order, then rank.
    for tid in frontier:
        frontier[tid].sort(key=lambda c: (
            _PRIORITY_RANK[c["priority"]], c["batch_index"], c["response_rank"],
        ))

    # Source relevance mirrors the strongest RETAINED record, post cap/dedup,
    # so rendering and attribution never disagree.
    strongest: dict[str, str] = {}
    for lst in frontier.values():
        for c in lst:
            prev = strongest.get(c["source_id"])
            if prev is None or _PRIORITY_RANK[c["priority"]] < _PRIORITY_RANK[prev]:
                strongest[c["source_id"]] = c["priority"]
    reasons: dict[str, str] = {}
    for lst in frontier.values():
        for c in lst:
            if c["priority"] == strongest[c["source_id"]]:
                reasons.setdefault(c["source_id"], c["reason"])
    for sid, priority in strongest.items():
        src = board.find_source(sid)
        if src is not None:
            src.relevance = priority
            src.relevance_reason = reasons.get(sid, "")

    candidate_ids = set(strongest)
    # Corpora can contain duplicate source ids (identical files ingested from
    # several directories share a content-hash id) — fallback is id-unique.
    fallback: list[str] = []
    seen_fb: set[str] = set()
    for s in docs:
        if s.id not in candidate_ids and s.id not in seen_fb:
            fallback.append(s.id)
            seen_fb.add(s.id)

    detail = {
        "mode": "frontier", "source_count": len(docs),
        "batch_count": batch_count, "malformed_batches": malformed_batches,
        "candidates_per_target": {t: len(v) for t, v in frontier.items()},
        "unique_candidates": len(candidate_ids),
        "fallback_count": len(fallback),
        "candidate_ids_by_target": {
            t: [c["source_id"] for c in v] for t, v in frontier.items()
        },
    }

    if not candidate_ids:
        board.metadata["retrieval_frontier_enabled"] = False
        detail["mode"] = "frontier_failed"
        board.log("triage", "frontier triage yielded no valid candidates — "
                            "falling back to global catalog", detail=detail)
        return

    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = frontier
    board.metadata["retrieval_fallback"] = fallback

    board.log(
        "triage",
        f"frontier: {len(candidate_ids)} candidates across "
        f"{sum(1 for t in frontier if frontier[t])} targets, "
        f"{len(fallback)} in fallback ({malformed_batches} malformed batches)",
        detail=detail,
    )


def _valid_frontier(board: Board, frontier, fallback) -> bool:
    """Persisted frontier metadata must match live board state exactly:
    live target keys, complete well-typed document-candidate records, and a
    deduplicated fallback that is the exact document-source complement.
    Anything less selects legacy rendering."""
    if not isinstance(frontier, dict) or not frontier:
        return False
    if not isinstance(fallback, list):
        return False
    # A candidate's batch_index is not free-form: it must equal a batch its
    # source actually occupies in board document order (how triage batches).
    # Duplicate ids occupy every batch where an occurrence sits.
    doc_batch: dict[str, set[int]] = {}
    for i, s in enumerate(s for s in board.sources if s.kind == "document"):
        doc_batch.setdefault(s.id, set()).add(i // _BATCH)
    candidate_ids: set[str] = set()
    for tid, lst in frontier.items():
        if not isinstance(tid, str) or board.find_target(tid) is None:
            return False
        if not isinstance(lst, list):
            return False
        seen_in_target: set[str] = set()
        per_batch: dict[int, int] = {}
        prev_key = None
        for c in lst:
            if not isinstance(c, dict):
                return False
            if str(c.get("priority", "")) not in _PRIORITY_RANK:
                return False
            if not isinstance(c.get("reason"), str):
                return False
            if not isinstance(c.get("batch_index"), int):
                return False
            if not isinstance(c.get("response_rank"), int):
                return False
            sid = c.get("source_id")
            if not isinstance(sid, str) or sid in seen_in_target:
                return False
            src = board.find_source(sid)
            if src is None or src.kind != "document":
                return False
            if c["batch_index"] not in doc_batch.get(sid, ()):
                return False
            # Behavioral invariants the records encode: contract ordering
            # and the per-target-per-batch retention cap.
            key = (_PRIORITY_RANK[c["priority"]], c["batch_index"],
                   c["response_rank"])
            if prev_key is not None and key < prev_key:
                return False
            prev_key = key
            per_batch[c["batch_index"]] = per_batch.get(c["batch_index"], 0) + 1
            if per_batch[c["batch_index"]] > _PER_TARGET_CAP:
                return False
            seen_in_target.add(sid)
            candidate_ids.add(sid)
    fallback_ids: set[str] = set()
    for sid in fallback:
        if not isinstance(sid, str) or sid in fallback_ids or sid in candidate_ids:
            return False
        src = board.find_source(sid)
        if src is None or src.kind != "document":
            return False
        fallback_ids.add(sid)
    all_doc_ids = {s.id for s in board.sources if s.kind == "document"}
    return candidate_ids | fallback_ids == all_doc_ids


def _unstratified_prefix(board: Board, frontier: dict, limit: int) -> list[str]:
    """Diagnostic view: retained candidates by priority then board order, no
    target round-robin. Isolates what stratification itself changed."""
    best: dict[str, str] = {}
    for lst in frontier.values():
        for c in lst:
            prev = best.get(c["source_id"])
            if prev is None or _PRIORITY_RANK[c["priority"]] < _PRIORITY_RANK[prev]:
                best[c["source_id"]] = c["priority"]
    order = {s.id: i for i, s in enumerate(board.sources)}
    ranked = sorted(best, key=lambda sid: (_PRIORITY_RANK[best[sid]],
                                           order.get(sid, len(order))))
    return ranked[:limit]


def _rotate(queue: list[str], iteration: int, stride: int, take: int) -> list[str]:
    """Deterministic rotation: offset advances by `stride` per iteration;
    `take` rows are returned with wraparound and no duplicates."""
    if not queue or take <= 0:
        return []
    offset = ((max(iteration, 1) - 1) * stride) % len(queue)
    rotated = queue[offset:] + queue[:offset]
    return rotated[:take]


def catalog_summary(board: Board, limit: int = 60) -> str:
    """Compact source catalog for the controller prompt — read state + relevance.

    Not thread-safe against concurrent board mutation; call from the
    controller thread only (the loop calls it once per controller prompt).
    """
    frontier = board.metadata.get("retrieval_frontier")
    fallback = board.metadata.get("retrieval_fallback")
    doc_count = sum(1 for s in board.sources if s.kind == "document")
    if board.metadata.get("retrieval_frontier_enabled") and doc_count > _WINDOW:
        if _valid_frontier(board, frontier, fallback):
            try:
                return _frontier_page(board, frontier, limit)
            except Exception as exc:  # never let a frontier defect blind the controller
                board.log("frontier_page", f"frontier render failed ({exc}) — "
                                           "legacy catalog used",
                          detail={"error": str(exc)[:200]})
        else:
            # Rejection must never be silent — it is the difference between
            # "treatment ran" and "treatment quietly disabled".
            board.log("frontier_page", "frontier metadata failed validation — "
                                       "legacy catalog used",
                      detail={"validation_failed": True,
                              "iteration": board.iteration})

    docs = sorted(
        board.sources,
        key=lambda s: (
            {"definite": 0, "maybe": 1, "unknown": 2, "unlikely": 3}.get(s.relevance, 2),
            s.read_status == "read",
        ),
    )
    lines = []
    for s in docs[:limit]:
        lines.append(
            f"{s.id} [{s.read_status}/{s.relevance}] {s.path_hint}/{s.name}"
            f" ({s.size_bytes // 1024}KB)"
        )
    if len(docs) > limit:
        unread = sum(1 for s in docs[limit:] if s.read_status == "unread")
        lines.append(f"... and {len(docs) - limit} more ({unread} unread)")
    return "\n".join(lines)


def _frontier_page(board: Board, frontier: dict, limit: int) -> str:
    """Target-balanced controller view: round-robin unread candidates across
    open targets, rotated per target by iteration, topped up from every
    remaining source so nothing is ever hidden from the catalog."""
    def _is_unread(sid: str) -> bool:
        src = board.find_source(sid)
        return src is not None and src.read_status != "read"

    # Full association map from the retained frontier — eligibility-independent,
    # so diagnostics keep associations for closed/waived targets too.
    associations: dict[str, list[str]] = {}
    for tid, lst in frontier.items():
        for c in lst:
            assoc = associations.setdefault(c["source_id"], [])
            if tid not in assoc:
                assoc.append(tid)

    # Eligible targets: open material first, then other open, frontier-listed.
    open_by_id = {t.id: t for t in board.open_targets()}
    eligible = [tid for tid in frontier
                if tid in open_by_id and open_by_id[tid].rank >= 2]
    eligible += [tid for tid in frontier
                 if tid in open_by_id and open_by_id[tid].rank < 2]

    # Per-target unread candidate lists, each rotated by iteration so later
    # iterations expose later candidates WITHIN each target. Fairness is then
    # guaranteed by round-robin page construction across targets.
    per_target_stride = max(1, limit // max(1, len(eligible)))
    rotated_lists: dict[str, list[str]] = {}
    for tid in eligible:
        unread = [c["source_id"] for c in frontier[tid]
                  if _is_unread(c["source_id"])]
        rotated_lists[tid] = _rotate(unread, board.iteration,
                                     per_target_stride, len(unread))

    # Rotate which target leads so uneven page splits stay fair over time.
    if eligible:
        lead = (max(board.iteration, 1) - 1) % len(eligible)
        eligible = eligible[lead:] + eligible[:lead]

    page: list[str] = []
    seen: set[str] = set()
    cursors = {tid: 0 for tid in eligible}
    progressed = True
    while progressed and len(page) < limit:
        progressed = False
        for tid in eligible:
            if len(page) >= limit:
                break
            lst = rotated_lists[tid]
            while cursors[tid] < len(lst):
                sid = lst[cursors[tid]]
                cursors[tid] += 1
                if sid not in seen:
                    page.append(sid)
                    seen.add(sid)
                    progressed = True
                    break

    # Top up from every remaining source — candidates of closed targets,
    # fallback documents, and non-document sources alike. Unread first (with
    # iteration rotation at full page stride), then read, in board order.
    # No source is ever invisible to the catalog.
    def _rest(read_state_read: bool) -> list[str]:
        out: list[str] = []
        emitted: set[str] = set()
        for s in board.sources:  # id-unique even when the corpus has dup ids
            if s.id in seen or s.id in emitted:
                continue
            # Read state comes from the canonical indexed source — the read
            # action mutates only that entry, never duplicate occurrences.
            canonical = board.find_source(s.id)
            status = canonical.read_status if canonical else s.read_status
            if (status == "read") == read_state_read:
                out.append(s.id)
                emitted.add(s.id)
        return out

    if len(page) < limit:
        fill = _rotate(_rest(False), board.iteration, limit, limit - len(page))
        page += fill
        seen.update(fill)
    if len(page) < limit:
        fill = _rotate(_rest(True), board.iteration, limit, limit - len(page))
        page += fill
        seen.update(fill)

    unstrat = set(_unstratified_prefix(board, frontier, limit))
    lines = []
    per_target_shown: dict[str, int] = {}
    fallback_fills = 0
    for sid in page:
        src = board.find_source(sid)
        if src is None:
            continue
        assoc = associations.get(sid, [])
        if assoc:
            for tid in assoc:
                per_target_shown[tid] = per_target_shown.get(tid, 0) + 1
        else:
            fallback_fills += 1
        tag = f" -> {','.join(assoc)}" if assoc else ""
        lines.append(
            f"{src.id} [{src.read_status}/{src.relevance}] {src.path_hint}/{src.name}"
            f" ({src.size_bytes // 1024}KB){tag}"
        )
    total = len(board.sources)
    unread_total = sum(1 for s in board.sources if s.read_status != "read")
    lines.append(f"... frontier view: {total} sources total, "
                 f"{unread_total} unread; all remain readable by id")

    board.log(
        "frontier_page",
        f"iter {board.iteration}: {len(page)} sources shown across "
        f"{len(per_target_shown)} targets",
        detail={
            "iteration": board.iteration,
            "shown_source_ids": page,
            "target_associations": {sid: associations.get(sid, [])
                                    for sid in page},
            "shown_per_target": per_target_shown,
            "fallback_fills": fallback_fills,
            "outside_unstratified_prefix": len([s for s in page
                                                if s not in unstrat]),
        },
    )
    return "\n".join(lines)


