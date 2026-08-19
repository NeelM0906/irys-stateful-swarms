"""Tests for source triage: legacy paths and the target-stratified frontier."""
import json
from dataclasses import dataclass

from src.loop.state import Board, Source, Target
from src.loop.triage import triage_sources, catalog_summary, _WINDOW


# --- fakes -----------------------------------------------------------------

@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 10
    tokens_output: int = 10
    tokens_total: int = 20
    model: str = "fake"


class _FakeCaller:
    """Returns queued JSON payloads, one per call, then empty objects."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0) if self.payloads else {}
        return _FakeResult(text=json.dumps(payload))


def _make_board(n_docs, n_targets=3, instruction="find the facts"):
    board = Board(instruction=instruction)
    for i in range(n_docs):
        board.add_source(Source(
            id=f"s{i}", name=f"doc{i}.txt", path=f"corpus/a/doc{i}.txt",
            kind="document", size_bytes=2048,
        ))
    for i in range(n_targets):
        board.add_target(Target(
            id=f"t{i}", need=f"need {i}",
            materiality="critical" if i == 0 else "medium",
        ))
    return board


def _candidates(*rows):
    return {"candidates": [
        {"id": sid, "target_ids": tids, "priority": prio, "reason": "name match"}
        for sid, tids, prio in rows
    ]}


# --- 1. small corpus path --------------------------------------------------

def test_small_corpus_all_definite_no_call():
    board = _make_board(8)
    caller = _FakeCaller([])
    triage_sources(caller, board)
    assert caller.prompts == []
    assert all(s.relevance == "definite" for s in board.sources)
    assert "retrieval_frontier" not in board.metadata


# --- 2. legacy classification path (9..60) ----------------------------------

def test_legacy_classification_path_unchanged():
    board = _make_board(10)
    payload = {"sources": [
        {"id": f"s{i}", "relevance": "definite" if i < 3 else "unlikely",
         "reason": "r"} for i in range(10)
    ]}
    caller = _FakeCaller([payload])
    triage_sources(caller, board)
    assert len(caller.prompts) == 1
    assert "relevance" in caller.prompts[0]  # legacy one-row-per-source schema
    assert board.find_source("s0").relevance == "definite"
    assert board.find_source("s9").relevance == "unlikely"
    assert "retrieval_frontier" not in board.metadata
    # catalog stays on the legacy globally-sorted view
    summary = catalog_summary(board)
    assert summary.splitlines()[0].startswith("s0 ")


# --- 3. frontier built from multiple batches --------------------------------

def _big_board(n_docs=200, n_targets=3):
    return _make_board(n_docs, n_targets)


def test_frontier_built_from_multiple_batches():
    board = _big_board(200)
    # batch 1 covers s0..s149, batch 2 covers s150..s199
    caller = _FakeCaller([
        _candidates(("s5", ["t0"], "definite"), ("s10", ["t1"], "maybe")),
        _candidates(("s160", ["t2"], "definite"), ("s170", ["t0"], "maybe")),
    ])
    triage_sources(caller, board)
    assert len(caller.prompts) == 2
    assert board.metadata["retrieval_frontier_enabled"] is True
    fr = board.metadata["retrieval_frontier"]
    assert [c["source_id"] for c in fr["t0"]] == ["s5", "s170"]
    assert [c["source_id"] for c in fr["t2"]] == ["s160"]
    # retained candidates got relevance from priority
    assert board.find_source("s5").relevance == "definite"
    assert board.find_source("s10").relevance == "maybe"
    # non-candidates stay unknown and land in fallback
    assert board.find_source("s0").relevance == "unknown"
    assert "s0" in board.metadata["retrieval_fallback"]


# --- 4. no target monopolizes the page ---------------------------------------

def test_round_robin_prevents_target_monopoly():
    board = _big_board(200)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(3)]  # cap is 3/target/batch
    rows += [("s50", ["t1"], "definite"), ("s60", ["t2"], "maybe")]
    caller = _FakeCaller([_candidates(*rows), _candidates()])
    triage_sources(caller, board)
    page = catalog_summary(board, limit=3).splitlines()
    shown = {line.split()[0] for line in page if line.startswith("s")}
    # every target with candidates is represented before t0 gets a second slot
    assert "s50" in shown and "s60" in shown


# --- 5. multi-target source renders once, keeps associations ----------------

def test_multi_target_source_dedup_and_associations():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(("s7", ["t0", "t1"], "definite"))])
    triage_sources(caller, board)
    summary = catalog_summary(board)
    rows = [l for l in summary.splitlines() if l.startswith("s7 ")]
    assert len(rows) == 1
    assert "t0" in rows[0] and "t1" in rows[0]
    page_events = [e for e in board.events if e.kind == "frontier_page"]
    assoc = page_events[-1].detail["target_associations"]["s7"]
    assert set(assoc) == {"t0", "t1"}


# --- 6. later iterations expose later candidates; read ones drop out --------

def test_pagination_progresses_and_read_drop_out():
    board = _big_board(100, n_targets=1)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(3)]
    caller = _FakeCaller([_candidates(*rows)])
    triage_sources(caller, board)
    board.iteration = 1
    first = catalog_summary(board, limit=2)
    board.iteration = 2
    second = catalog_summary(board, limit=2)
    assert first != second  # rotation exposes later unread candidates
    # reading a candidate removes it from the unread queue
    board.find_source("s0").read_status = "read"
    board.iteration = 1
    after_read = catalog_summary(board, limit=2)
    assert not any(line.startswith("s0 ") for line in after_read.splitlines())


# --- 7. closed target leaves round-robin, sources stay catalog-visible ------

def test_closed_target_leaves_allocation_sources_survive():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(
        ("s1", ["t0"], "definite"), ("s2", ["t1"], "definite"),
    )])
    triage_sources(caller, board)
    board.targets[0].status = "closed"
    summary = catalog_summary(board, limit=1)
    assert summary.splitlines()[0].startswith("s2 ")  # t1's candidate leads
    assert board.find_source("s1") is not None  # source not deleted
    # ...and the closed target's candidate remains reachable in the catalog
    full = catalog_summary(board, limit=100)
    assert any(line.startswith("s1 ") for line in full.splitlines())


# --- 8. malformed batches keep sources in fallback, no crash -----------------

def test_malformed_batch_keeps_sources_in_fallback():
    board = _big_board(200)
    caller = _FakeCaller([
        {"nonsense": True},
        _candidates(("s160", ["t0"], "definite")),
    ])
    triage_sources(caller, board)
    assert board.metadata["retrieval_frontier_enabled"] is True
    assert "s5" in board.metadata["retrieval_fallback"]  # batch-1 source retained


# --- 9. fully invalid frontier falls back to legacy summary -----------------

def test_invalid_frontier_falls_back_to_legacy():
    board = _big_board(100)
    caller = _FakeCaller([{"nonsense": True}])
    triage_sources(caller, board)
    assert board.metadata["retrieval_frontier_enabled"] is False
    summary = catalog_summary(board, limit=10)
    assert summary.splitlines()[0].startswith("s0 ")  # legacy global order
    assert "... and" in summary.splitlines()[-1]


# --- 10. every source stays present and resolvable ---------------------------

def test_all_sources_remain_resolvable():
    board = _big_board(150)
    caller = _FakeCaller([_candidates(("s3", ["t0"], "definite"))])
    triage_sources(caller, board)
    assert len(board.sources) == 150
    for i in range(150):
        assert board.find_source(f"s{i}") is not None
    frontier_ids = {c["source_id"]
                    for lst in board.metadata["retrieval_frontier"].values()
                    for c in lst}
    fallback_ids = set(board.metadata["retrieval_fallback"])
    assert frontier_ids | fallback_ids == {f"s{i}" for i in range(150)}


# --- 11. events carry attribution fields ------------------------------------

def test_events_carry_attribution_fields():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(("s3", ["t0"], "definite"))])
    triage_sources(caller, board)
    triage_events = [e for e in board.events
                     if e.kind == "triage" and e.detail
                     and e.detail.get("mode") == "frontier"]
    assert triage_events
    d = triage_events[-1].detail
    for key in ("source_count", "batch_count", "malformed_batches",
                "candidates_per_target", "unique_candidates", "fallback_count",
                "candidate_ids_by_target"):
        assert key in d
    catalog_summary(board)
    page = [e for e in board.events if e.kind == "frontier_page"][-1].detail
    for key in ("iteration", "shown_source_ids", "target_associations",
                "shown_per_target", "fallback_fills",
                "outside_unstratified_prefix"):
        assert key in page


# --- adversarial additions (Tier-1 review gate) ------------------------------

def test_per_target_cap_rejects_fourth_candidate():
    board = _big_board(100)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(4)]
    caller = _FakeCaller([_candidates(*rows)])
    triage_sources(caller, board)
    fr = board.metadata["retrieval_frontier"]
    assert len(fr["t0"]) == 3
    assert "s3" not in [c["source_id"] for c in fr["t0"]]


def test_duplicate_rows_merge_strongest_priority_consistently():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(
        ("s0", ["t0"], "maybe"), ("s0", ["t0"], "definite"),
    )])
    triage_sources(caller, board)
    rec = board.metadata["retrieval_frontier"]["t0"][0]
    assert rec["priority"] == "definite"
    assert board.find_source("s0").relevance == "definite"  # never disagrees


def test_candidate_ordering_priority_batch_rank():
    board = _big_board(200)
    caller = _FakeCaller([
        _candidates(("s10", ["t0"], "maybe"), ("s11", ["t0"], "definite")),
        _candidates(("s160", ["t0"], "definite")),
    ])
    triage_sources(caller, board)
    ordered = [c["source_id"] for c in board.metadata["retrieval_frontier"]["t0"]]
    # definite before maybe; within definite, batch 0 before batch 1
    assert ordered == ["s11", "s160", "s10"]


def test_later_iteration_fairness_with_dominant_target():
    board = _big_board(200, n_targets=2)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(3)]
    rows += [(f"s{i}", ["t0"], "maybe") for i in range(10, 13)]
    rows += [("s50", ["t1"], "definite"), ("s51", ["t1"], "maybe")]
    caller = _FakeCaller([_candidates(*rows[:6] + rows[6:]), _candidates()])
    triage_sources(caller, board)
    for it in (1, 2, 3, 4):
        board.iteration = it
        lines = catalog_summary(board, limit=2).splitlines()
        shown = [l.split()[0] for l in lines if not l.startswith("...")]
        t1_candidates = {"s50", "s51"}
        # t1 has unread candidates on every page, so with 2 slots it must
        # never be shut out while t0 takes both
        assert t1_candidates & set(shown), f"iteration {it}: t1 shut out: {shown}"


def test_zero_open_targets_page_still_renders():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(("s1", ["t0"], "definite"))])
    triage_sources(caller, board)
    for t in board.targets:
        t.status = "closed"
    lines = catalog_summary(board, limit=10).splitlines()
    rows = [l for l in lines if not l.startswith("...")]
    assert len(rows) == 10  # catalog never renders empty
    full = catalog_summary(board, limit=100)
    assert any(line.startswith("s1 ") for line in full.splitlines())


def test_all_candidates_read_page_renders_read_rows():
    board = _big_board(70, n_targets=1)
    caller = _FakeCaller([_candidates(("s0", ["t0"], "definite"))])
    triage_sources(caller, board)
    for s in board.sources:
        s.read_status = "read"
    lines = catalog_summary(board, limit=5).splitlines()
    rows = [l for l in lines if not l.startswith("...")]
    assert len(rows) == 5  # read sources still render when nothing is unread


def test_web_source_visible_in_frontier_mode():
    board = _big_board(70)
    board.add_source(Source(id="web1", name="result", kind="web"))
    caller = _FakeCaller([_candidates(("s1", ["t0"], "definite"))])
    triage_sources(caller, board)
    full = catalog_summary(board, limit=200)
    assert any(line.startswith("web1 ") for line in full.splitlines())


def test_fallback_rotation_uses_full_page_limit_stride():
    board = _big_board(70, n_targets=1)
    caller = _FakeCaller([_candidates(("s0", ["t0"], "definite"))])
    triage_sources(caller, board)
    limit = 3
    board.iteration = 2
    lines = catalog_summary(board, limit=limit).splitlines()
    shown = [l.split()[0] for l in lines if not l.startswith("...")]
    # candidate queue: [s0] rotated by per-target stride; fill comes from the
    # remaining unread sources rotated by offset ((2-1)*limit) % 69
    rest = [f"s{i}" for i in range(1, 70)]
    expected_fill = (rest[limit % len(rest):] + rest[:limit % len(rest)])[:limit - 1]
    assert shown == ["s0"] + expected_fill


def test_metadata_collision_small_corpus_stays_legacy():
    board = _make_board(10)
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = {"t0": [
        {"source_id": "s9", "priority": "definite", "reason": "x",
         "batch_index": 0, "response_rank": 0},
    ]}
    board.metadata["retrieval_fallback"] = []
    summary = catalog_summary(board, limit=10)
    assert summary.splitlines()[0].startswith("s0 ")  # legacy order, no tags
    assert "->" not in summary


def test_invalid_persisted_frontier_falls_back_without_crash():
    board = _big_board(100)
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = {"t0": [{"source_id": "s1"}]}  # no priority
    board.metadata["retrieval_fallback"] = []
    summary = catalog_summary(board, limit=10)  # must not raise
    assert summary.splitlines()[0].startswith("s")
    assert "->" not in summary  # legacy rendering


def _rec(sid, prio="definite", **over):
    rec = {"source_id": sid, "priority": prio, "reason": "r",
           "batch_index": 0, "response_rank": 0}
    rec.update(over)
    return rec


def _seed_frontier(board, frontier, fallback=None):
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = frontier
    if fallback is None:
        cand = {c["source_id"] for lst in frontier.values() for c in lst}
        fallback = [s.id for s in board.sources
                    if s.kind == "document" and s.id not in cand]
    board.metadata["retrieval_fallback"] = fallback


def _is_legacy(summary):
    return "->" not in summary and "frontier view" not in summary


def test_foreign_target_id_falls_back_to_legacy():
    board = _big_board(100)
    _seed_frontier(board, {"ghost": [_rec("s1")]})
    summary = catalog_summary(board, limit=10)
    assert _is_legacy(summary)
    assert "ghost" not in summary


def test_missing_ordering_fields_fall_back_to_legacy():
    board = _big_board(100)
    rec = {"source_id": "s1", "priority": "definite", "reason": "r"}
    _seed_frontier(board, {"t0": [rec]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_duplicate_candidate_in_target_falls_back_to_legacy():
    board = _big_board(100)
    _seed_frontier(board, {"t0": [_rec("s1"), _rec("s1", prio="maybe")]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_duplicate_fallback_ids_fall_back_to_legacy():
    board = _big_board(100)
    cand = {"t0": [_rec("s1")]}
    fb = [s.id for s in board.sources if s.id != "s1"] + ["s2"]
    _seed_frontier(board, cand, fb)
    assert _is_legacy(catalog_summary(board, limit=10))


def test_candidate_fallback_overlap_falls_back_to_legacy():
    board = _big_board(100)
    fb = [s.id for s in board.sources]  # includes the candidate s1
    _seed_frontier(board, {"t0": [_rec("s1")]}, fb)
    assert _is_legacy(catalog_summary(board, limit=10))


def test_incomplete_fallback_complement_falls_back_to_legacy():
    board = _big_board(100)
    _seed_frontier(board, {"t0": [_rec("s1")]}, fallback=["s2", "s3"])
    assert _is_legacy(catalog_summary(board, limit=10))


def test_web_source_candidate_falls_back_to_legacy():
    board = _big_board(100)
    board.add_source(Source(id="web1", name="result", kind="web"))
    _seed_frontier(board, {"t0": [_rec("web1")]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_unsorted_target_list_falls_back_to_legacy():
    board = _big_board(100)
    _seed_frontier(board, {"t0": [
        _rec("s1", prio="maybe"), _rec("s2", prio="definite"),  # wrong order
    ]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_per_batch_cap_violation_falls_back_to_legacy():
    board = _big_board(100)
    _seed_frontier(board, {"t0": [
        _rec(f"s{i}", response_rank=i) for i in range(4)  # 4 in batch 0
    ]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_forged_batch_index_falls_back_to_legacy():
    board = _big_board(100)  # all sources are genuinely in batch 0
    _seed_frontier(board, {"t0": [
        _rec("s1"), _rec("s2", batch_index=1),  # forged: s2 is in batch 0
        _rec("s3", batch_index=1, response_rank=1),
        _rec("s4", batch_index=1, response_rank=2),
        _rec("s5", batch_index=1, response_rank=3),  # would exceed real cap
    ]})
    assert _is_legacy(catalog_summary(board, limit=10))


def test_duplicate_source_ids_generated_frontier_still_activates():
    # Real corpora contain identical files ingested from several directories,
    # which share a content-hash id. The generated frontier must survive that.
    board = _big_board(80)
    for i in range(5):  # duplicate entries: same id, different path
        board.add_source(Source(
            id=f"s{i}", name=f"copy{i}.txt", path=f"corpus/b/copy{i}.txt",
            kind="document", size_bytes=1024,
        ))
    caller = _FakeCaller([_candidates(("s1", ["t0"], "definite"))])
    triage_sources(caller, board)
    assert board.metadata["retrieval_frontier_enabled"] is True
    fb = board.metadata["retrieval_fallback"]
    assert len(fb) == len(set(fb))  # fallback is id-unique
    summary = catalog_summary(board, limit=100)
    assert "frontier view" in summary  # validation accepts, paging activates
    rows = [l.split()[0] for l in summary.splitlines() if not l.startswith("...")]
    assert len(rows) == len(set(rows))  # no duplicate page rows


def test_duplicate_id_in_later_batch_validates():
    # A duplicate occurrence in a different batch makes both batch indices
    # legitimate for that id.
    board = _big_board(200)
    board.add_source(Source(  # duplicate of s10, lands in batch 1 (index 200)
        id="s10", name="copy.txt", path="corpus/b/copy.txt",
        kind="document", size_bytes=1024,
    ))
    _seed_frontier(board, {"t0": [_rec("s10", batch_index=1)]})
    summary = catalog_summary(board, limit=10)
    assert "frontier view" in summary


def test_read_source_with_unread_duplicate_does_not_resurrect():
    board = _big_board(70, n_targets=1)
    board.add_source(Source(  # duplicate occurrence of s0, stays "unread"
        id="s0", name="copy.txt", path="corpus/b/copy.txt",
        kind="document", size_bytes=1024,
    ))
    caller = _FakeCaller([_candidates(("s0", ["t0"], "definite"))])
    triage_sources(caller, board)
    board.find_source("s0").read_status = "read"  # how the read action mutates
    lines = catalog_summary(board, limit=5).splitlines()
    rows = [l for l in lines if not l.startswith("...")]
    s0_rows = [l for l in rows if l.startswith("s0 ")]
    # s0 is read: it must not re-enter through unread fill; if rendered at all
    # it must display as read
    assert all("[read/" in l for l in s0_rows)
    unread_rows = [l for l in rows if "[unread/" in l]
    assert not any(l.startswith("s0 ") for l in unread_rows)


def test_validation_rejection_is_logged():
    board = _big_board(100)
    _seed_frontier(board, {"ghost": [_rec("s1")]})
    catalog_summary(board, limit=10)
    events = [e for e in board.events if e.kind == "frontier_page"
              and e.detail and e.detail.get("validation_failed")]
    assert events


def test_generated_frontier_passes_validation_and_activates():
    board = _big_board(100)
    caller = _FakeCaller([_candidates(("s1", ["t0"], "definite"))])
    triage_sources(caller, board)
    summary = catalog_summary(board, limit=10)
    assert "frontier view" in summary  # the real generated shape activates


def test_failed_triage_event_carries_full_attribution_fields():
    board = _big_board(100)
    caller = _FakeCaller([{"nonsense": True}])
    triage_sources(caller, board)
    events = [e for e in board.events if e.kind == "triage" and e.detail
              and e.detail.get("mode") == "frontier_failed"]
    assert events
    d = events[-1].detail
    for key in ("source_count", "batch_count", "malformed_batches",
                "candidates_per_target", "unique_candidates", "fallback_count",
                "candidate_ids_by_target"):
        assert key in d
    assert d["unique_candidates"] == 0


# --- 12. page never exceeds the limit ----------------------------------------

def test_page_never_exceeds_limit():
    board = _big_board(300)
    rows = [(f"s{i}", [f"t{i % 3}"], "definite") for i in range(9)]
    caller = _FakeCaller([_candidates(*rows), _candidates()])
    triage_sources(caller, board)
    for it in (1, 2, 5):
        board.iteration = it
        lines = catalog_summary(board, limit=_WINDOW).splitlines()
        source_rows = [l for l in lines if not l.startswith("...")]
        assert len(source_rows) <= _WINDOW
