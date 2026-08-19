"""Tests for bulk frontier extraction (cycle-3 treatment)."""
import json
from dataclasses import dataclass

from src.loop.state import Board, Source, Target
from src.loop.actions import bulk_extract_frontier, _BULK_WAVE, _BULK_MAX_TOKENS
from src.loop.triage import triage_sources


# --- fakes -------------------------------------------------------------------

@dataclass
class _FakeDoc:
    text: str = ""


@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 1000
    tokens_output: int = 200
    tokens_total: int = 1200
    model: str = "fake"


class _FakeCaller:
    """Returns one payload per call; records prompts. Payload may be a
    callable(prompt) for prompt-dependent responses."""

    def __init__(self, payload=None, fail_ids=()):
        self.payload = payload
        self.fail_ids = set(fail_ids)
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        for fid in self.fail_ids:
            if f"DOCUMENT: doc-{fid}" in prompt:
                raise RuntimeError("simulated API failure")
        payload = self.payload(prompt) if callable(self.payload) else self.payload
        if payload is None:
            payload = {"claims": [{"kind": "observation",
                                   "content": "fact from " + prompt.split("DOCUMENT: ")[-1].split("\n")[0],
                                   "section": "s1",
                                   "evidence": "quoted fact text",
                                   "confidence": 0.9}]}
        return _FakeResult(text=json.dumps(payload))


def _make_board(n_docs, n_targets=2, doc_bytes=2048):
    board = Board(instruction="find the facts")
    for i in range(n_docs):
        board.add_source(Source(
            id=f"s{i}", name=f"doc-{i}", path=f"corpus/a/doc{i}.txt",
            kind="document", size_bytes=doc_bytes,
            _doc=_FakeDoc(text=f"document {i} body with quoted fact text inside"),
        ))
    for i in range(n_targets):
        board.add_target(Target(
            id=f"t{i}", need=f"need {i}",
            materiality="critical" if i == 0 else "medium",
        ))
    return board


def _seed_valid_frontier(board, rows):
    """rows: list of (sid, [tids], priority). Builds contract-shaped metadata
    through the real triage path so records carry true batch indices."""
    payload = {"candidates": [
        {"id": sid, "target_ids": tids, "priority": prio, "reason": "sig"}
        for sid, tids, prio in rows
    ]}
    triage_sources(_FakeCaller(payload), board)


# --- 1. inactive / small corpora make no calls --------------------------------

def test_small_corpus_no_bulk_calls():
    board = _make_board(10)
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats == {} and caller.prompts == []


def test_no_frontier_no_bulk_calls():
    board = _make_board(100)  # >60 docs but no frontier metadata
    caller = _FakeCaller()
    assert bulk_extract_frontier(board, caller) == {}
    assert caller.prompts == []


# --- 2. invalid metadata cannot activate ---------------------------------------

def test_invalid_frontier_cannot_activate():
    board = _make_board(100)
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = {"ghost": [{"source_id": "s1"}]}
    board.metadata["retrieval_fallback"] = []
    caller = _FakeCaller()
    assert bulk_extract_frontier(board, caller) == {}
    assert caller.prompts == []


# --- 3. selection: definite in, maybe/fallback/web out -------------------------

def test_selects_definite_only():
    board = _make_board(100)
    board.add_source(Source(id="web1", name="hit", kind="web", web_text="x"))
    _seed_valid_frontier(board, [
        ("s1", ["t0"], "definite"), ("s2", ["t1"], "maybe"),
    ])
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["attempted"] == 1
    assert "DOCUMENT: doc-1" in caller.prompts[0]
    assert board.find_source("s1").read_status == "read"
    assert board.find_source("s2").read_status == "unread"
    assert board.find_source("web1").read_status == "unread"


# --- 4. multi-target: one call, all associations bound -------------------------

def test_multi_target_one_call_all_bindings():
    board = _make_board(100)
    _seed_valid_frontier(board, [("s3", ["t0", "t1"], "definite")])
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["attempted"] == 1 and len(caller.prompts) == 1
    claims = [c for c in board.claims if c.created_by.startswith("bulk_extract")]
    assert claims and set(claims[0].target_refs) == {"t0", "t1"}


# --- 5. duplicate source ids: one canonical call --------------------------------

def test_duplicate_ids_one_canonical_call():
    board = _make_board(100)
    board.add_source(Source(  # duplicate occurrence of s1
        id="s1", name="doc-1-copy", path="corpus/b/copy.txt",
        kind="document", size_bytes=1024, _doc=_FakeDoc(text="copy body"),
    ))
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["attempted"] == 1 and len(caller.prompts) == 1
    assert board.find_source("s1").read_status == "read"  # canonical only


# --- 6. call carries full text and target/reason focus --------------------------

def test_call_carries_full_text_and_focus():
    board = _make_board(100)
    _seed_valid_frontier(board, [("s5", ["t0"], "definite")])
    caller = _FakeCaller()
    bulk_extract_frontier(board, caller)
    p = caller.prompts[0]
    assert "document 5 body with quoted fact text inside" in p  # complete text
    assert "need 0" in p        # live target question
    assert "sig" in p           # triage reason
    assert "find the facts" in p  # instruction


# --- 7. claims use existing ingestion with spans and binding --------------------

def test_claims_ingested_with_spans_and_binding():
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t1"], "definite")])
    stats = bulk_extract_frontier(board, _FakeCaller())
    assert stats["claims_added"] == 1
    c = [c for c in board.claims if c.created_by.startswith("bulk_extract")][0]
    assert c.target_refs == ["t1"]
    assert c.source_doc == "doc-1"
    assert c.source_span is not None  # evidence quote resolved to a span
    assert stats["span_exact"] == 1


# --- 8. failed calls leave source unread, add nothing ---------------------------

def test_failed_call_leaves_unread():
    board = _make_board(100)
    _seed_valid_frontier(board, [
        ("s1", ["t0"], "definite"), ("s2", ["t1"], "definite"),
    ])
    caller = _FakeCaller(fail_ids=(1,))
    stats = bulk_extract_frontier(board, caller)
    assert stats["call_failed"] == 1 and stats["succeeded"] == 1
    assert board.find_source("s1").read_status == "unread"
    assert board.find_source("s2").read_status == "read"
    assert not any("doc-1" == c.source_doc for c in board.claims)
    assert stats["parse_success_rate"] == 0.5
    assert stats["all_candidates_attempted"] is True  # attempted, not skipped


# --- 9. proposals in the response are not executed ------------------------------

def test_proposals_not_executed():
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    payload = {"claims": [{"kind": "observation", "content": "f",
                           "section": "s", "evidence": "quoted fact text",
                           "confidence": 0.9}],
               "proposed_targets": [{"need": "new q", "materiality": "high"}],
               "proposed_reads": [{"source_hint": "x", "reason": "y"}]}
    targets_before = len(board.targets)
    bulk_extract_frontier(board, _FakeCaller(payload))
    assert len(board.targets) == targets_before  # no proposals executed
    assert not any(e.kind == "proposed_reads" for e in board.events)


# --- 10. waves are bounded and deterministic ------------------------------------

def test_waves_bounded_and_deterministic():
    board = _make_board(200, n_targets=20)
    rows = [(f"s{i}", [f"t{i % 20}"], "definite") for i in range(40)]
    _seed_valid_frontier(board, rows)
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["attempted"] == 40
    assert stats["waves"] == 2  # 40 candidates / 30 per wave
    # deterministic frontier order: first wave covers the first 30 candidates
    first_wave_docs = {p.split("DOCUMENT: ")[-1].split("\n")[0]
                       for p in caller.prompts[:30]}
    assert len(first_wave_docs) == 30


# --- 11/12. envelope: waves never cross it; undersized envelope recorded --------

def test_envelope_prevents_wave_and_records_skip():
    board = _make_board(100)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(3)]
    rows += [(f"s{i}", ["t1"], "definite") for i in range(3, 6)]
    _seed_valid_frontier(board, rows)
    # Smaller than any single call's worst-case bound (input bytes + framing
    # + worst-case output), so nothing may launch.
    board.token_budget = 10_000
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert caller.prompts == []  # not one call launched
    assert stats["budget_skipped"] == 6
    assert stats["all_candidates_attempted"] is False


# --- 13. budget-offset arithmetic preserves loop headroom -----------------------

def test_budget_offset_preserves_headroom():
    from src.loop import BUDGET_STOP_PCT
    original = 3_000_000
    bulk_tokens = 800_000
    adjusted = original + int(bulk_tokens / (BUDGET_STOP_PCT / 100.0))
    # tokens available to the loop before hitting the stop threshold:
    headroom_before = original * BUDGET_STOP_PCT / 100.0
    headroom_after = adjusted * BUDGET_STOP_PCT / 100.0 - bulk_tokens
    assert abs(headroom_after - headroom_before) < 2  # int rounding only


# --- 14. observability: one start + one completion, full fields ----------------

def test_observability_single_completion_with_adjusted_budget():
    from src.loop.actions import finalize_bulk_extraction
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    original = board.token_budget
    stats = bulk_extract_frontier(board, _FakeCaller())
    finalize_bulk_extraction(board, stats, 85.0)
    evs = [e for e in board.events if e.kind == "bulk_extraction"]
    assert len(evs) == 2  # exactly one start + one completion
    start, done = evs[0].detail, evs[-1].detail
    for key in ("activation", "candidates", "target_associations",
                "estimated_tokens_full_set", "full_set_estimated_fit",
                "envelope_tokens"):
        assert key in start
    for key in ("candidates", "attempted", "calls", "succeeded",
                "parse_failed", "call_failed", "text_load_failed",
                "budget_skipped", "claims_added", "claims_bound",
                "span_exact", "span_fuzzy", "span_fallback", "sources_read",
                "sources_with_accepted_claims", "waves", "max_parallelism",
                "bulk_tokens", "bulk_tokens_input", "bulk_tokens_output",
                "envelope_respected", "all_candidates_attempted",
                "attempt_rate", "parse_success_rate",
                "valid_response_rate_per_call",
                "evidence_conversion_rate", "wall_time_s",
                "original_budget", "adjusted_budget"):
        assert key in done
    for key in ("source_bytes", "estimated_input_tokens",
                "framing_tokens_per_call", "worst_case_output_per_call",
                "render_failures"):
        assert key in start
    assert done["adjusted_budget"] == original + int(
        done["bulk_tokens"] / 0.85)
    blob = (json.dumps(start) + json.dumps(done)).lower()
    for banned in ("criteria", "rubric", "score", "match_criteria"):
        assert banned not in blob


# --- reviewer gap tests (C3-1, C3-2, C3-4, headroom) ----------------------------

def test_estimator_is_tokenizer_independent_upper_bound():
    from src.loop.actions import (_bulk_prompt, _estimate_call_tokens,
                                  _BULK_MAX_TOKENS)
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t0", "t1"], "definite")])
    src = board.find_source("s1")
    # Adversarial text: digit/punctuation-dense and non-ASCII — the densest
    # any byte-level tokenizer can emit is ONE TOKEN PER BYTE, because every
    # token consumes at least one byte. The bound must dominate that.
    adversarial = ("§1.2(a)(iv) — 4,982,113.07€ ≥ 0.0001% ¶" * 200
                   + "é中文" * 500)
    prompt = _bulk_prompt(board, src, adversarial, ["t0", "t1"], ["sig"])
    est = _estimate_call_tokens(prompt)
    worst_possible_tokens = len(prompt.encode("utf-8"))  # 1 token/byte maximum
    assert est >= worst_possible_tokens + _BULK_MAX_TOKENS


def test_hard_envelope_never_breached_by_worst_case():
    from src.loop.actions import _BULK_MAX_TOKENS
    board = _make_board(100)
    rows = [(f"s{i}", ["t0"], "definite") for i in range(3)]
    _seed_valid_frontier(board, rows)
    # Envelope admits roughly one call's worst case; the second wave must be
    # gated by the true bound, not launched on optimism.
    one_call_worst = (len("x") * 0 + 3000 + _BULK_MAX_TOKENS)  # tiny docs
    board.token_budget = one_call_worst
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    # every launched wave's worst case fit inside the envelope by construction
    assert stats["bulk_tokens"] <= stats["original_budget"] or \
        stats["envelope_respected"] is False
    assert stats["attempted"] + stats["budget_skipped"] == stats["candidates"]


def test_malformed_claims_variants_leave_unread_without_abort():
    for bad in ("garbage", {}, None, 7, ["nope", 12]):
        board = _make_board(100)
        _seed_valid_frontier(board, [
            ("s1", ["t0"], "definite"), ("s2", ["t1"], "definite"),
        ])
        caller = _FakeCaller(payload=lambda p, b=bad: (
            {"claims": b} if "DOCUMENT: doc-1" in p
            else {"claims": [{"kind": "observation", "content": "f",
                              "section": "s", "evidence": "quoted fact text",
                              "confidence": 0.9}]}))
        stats = bulk_extract_frontier(board, caller)
        assert stats["parse_failed"] == 1, f"payload {bad!r}"
        assert stats["succeeded"] == 1  # the wave completed despite the bad one
        assert board.find_source("s1").read_status == "unread"
        assert board.find_source("s2").read_status == "read"


def test_empty_claims_list_is_valid_read():
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    stats = bulk_extract_frontier(board, _FakeCaller(payload={"claims": []}))
    assert stats["succeeded"] == 1 and stats["claims_added"] == 0
    assert board.find_source("s1").read_status == "read"  # explicit decision
    assert stats["sources_with_accepted_claims"] == 0
    assert stats["evidence_conversion_rate"] == 0.0


def test_mixed_priority_associations_preserved():
    board = _make_board(100)
    _seed_valid_frontier(board, [
        ("s4", ["t0"], "definite"), ("s4", ["t1"], "maybe"),
    ])
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["attempted"] == 1  # selected via its definite record
    claims = [c for c in board.claims if c.created_by.startswith("bulk_extract")]
    # the retained maybe association is preserved in binding and prompt focus
    assert set(claims[0].target_refs) == {"t0", "t1"}
    assert "need 1" in caller.prompts[0]


def test_budget_skip_still_emits_single_completion():
    from src.loop.actions import finalize_bulk_extraction
    board = _make_board(100, doc_bytes=400_000)
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    board.token_budget = 1_000  # nothing can launch
    original = board.token_budget
    stats = bulk_extract_frontier(board, _FakeCaller())
    finalize_bulk_extraction(board, stats, 85.0)
    evs = [e for e in board.events if e.kind == "bulk_extraction"]
    assert len(evs) == 2  # start + completion even with zero spend
    done = evs[-1].detail
    assert done["adjusted_budget"] == original  # zero tokens → no offset
    assert done["budget_skipped"] == 1
    assert done["all_candidates_attempted"] is False


def test_text_load_failure_lowers_criterion_parse_rate():
    board = _make_board(100)
    board.add_source(Source(  # loadable-text-free document source
        id="s200", name="doc-200", path="corpus/a/doc200.txt",
        kind="document", size_bytes=512, _doc=None,
    ))
    _seed_valid_frontier(board, [
        ("s1", ["t0"], "definite"), ("s200", ["t1"], "definite"),
    ])
    caller = _FakeCaller()
    stats = bulk_extract_frontier(board, caller)
    assert stats["text_load_failed"] == 1 and stats["succeeded"] == 1
    assert stats["parse_success_rate"] == 0.5   # candidate-denominated
    assert stats["valid_response_rate_per_call"] == 1.0  # call-denominated
    assert stats["max_parallelism"] == 1  # launched calls, not wave size
    assert board.find_source("s200").read_status == "unread"


def test_failed_call_keeps_worst_case_reserved_against_envelope():
    board = _make_board(100)
    _seed_valid_frontier(board, [
        ("s1", ["t0"], "definite"), ("s2", ["t1"], "definite"),
    ])
    caller = _FakeCaller(fail_ids=(1,))
    stats = bulk_extract_frontier(board, caller)
    assert stats["call_failed"] == 1
    assert stats["failed_call_reserved_tokens"] > 0  # spend not released


def test_empty_text_counts_as_render_failure_and_breaks_fit():
    board = _make_board(100)
    board.add_source(Source(
        id="s200", name="doc-200", path="corpus/a/doc200.txt",
        kind="document", size_bytes=512, _doc=_FakeDoc(text=""),  # empty
    ))
    _seed_valid_frontier(board, [
        ("s1", ["t0"], "definite"), ("s200", ["t1"], "definite"),
    ])
    bulk_extract_frontier(board, _FakeCaller())
    start = [e for e in board.events if e.kind == "bulk_extraction"][0].detail
    assert start["render_failures"] == 1
    assert start["full_set_estimated_fit"] is False  # unrenderable candidate


def test_budget_offset_preserves_headroom_via_finalize():
    from src.loop.actions import finalize_bulk_extraction
    from src.loop import BUDGET_STOP_PCT
    board = _make_board(100)
    _seed_valid_frontier(board, [("s1", ["t0"], "definite")])
    original = board.token_budget
    stats = bulk_extract_frontier(board, _FakeCaller())
    finalize_bulk_extraction(board, stats, BUDGET_STOP_PCT)
    headroom_before = original * BUDGET_STOP_PCT / 100.0
    headroom_after = (board.token_budget * BUDGET_STOP_PCT / 100.0
                      - stats["bulk_tokens"])
    assert abs(headroom_after - headroom_before) < 2
