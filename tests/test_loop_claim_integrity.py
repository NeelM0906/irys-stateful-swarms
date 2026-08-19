"""Tests for deterministic source-evidence admission (cycle-4 treatment)."""
from dataclasses import dataclass

from src.loop.state import Board, Source, Target
from src.loop.actions import (_ingest_claims, _usable_source_text,
                              _evidence_supported, _digits_conserved,
                              _read_jobs)


@dataclass
class _FakeDoc:
    text: str = ""


def _board_with_doc(text):
    board = Board(instruction="find the facts")
    board.add_source(Source(id="s0", name="doc-0", path="corpus/a/d.txt",
                            kind="document", size_bytes=len(text),
                            _doc=_FakeDoc(text=text)))
    board.add_target(Target(id="t0", need="need 0", materiality="critical"))
    return board


def _claim(content, evidence, section="s"):
    return {"kind": "observation", "content": content,
            "section": section, "evidence": evidence, "confidence": 0.9}


DOC = ("Fourth quarter revenue grew 29% year-over-year to $953,194 thousand. "
       "Free cash flow was $914,717 thousand for fiscal 2025. "
       "The administrative agent fee is $150,000 per annum.")


def _ingest(board, claims, text=DOC):
    src = board.find_source("s0")
    return _ingest_claims({"claims": claims}, board, source=src,
                          created_by="test", bind_to=["t0"],
                          span_text=text, span_start=0)


# --- admission of grounded claims ---------------------------------------------

def test_grounded_claim_admitted():
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim(
        "revenue grew 29% to $953,194 thousand",
        "revenue grew 29% year-over-year to $953,194 thousand",
    )])
    assert out["claims"] == 1
    assert board.claims[0].source_span is not None


def test_contextual_digits_outside_evidence_rejected_by_design():
    # Deliberate contract strictness: a claim asserting "Q4" must quote
    # evidence containing that digit token. A deterministic rule cannot
    # distinguish contextual qualifiers from fabricated values; rejection
    # rates are counted in observability so smoke can measure this tradeoff.
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim(
        "Q4 revenue grew 29% to $953,194 thousand",
        "revenue grew 29% year-over-year to $953,194 thousand",
    )])
    assert out["claims"] == 0
    assert out["rejected_unsupported_digits"] == 1


def test_layout_variant_evidence_admitted_by_token_coverage():
    board = _board_with_doc(DOC)
    # OCR-ish spacing/line-break variant of a real quote — not an exact
    # substring, but its nontrivial tokens appear in order
    out = _ingest(board, [_claim(
        "free cash flow was $914,717 thousand",
        "Free cash  flow was\n$914,717   thousand for fiscal 2025",
    )])
    assert out["claims"] == 1


# --- rejections ----------------------------------------------------------------

def test_empty_evidence_rejected():
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim("revenue grew", "")])
    assert out["claims"] == 0
    assert out["rejected_empty_evidence"] == 1


def test_invented_prose_rejected():
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim(
        "the company uses a tripartite index architecture",
        "the tripartite index architecture achieved multi-hop precision gains",
    )])
    assert out["claims"] == 0
    assert out["rejected_unsupported_quote"] == 1
    assert board.claims == []


def test_invented_digits_in_evidence_rejected():
    board = _board_with_doc(DOC)
    # evidence quotes real prose but with invented digits (the datadog mode):
    # the digit runs are absent from the cited slice
    out = _ingest(board, [_claim(
        "Q4 revenue grew 28% to $858,169 thousand",
        "revenue grew 28% year-over-year to $858,169 thousand",
    )])
    assert out["claims"] == 0
    # rejection may classify as quote or digits depending on match path,
    # but it must NOT be admitted and must be counted
    assert (out["rejected_unsupported_quote"]
            + out["rejected_unsupported_digits"]) == 1


def test_content_digits_absent_from_evidence_rejected():
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim(
        "the fee is $175,000 per annum",  # content digit not in evidence
        "The administrative agent fee is $150,000 per annum",
    )])
    assert out["claims"] == 0
    assert out["rejected_unsupported_digits"] == 1


def test_mixed_response_admits_valid_and_suppresses_proposals():
    board = _board_with_doc(DOC)
    src = board.find_source("s0")
    parsed = {
        "claims": [
            _claim("fee is $150,000 per annum",
                   "The administrative agent fee is $150,000 per annum"),
            _claim("revenue was $858,169 thousand",
                   "revenue grew to $858,169 thousand"),  # invented digits
        ],
        "proposed_targets": [{"need": "new q", "materiality": "high"}],
    }
    targets_before = len(board.targets)
    out = _ingest_claims(parsed, board, source=src, created_by="test",
                         span_text=DOC, span_start=0)
    assert out["claims"] == 1  # valid one admitted individually
    assert len(board.targets) == targets_before  # proposals suppressed


def test_derived_claims_without_source_unchanged():
    board = _board_with_doc(DOC)
    out = _ingest_claims(
        {"claims": [{"kind": "analysis", "content": "derived conclusion 42",
                     "evidence": "", "confidence": 0.8}]},
        board, source=None, created_by="analyze:test", bind_to=["t0"],
    )
    assert out["claims"] == 1  # derived path is not gated


# --- whitespace-only source usability -------------------------------------------

def test_usable_source_text():
    assert not _usable_source_text("")
    assert not _usable_source_text("   \n\t  ")
    assert not _usable_source_text(None)
    assert _usable_source_text("x")


def test_read_jobs_skip_whitespace_only_source():
    board = _board_with_doc("   \n\n\t   ")
    action = {"kind": "read", "source_id": "s0", "_id": "a1"}
    jobs = _read_jobs(action, board)
    assert jobs == []
    assert board.find_source("s0").read_status == "unread"  # not marked read


# --- helper-level checks ---------------------------------------------------------

def test_evidence_supported_substring_and_threshold():
    ok, path = _evidence_supported("revenue grew 29%",
                                   "total revenue grew 29% this year")
    assert ok and path == "exact"
    ok, _ = _evidence_supported("entirely invented architecture claims",
                                "revenue grew 29% this year")
    assert not ok


def test_matcher_missing_first_token_still_covers():
    # C4-1 counterexample 1: one absent leading token must not zero out the
    # rest — 4 of 5 tokens appear in order = 0.8 coverage = supported.
    ok, path = _evidence_supported("missing alpha beta gamma delta",
                                   "alpha beta gamma delta")
    assert ok and path == "ordered"


def test_matcher_common_tokens_cannot_carry_threshold():
    # C4-1 counterexample 2: four structural tokens with the only substantive
    # token invented must fail (distinct-token coverage guard).
    ok, _ = _evidence_supported("the and the and invented",
                                "the and the and actual")
    assert not ok


def test_matcher_reversed_substantive_tokens_fail():
    ok, _ = _evidence_supported("delta gamma beta alpha",
                                "alpha beta gamma delta")
    # LCS of a reversed sequence is 1 of 4 — far below threshold
    assert not ok


def test_matcher_repeated_tokens_layout_variant_passes():
    ok, _ = _evidence_supported(
        "fee fee schedule lists fee amounts",
        "fee\n fee schedule   lists fee amounts (table)",
    )
    assert ok


def test_matcher_title_stub_false_accept_blocked():
    # A 25-char title stub cannot support architecture prose.
    ok, _ = _evidence_supported(
        "tripartite index architecture achieved multi-hop precision gains",
        "138928981.multi-index-rag",
    )
    assert not ok


# --- tfidf-resolved slices (no exact quote) ---------------------------------------

def test_tfidf_slice_with_qualifying_tokens_passes():
    board = _board_with_doc(DOC)
    # not an exact quote (word order tweak) — resolved via tfidf, then the
    # support check runs against the resolved slice and passes
    out = _ingest(board, [_claim(
        "free cash flow was $914,717 thousand for fiscal 2025",
        "cash flow free was $914,717 thousand fiscal 2025",
    )])
    assert out["claims"] == 1
    assert out["support_ordered"] + out["support_exact"] == 1


def test_tfidf_slice_with_incidental_tokens_fails():
    board = _board_with_doc(DOC)
    out = _ingest(board, [_claim(
        "the company operates a global network of data centers",
        "the company operates a global network of data centers worldwide",
    )])
    assert out["claims"] == 0
    assert out["rejected_unsupported_quote"] == 1


# --- search-path ingestion (span_text=None) ----------------------------------------

def test_search_path_exact_quote_admitted():
    board = _board_with_doc(DOC)
    src = board.find_source("s0")
    out = _ingest_claims(
        {"claims": [_claim("the agent fee is $150,000",
                           "administrative agent fee is $150,000")]},
        board, source=src, created_by="search:test",
    )  # span_text omitted — search-style
    assert out["claims"] == 1


def test_search_path_ordered_tokens_admitted():
    text = "alpha xx beta xx gamma xx delta xx epsilon"
    board = _board_with_doc(text)
    src = board.find_source("s0")
    out = _ingest_claims(
        {"claims": [_claim("alpha beta gamma delta epsilon",
                           "alpha beta gamma delta epsilon")]},
        board, source=src, created_by="search:test",
    )
    assert out["claims"] == 1  # tfidf slice + ordered-token support


def test_search_path_invented_prose_rejected():
    board = _board_with_doc(DOC)
    src = board.find_source("s0")
    out = _ingest_claims(
        {"claims": [_claim("uses a tripartite index",
                           "the tripartite index delivers precision gains")]},
        board, source=src, created_by="search:test",
    )
    assert out["claims"] == 0
    assert out["rejected_unsupported_quote"] == 1


# --- shared-token guard (V2-1: decisive-noun exploit) ------------------------------

def test_decisive_noun_swap_rejected_despite_high_coverage():
    # Boilerplate matches, the one fact-bearing term is invented: the shared
    # content∩evidence token 'tripartite' is absent from the slice → reject.
    ok, _ = _evidence_supported(
        "the company uses the system for tripartite",
        "the company uses the system for records",
        content="the company uses a tripartite system",
    )
    assert not ok


def test_grounded_ocr_variant_with_peripheral_noise_still_passes():
    # Peripheral OCR junk in the evidence (tokens NOT asserted in content)
    # consumes the 20% tolerance without blocking admission.
    ok, path = _evidence_supported(
        "administrative agent fee is $150,000 per annum xj7",
        "administrative agent fee is $150,000 per annum",
        content="the agent fee is $150,000 per annum",
    )
    assert ok  # 1 junk token of 8 (12.5%) sits inside the 20% tolerance


def test_decisive_status_swap_rejected_end_to_end():
    board = _board_with_doc(
        "The merger was terminated by the board in March following review."
    )
    out = _ingest(board, [_claim(
        "the merger was approved by the board",
        "the merger was approved by the board in March following review",
    )], text="The merger was terminated by the board in March following review.")
    assert out["claims"] == 0
    assert out["rejected_unsupported_quote"] == 1


# --- observability denominators -----------------------------------------------------

def test_offered_admitted_and_numeric_counters():
    board = _board_with_doc(DOC)
    out = _ingest(board, [
        _claim("fee is $150,000 per annum",
               "The administrative agent fee is $150,000 per annum"),
        _claim("revenue was $858,169 thousand",
               "revenue grew to $858,169 thousand"),  # invented digits
        _claim("free cash flow existed", ""),          # empty evidence
    ])
    assert out["claims_offered"] == 3
    assert out["claims_admitted"] == 1
    assert out["numeric_claims_checked"] == 2  # two digit-bearing contents
    assert out["support_exact"] == 1


def test_completeness_pass_counters_merge_into_action_summary(monkeypatch):
    import json as _json
    from src.loop.actions import execute_actions

    monkeypatch.setenv("LOOP_READ_COMPLETENESS", "1")
    board = _board_with_doc(DOC)

    class _TwoPassCaller:
        def __init__(self):
            self.calls = 0

        def complete(self, prompt, **kwargs):
            self.calls += 1
            if self.calls == 1:  # first extraction pass: one grounded claim
                payload = {"claims": [_claim(
                    "fee is $150,000 per annum",
                    "The administrative agent fee is $150,000 per annum")]}
            else:  # completeness pass: one grounded + one invented-digits
                payload = {"claims": [
                    _claim("revenue grew 29% year-over-year",
                           "revenue grew 29% year-over-year"),
                    _claim("cash flow was $610,000 thousand",
                           "cash flow was $610,000 thousand"),  # invented
                ]}

            class _R:
                text = _json.dumps(payload)
                tokens_input = 100
                tokens_output = 50
                tokens_total = 150
                model = "fake"
            return _R()

    actions = [{"kind": "read", "source_id": "s0", "focus": "",
                "target_ids": ["t0"]}]
    execute_actions(actions, board, _TwoPassCaller())

    summaries = [e for e in board.events if e.kind == "action_summary"]
    assert summaries, "action_summary event must persist"
    d = summaries[-1].detail
    assert d["claims"] == 2                       # both passes' admissions
    assert d["claims_offered"] == 3               # 1 + 2 offered
    assert d["claims_admitted"] == 2
    # deterministic fixture: the invented claim dies on the ordered-support
    # check (its "$610" digit token never resolves an exact span)
    assert d["rejected_unsupported_quote"] == 1
    assert d["rejected_unsupported_digits"] == 0
    assert d["rejected_empty_evidence"] == 0
    assert d["support_exact"] + d["support_ordered"] == 2
    assert d["numeric_claims_checked"] == 3       # all three carry digits
    assert d["completeness_added"] == 1           # second pass only
    assert d["span_hits"] == 2                    # one exact span per pass


def test_duplicate_admitted_claim_not_integrity_failed():
    board = _board_with_doc(DOC)
    row = _claim("fee is $150,000 per annum",
                 "The administrative agent fee is $150,000 per annum")
    first = _ingest(board, [row])
    assert first["claims"] == 1
    second = _ingest(board, [dict(row)])  # exact duplicate content
    assert second["claims"] == 0            # Board dedup rejects the add
    assert second["claims_admitted"] == 1   # but the gate admitted it


def test_digits_conserved_strictness():
    assert _digits_conserved("fee $150,000", "fee is $150,000",
                             "the fee is $150,000 per annum")
    assert not _digits_conserved("fee $150,000", "fee is $150,000",
                                 "the fee is one hundred fifty thousand")
    # no arithmetic equivalence: 0.15M != 150,000
    assert not _digits_conserved("fee 0.15M", "fee is $150,000",
                                 "the fee is $150,000 per annum")
