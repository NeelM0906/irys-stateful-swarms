"""Tests for Verification Barrier V2 (cycle 8 treatment)."""
import json
from dataclasses import dataclass

from src.loop.synthesis import (
    VerificationLedger, _validate_audit, _blocking_findings,
    _validate_correction, _apply_edits, _verify_chunk, _withhold_text,
    _VERIFICATION_BUDGET_RATIO, _VERIFY_CALL_RESERVE,
)
from src.loop.state import Board, Claim, Source, Target


@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 1000
    tokens_output: int = 200
    tokens_total: int = 1200
    model: str = "fake"


class _MockCaller:
    """Configurable caller that returns preset JSON responses."""

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.prompts = []
        self._idx = 0

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self._idx < len(self.responses):
            resp = self.responses[self._idx]
            self._idx += 1
        else:
            resp = '{"findings": []}'
        return _FakeResult(text=resp, tokens_input=100,
                           tokens_output=50, tokens_total=150)


def _make_board():
    board = Board(instruction="produce the report", output_dir="")
    board.add_source(Source(id="s0", name="doc-0", kind="document",
                            size_bytes=100))
    board.add_target(Target(id="t0", need="question zero",
                            materiality="critical", status="closed"))
    c = Claim(id="c100", content="fact alpha", kind="analysis",
              target_refs=["t0"], confidence=0.9)
    board.add_claim(c)
    board.find_target("t0").claim_refs.append(c.id)
    return board


# --- 1. VerificationLedger ---------------------------------------------------

def test_ledger_empty_budget():
    ledger = VerificationLedger()
    assert ledger.budget == 0
    assert ledger.can_reserve() is False

def test_ledger_budget_grows_with_drafts():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)
    assert ledger.budget == int(12000 * _VERIFICATION_BUDGET_RATIO)
    assert ledger.can_reserve() is True

def test_ledger_charge_tracks_spending():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)
    ledger.charge(500, 100)
    assert ledger.spent == 600
    assert ledger.calls == 1

def test_ledger_exhaustion():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(1000, 0)
    budget = ledger.budget  # 150
    ledger.charge(budget, 0)
    assert ledger.can_reserve() is False

def test_ledger_reserve_requires_minimum():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(2000, 0)
    budget = ledger.budget  # 300
    assert budget < _VERIFY_CALL_RESERVE
    assert ledger.can_reserve() is False

def test_ledger_reserve_sufficient():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(20000, 0)
    budget = ledger.budget  # 3000
    assert budget >= _VERIFY_CALL_RESERVE
    assert ledger.can_reserve() is True
    ledger.charge(budget - _VERIFY_CALL_RESERVE, 0)
    assert ledger.can_reserve() is True
    ledger.charge(1, 0)
    assert ledger.can_reserve() is False

def test_ledger_summary():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(5000, 1000)
    ledger.chunks_audited = 3
    ledger.chunks_clean = 2
    ledger.chunks_corrected = 1
    s = ledger.summary()
    assert s["draft_tokens"] == 6000
    assert s["chunks_audited"] == 3


# --- 2. Audit validation ---------------------------------------------------

def test_validate_audit_clean():
    assert _validate_audit({"findings": []}, ["c100"]) is True

def test_validate_audit_with_findings():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "wrong text", "explanation": "should be X"},
    ]}
    assert _validate_audit(audit, ["c100"]) is True

def test_validate_audit_rejects_non_dict():
    assert _validate_audit("not a dict", []) is False
    assert _validate_audit(None, []) is False

def test_validate_audit_rejects_missing_findings():
    assert _validate_audit({}, []) is False

def test_validate_audit_rejects_overflow():
    findings = [{"id": f"f{i}", "defect": "factual_error",
                 "severity": "blocking", "span": "x", "explanation": "y"}
                for i in range(13)]
    assert _validate_audit({"findings": findings}, []) is False

def test_validate_audit_rejects_duplicate_ids():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "x", "explanation": "y"},
        {"id": "f1", "defect": "wrong_entity", "severity": "blocking",
         "span": "z", "explanation": "w"},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_rejects_bad_defect():
    audit = {"findings": [
        {"id": "f1", "defect": "spelling_error", "severity": "blocking",
         "span": "x", "explanation": "y"},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_rejects_bad_severity():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "critical",
         "span": "x", "explanation": "y"},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_omission_allows_empty_span():
    audit = {"findings": [
        {"id": "f1", "defect": "omission", "severity": "blocking",
         "span": "", "explanation": "missing important fact"},
    ]}
    assert _validate_audit(audit, []) is True

def test_validate_audit_non_omission_requires_span():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "", "explanation": "wrong number"},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_rejects_missing_explanation():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "wrong"},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_rejects_empty_explanation():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "wrong", "explanation": ""},
    ]}
    assert _validate_audit(audit, []) is False

def test_validate_audit_rejects_non_string_fields():
    audit = {"findings": [
        {"id": 1, "defect": "factual_error", "severity": "blocking",
         "span": "x", "explanation": "y"},
    ]}
    assert _validate_audit(audit, []) is False
    audit2 = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": 123, "explanation": "y"},
    ]}
    assert _validate_audit(audit2, []) is False
    audit3 = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "x", "explanation": 999},
    ]}
    assert _validate_audit(audit3, []) is False


# --- 3. Blocking findings ---------------------------------------------------

def test_blocking_findings_filters():
    audit = {"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "x", "explanation": "y"},
        {"id": "f2", "defect": "omission", "severity": "advisory",
         "span": "", "explanation": "minor"},
    ]}
    blockers = _blocking_findings(audit)
    assert len(blockers) == 1
    assert blockers[0]["id"] == "f1"


# --- 4. Correction validation -----------------------------------------------

def test_validate_correction_valid():
    corr = {"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "wrong", "replacement": "right"},
    ]}
    assert _validate_correction(corr, {"f1"}) is True

def test_validate_correction_rejects_unknown_finding():
    corr = {"edits": [
        {"finding_id": "f99", "operation": "replace",
         "span": "wrong", "replacement": "right"},
    ]}
    assert _validate_correction(corr, {"f1"}) is False

def test_validate_correction_rejects_bad_operation():
    corr = {"edits": [
        {"finding_id": "f1", "operation": "rewrite",
         "span": "wrong", "replacement": "right"},
    ]}
    assert _validate_correction(corr, {"f1"}) is False

def test_validate_correction_delete_no_replacement():
    corr = {"edits": [
        {"finding_id": "f1", "operation": "delete", "span": "remove this"},
    ]}
    assert _validate_correction(corr, {"f1"}) is True

def test_validate_correction_rejects_empty_edits():
    corr = {"edits": []}
    assert _validate_correction(corr, {"f1"}) is False

def test_validate_correction_rejects_non_string_finding_id():
    corr = {"edits": [
        {"finding_id": ["f1"], "operation": "replace",
         "span": "wrong", "replacement": "right"},
    ]}
    assert _validate_correction(corr, {"f1"}) is False

def test_validate_correction_rejects_non_string_span():
    corr = {"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": 123, "replacement": "right"},
    ]}
    assert _validate_correction(corr, {"f1"}) is False

def test_validate_correction_rejects_non_string_replacement():
    corr = {"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "wrong", "replacement": 456},
    ]}
    assert _validate_correction(corr, {"f1"}) is False


# --- 5. Apply edits --------------------------------------------------------

def test_apply_replace():
    result = _apply_edits("The revenue was $50M in 2024.",
                          [{"finding_id": "f1", "operation": "replace",
                            "span": "$50M", "replacement": "$75M"}])
    assert result == "The revenue was $75M in 2024."

def test_apply_insert_after():
    result = _apply_edits("The company grew.",
                          [{"finding_id": "f1", "operation": "insert_after",
                            "span": "grew.", "replacement": " Revenue doubled."}])
    assert result == "The company grew. Revenue doubled."

def test_apply_delete():
    result = _apply_edits("The revenue was approximately $50M.",
                          [{"finding_id": "f1", "operation": "delete",
                            "span": "approximately "}])
    assert result == "The revenue was $50M."

def test_apply_missing_span_returns_none():
    result = _apply_edits("The revenue was $50M.",
                          [{"finding_id": "f1", "operation": "replace",
                            "span": "$75M", "replacement": "$50M"}])
    assert result is None

def test_apply_dedup_finding_ids():
    result = _apply_edits("AAA BBB",
                          [{"finding_id": "f1", "operation": "replace",
                            "span": "AAA", "replacement": "CCC"},
                           {"finding_id": "f1", "operation": "replace",
                            "span": "BBB", "replacement": "DDD"}])
    assert result == "CCC BBB"


# --- 6. Withhold text -------------------------------------------------------

def test_withhold_text_docx():
    text = _withhold_text("Analysis", is_xlsx=False)
    assert "Verification limitation" in text
    assert "Analysis" in text

def test_withhold_text_xlsx():
    text = _withhold_text("Data", is_xlsx=True)
    assert "## Sheet:" in text
    assert "Verification limitation" in text


# --- 7. Full verify_chunk flow ----------------------------------------------

def test_verify_chunk_clean_audit():
    board = _make_board()
    board.add_tokens(5000, 1000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(5000, 1000)
    caller = _MockCaller(['{"findings": []}'])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="The fact is alpha.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert text == "The fact is alpha."
    assert record["status"] == "audit_clean"
    assert ledger.chunks_clean == 1

def test_verify_chunk_corrected():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "beta", "explanation": "should be alpha"},
    ]})
    correction_resp = json.dumps({"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "beta", "replacement": "alpha"},
    ]})
    reaudit_resp = '{"findings": []}'

    caller = _MockCaller([audit_resp, correction_resp, reaudit_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="The answer is beta.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert text == "The answer is alpha."
    assert record["status"] == "corrected"
    assert ledger.chunks_corrected == 1

def test_verify_chunk_withheld_on_reaudit_blockers():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "wrong_entity", "severity": "blocking",
         "span": "Acme", "explanation": "should be Beta Corp"},
    ]})
    correction_resp = json.dumps({"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "Acme", "replacement": "Beta Corp"},
    ]})
    reaudit_resp = json.dumps({"findings": [
        {"id": "f2", "defect": "factual_error", "severity": "blocking",
         "span": "Beta Corp", "explanation": "still wrong"},
    ]})

    caller = _MockCaller([audit_resp, correction_resp, reaudit_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="Acme reported revenue.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "reaudit_blockers"
    assert ledger.chunks_withheld == 1

def test_verify_chunk_withheld_on_invalid_audit():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)
    caller = _MockCaller(['not valid json at all'])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="Some content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "audit_invalid"

def test_verify_chunk_budget_exhausted():
    board = _make_board()
    ledger = VerificationLedger()
    # No draft tokens → budget = 0

    text, record = _verify_chunk(
        _MockCaller(), board, ledger,
        draft="Some content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "budget_exhausted"
    assert ledger.chunks_withheld == 1

def test_verify_chunk_edit_fails_withholds():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "nonexistent span", "explanation": "wrong"},
    ]})
    correction_resp = json.dumps({"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "nonexistent span", "replacement": "correct"},
    ]})

    caller = _MockCaller([audit_resp, correction_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="The actual content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "edit_failed"

def test_verify_chunk_invalid_correction_withholds():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "wrong text", "explanation": "should be X"},
    ]})
    correction_resp = '{"edits": "not a list"}'

    caller = _MockCaller([audit_resp, correction_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="Some wrong text here.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "correct_invalid"


# --- 8. Ledger shared across chunks ----------------------------------------

def test_ledger_shared_across_chunks():
    ledger = VerificationLedger()
    ledger.add_draft_tokens(5000, 500)
    ledger.add_draft_tokens(5000, 500)
    assert ledger.draft_tokens == 11000
    assert ledger.budget == int(11000 * _VERIFICATION_BUDGET_RATIO)

    ledger.charge(100, 50)
    ledger.charge(100, 50)
    assert ledger.spent == 300
    assert ledger.calls == 2


# --- 9. Advisory findings don't trigger correction --------------------------

def test_advisory_only_passes_clean():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "omission", "severity": "advisory",
         "span": "", "explanation": "could add more detail"},
    ]})
    caller = _MockCaller([audit_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="The fact is alpha.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert text == "The fact is alpha."
    assert record["status"] == "audit_clean"


# --- 10. Timing records present in every path --------------------------------

def test_verify_chunk_records_timing():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)
    caller = _MockCaller(['{"findings": []}'])

    _, record = _verify_chunk(
        caller, board, ledger,
        draft="Content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "timing_ms" in record
    assert isinstance(record["timing_ms"], int)
    assert record["timing_ms"] >= 0


def test_verify_chunk_budget_exhausted_records_timing():
    board = _make_board()
    ledger = VerificationLedger()

    _, record = _verify_chunk(
        _MockCaller(), board, ledger,
        draft="Content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert record["status"] == "budget_exhausted"
    assert "timing_ms" in record


# --- 11. Budget reserve prevents undersized dispatch -------------------------

def test_verify_chunk_small_budget_withholds():
    """Budget < _VERIFY_CALL_RESERVE prevents audit dispatch."""
    board = _make_board()
    board.add_tokens(100, 50, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(100, 50)
    assert ledger.budget < _VERIFY_CALL_RESERVE

    text, record = _verify_chunk(
        _MockCaller(), board, ledger,
        draft="Some content.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "budget_exhausted"
    assert ledger.chunks_withheld == 1


# --- 12. Exception safety in apply_edits ------------------------------------

def test_verify_chunk_edit_span_mismatch_withholds():
    """Correction with non-matching span withholds the chunk."""
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "wrong", "explanation": "should be right"},
    ]})
    correction_resp = json.dumps({"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "nonexistent", "replacement": "right"},
    ]})

    caller = _MockCaller([audit_resp, correction_resp])

    text, record = _verify_chunk(
        caller, board, ledger,
        draft="The wrong fact here.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert "Verification limitation" in text
    assert record["status"] == "edit_failed"


# --- 13. Corrected path persists final_findings and pre_barrier_draft --------

def test_verify_chunk_corrected_has_final_findings():
    board = _make_board()
    board.add_tokens(10000, 2000, "fake")
    ledger = VerificationLedger()
    ledger.add_draft_tokens(10000, 2000)

    audit_resp = json.dumps({"findings": [
        {"id": "f1", "defect": "factual_error", "severity": "blocking",
         "span": "beta", "explanation": "should be alpha"},
    ]})
    correction_resp = json.dumps({"edits": [
        {"finding_id": "f1", "operation": "replace",
         "span": "beta", "replacement": "alpha"},
    ]})
    reaudit_resp = '{"findings": []}'

    caller = _MockCaller([audit_resp, correction_resp, reaudit_resp])

    _, record = _verify_chunk(
        caller, board, ledger,
        draft="The answer is beta.", payloads="claim c100: fact alpha",
        claim_ids=["c100"], section_title="Analysis",
        filename="output.docx", is_xlsx=False,
    )
    assert record["status"] == "corrected"
    assert "final_findings" in record
    assert record["final_findings"] == []
    assert "pre_barrier_draft" in record
    assert record["pre_barrier_draft"] == "The answer is beta."
