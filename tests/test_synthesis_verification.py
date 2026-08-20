"""Tests for cycle-7 synthesis verification barrier.

The verification barrier audits each drafted synthesis chunk against its
serialized claims, catching factual errors the length-only repair acceptance
misses: wrong-party substitution, wrong arithmetic, unsupported classification,
omitted packet-supported facts, and unresolved high-risk defects.
"""
import json
from dataclasses import dataclass

from src.loop.state import Board, Claim, Source, Target
from src.loop.synthesis import (
    _audit_section, _validate_audit, _blocking_findings,
    _correct_findings, _apply_correction_edits, _degrade_unresolved,
    _CONTENT_CAP, _EVIDENCE_CAP, _BLOCKING_DEFECT_TYPES,
)


@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 100
    tokens_output: int = 50
    tokens_total: int = 150
    model: str = "fake"


class _AuditCaller:
    """Returns a scripted JSON response for audit/correction calls."""

    def __init__(self, responses: list[str] | None = None):
        self.prompts: list[str] = []
        self._responses = list(responses or [])
        self._call_idx = 0

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        if self._call_idx < len(self._responses):
            text = self._responses[self._call_idx]
        else:
            text = '{"schema_version":1,"scope_sha256":"x","status":"complete","overflow":false,"findings":[]}'
        self._call_idx += 1
        return _FakeResult(text=text)


def _board_with_claims(claims_data: list[dict]) -> Board:
    board = Board(instruction="analyze the agreement", output_dir="")
    board.add_source(Source(id="s0", name="agreement.pdf",
                            size_bytes=5000))
    board.add_target(Target(id="t0", need="identify issues",
                            materiality="critical", status="closed"))
    for cd in claims_data:
        c = Claim(
            id=cd["id"],
            content=cd["content"],
            evidence=cd.get("evidence", ""),
            kind=cd.get("kind", "analysis"),
            source_doc="agreement.pdf",
            target_refs=["t0"],
            confidence=0.9,
        )
        board.add_claim(c)
        board.find_target("t0").claim_refs.append(c.id)
    return board


def _make_scope_sha(draft, payloads, source_text_block=""):
    import hashlib
    return hashlib.sha256(
        (draft + payloads + source_text_block).encode()
    ).hexdigest()


# --- 1. Audit prompt includes draft and claims ------------------------------

def test_audit_prompt_includes_draft_and_claims():
    """The audit call must see both the draft text and the serialized claims."""
    board = _board_with_claims([
        {"id": "c1", "content": "Key fact about the matter"},
    ])
    draft = "This is the drafted section content."
    payloads = json.dumps([{"id": "c1", "content": "Key fact about the matter"}])
    sha = _make_scope_sha(draft, payloads)

    caller = _AuditCaller()
    _audit_section(caller, board, draft=draft, payloads=payloads,
                   source_text_block="", scope_sha256=sha)
    assert len(caller.prompts) == 1
    prompt = caller.prompts[0]
    assert "This is the drafted section content" in prompt
    assert "Key fact about the matter" in prompt


# --- 2. Clean audit returns no findings ------------------------------------

def test_clean_audit_returns_empty_findings():
    """A correct draft should produce no audit findings."""
    board = _board_with_claims([
        {"id": "c1", "content": "Theodore Langston is the lead partner",
         "evidence": "Matter file: Theodore Langston, lead partner"},
    ])
    draft = "Theodore Langston is the lead partner on this matter."
    payloads = json.dumps([{"id": "c1",
        "content": "Theodore Langston is the lead partner"}])
    sha = _make_scope_sha(draft, payloads)

    clean_response = json.dumps({
        "schema_version": 1, "scope_sha256": sha,
        "status": "complete", "overflow": False, "findings": []
    })
    caller = _AuditCaller(responses=[clean_response])
    result = _audit_section(caller, board, draft=draft, payloads=payloads,
                            source_text_block="", scope_sha256=sha)
    assert result is not None
    assert result.get("findings") == []


# --- 3. Validate audit rejects malformed output ----------------------------

def test_validate_audit_rejects_parse_failure():
    audit, err = _validate_audit(None, draft="x", scope_sha256="x",
                                 item_ids=set())
    assert audit is None
    assert err == "parse_failure"


def test_validate_audit_rejects_incomplete():
    raw = {"status": "insufficient", "findings": []}
    audit, err = _validate_audit(raw, draft="x", scope_sha256="x",
                                 item_ids=set())
    assert audit is None
    assert "incomplete_status" in err


def test_validate_audit_rejects_overflow():
    raw = {"status": "complete", "overflow": True, "findings": []}
    audit, err = _validate_audit(raw, draft="x", scope_sha256="x",
                                 item_ids=set())
    assert audit is None
    assert err == "overflow"


def test_validate_audit_rejects_span_not_in_draft():
    raw = {
        "status": "complete", "overflow": False,
        "findings": [{
            "finding_id": "f1", "defect_type": "wrong_entity",
            "impact": "blocking",
            "draft_span": {"quote": "NONEXISTENT TEXT", "occurrence": 1},
            "claim_ids": [], "supported_correction": "fix",
            "rationale": "test",
        }]
    }
    audit, err = _validate_audit(raw, draft="actual draft text",
                                 scope_sha256="x", item_ids=set())
    assert audit is None
    assert "span_not_in_draft" in err


def test_validate_audit_rejects_unknown_defect():
    raw = {
        "status": "complete", "overflow": False,
        "findings": [{
            "finding_id": "f1", "defect_type": "made_up_type",
            "impact": "blocking",
            "draft_span": {"quote": "some text", "occurrence": 1},
            "claim_ids": [], "supported_correction": "fix",
            "rationale": "test",
        }]
    }
    audit, err = _validate_audit(raw, draft="some text here",
                                 scope_sha256="x", item_ids=set())
    assert audit is None
    assert "unknown_defect_type" in err


def test_validate_audit_rejects_duplicate_ids():
    raw = {
        "status": "complete", "overflow": False,
        "findings": [
            {"finding_id": "f1", "defect_type": "wrong_entity",
             "impact": "blocking",
             "draft_span": {"quote": "text A", "occurrence": 1},
             "claim_ids": [], "supported_correction": "", "rationale": ""},
            {"finding_id": "f1", "defect_type": "wrong_entity",
             "impact": "blocking",
             "draft_span": {"quote": "text B", "occurrence": 1},
             "claim_ids": [], "supported_correction": "", "rationale": ""},
        ]
    }
    audit, err = _validate_audit(raw, draft="text A and text B",
                                 scope_sha256="x", item_ids=set())
    assert audit is None
    assert "duplicate" in err


# --- 4. Blocking classification is code-defined ----------------------------

def test_blocking_findings_classifies_by_defect_type():
    """Code-defined blocking types override model impact."""
    audit = {
        "findings": [
            {"finding_id": "f1", "defect_type": "wrong_entity",
             "impact": "advisory"},
            {"finding_id": "f2", "defect_type": "other",
             "impact": "advisory"},
        ]
    }
    blockers = _blocking_findings(audit)
    assert len(blockers) == 1
    assert blockers[0]["finding_id"] == "f1"


def test_blocking_findings_respects_model_upgrade():
    """Model can upgrade 'other' to blocking, but code types always block."""
    audit = {
        "findings": [
            {"finding_id": "f1", "defect_type": "other",
             "impact": "blocking"},
        ]
    }
    blockers = _blocking_findings(audit)
    assert len(blockers) == 1


# --- 5. Correction edit application ----------------------------------------

def test_apply_replace_edit():
    draft = "The lead partner is Rachel Bernstein on this matter."
    raw = {
        "schema_version": 1, "scope_sha256": "x",
        "edits": [{
            "finding_id": "f1", "operation": "replace",
            "match": "Rachel Bernstein", "occurrence": 1,
            "replacement": "Theodore Langston",
        }]
    }
    findings = [{"finding_id": "f1"}]
    result, err = _apply_correction_edits(draft, raw, findings=findings,
                                          scope_sha256="x")
    assert err is None
    assert "Theodore Langston" in result
    assert "Rachel Bernstein" not in result
    assert "The lead partner is" in result


def test_apply_insert_after_edit():
    draft = "The merger requires regulatory approvals. Additional terms apply."
    raw = {
        "schema_version": 1, "scope_sha256": "x",
        "edits": [{
            "finding_id": "f1", "operation": "insert_after",
            "match": "regulatory approvals.", "occurrence": 1,
            "replacement": " HSR Act filing is required.",
        }]
    }
    findings = [{"finding_id": "f1"}]
    result, err = _apply_correction_edits(draft, raw, findings=findings,
                                          scope_sha256="x")
    assert err is None
    assert "regulatory approvals. HSR Act filing" in result


def test_apply_rejects_overlapping_edits():
    draft = "ABCDEF"
    raw = {
        "edits": [
            {"finding_id": "f1", "operation": "replace",
             "match": "ABCD", "occurrence": 1, "replacement": "XXXX"},
            {"finding_id": "f2", "operation": "replace",
             "match": "CDEF", "occurrence": 1, "replacement": "YYYY"},
        ]
    }
    findings = [{"finding_id": "f1"}, {"finding_id": "f2"}]
    result, err = _apply_correction_edits(draft, raw, findings=findings,
                                          scope_sha256="x")
    assert result is None
    assert "overlapping" in err


def test_apply_rejects_unaddressed_findings():
    draft = "Some text here."
    raw = {"edits": []}
    findings = [{"finding_id": "f1"}]
    result, err = _apply_correction_edits(draft, raw, findings=findings,
                                          scope_sha256="x")
    assert result is None
    assert "unaddressed" in err


def test_apply_rejects_unreported_finding():
    draft = "Some text here."
    raw = {
        "edits": [{
            "finding_id": "f99", "operation": "replace",
            "match": "text", "occurrence": 1, "replacement": "data",
        }]
    }
    findings = [{"finding_id": "f1"}]
    result, err = _apply_correction_edits(draft, raw, findings=findings,
                                          scope_sha256="x")
    assert result is None
    assert "unreported" in err


# --- 6. Degradation -------------------------------------------------------

def test_degrade_replaces_defective_spans():
    draft = "The total is $39,311,600 based on the agreement."
    findings = [{
        "finding_id": "f1", "defect_type": "wrong_computation",
        "draft_span": {"quote": "$39,311,600", "occurrence": 1},
        "claim_ids": ["c1"],
    }]
    result = _degrade_unresolved(draft, filename="memo.docx",
                                 section_title="Summary",
                                 findings=findings, reason="test")
    assert "$39,311,600" not in result
    assert "verified" in result.lower()


def test_degrade_whole_chunk_when_no_spans():
    result = _degrade_unresolved("Some draft content.", filename="memo.docx",
                                 section_title="Analysis",
                                 findings=None, reason="audit_invalid")
    assert "verification limitation" in result.lower()
    assert "audit_invalid" in result


def test_degrade_xlsx_valid_syntax():
    result = _degrade_unresolved("Draft", filename="analysis.xlsx",
                                 section_title="Sheet1",
                                 findings=None, reason="test")
    assert "verification limitation" in result.lower()


# --- 7. All blocking defect types are defined ------------------------------

def test_blocking_types_cover_expected_defects():
    expected = {
        "wrong_entity", "wrong_number", "wrong_date", "wrong_computation",
        "wrong_attribution", "contradiction", "unsupported_assertion",
        "unsupported_classification", "material_omission",
        "requirement_omission", "coverage_omission", "citation_error",
        "internal_metadata_leak",
    }
    assert _BLOCKING_DEFECT_TYPES == expected
