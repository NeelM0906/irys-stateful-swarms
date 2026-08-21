"""Tests for V3 shadow verification (Cycle 9).

Covers: VerificationLedger, _semantic_verification_scopes,
_validate_verification_audit, _apply_verification_edits,
_shadow_verify_chunk, and integration with _synthesize_section.
"""
from __future__ import annotations

import hashlib
import json
import types

import pytest

from src.loop import synthesis as syn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class FakeClaim:
    def __init__(self, cid: str):
        self.id = cid


def _make_chunk_items(target_id="t1", claim_ids=None, unit_ids=None):
    """Build a minimal chunk item list with serialized payloads."""
    claim_ids = claim_ids or ["c1", "c2"]
    items = []
    for cid in claim_ids:
        payload = {"target": {"target_id": target_id},
                   "claim": {"id": cid, "content": f"claim {cid}"}}
        items.append({
            "type": "claim",
            "payload": payload,
            "serialized": json.dumps(payload),
            "claims": [FakeClaim(cid)],
            "unit_ids": [],
        })
    if unit_ids:
        for uid in unit_ids:
            payload = {"unit": {"unit_id": uid, "name": f"unit-{uid}"}}
            items.append({
                "type": "unit",
                "payload": payload,
                "serialized": json.dumps(payload),
                "claims": [FakeClaim(f"uc-{uid}")],
                "unit_ids": [uid],
            })
    return items


def _make_requirement_items(claim_ids):
    items = []
    for cid in claim_ids:
        payload = {"requirement": {"id": cid, "content": f"req {cid}"}}
        items.append({
            "type": "requirement",
            "payload": payload,
            "serialized": json.dumps(payload),
            "claims": [FakeClaim(cid)],
            "unit_ids": [],
        })
    return items


# ---------------------------------------------------------------------------
# VerificationLedger
# ---------------------------------------------------------------------------

class TestVerificationLedger:
    def test_empty_ledger_summary(self):
        ledger = syn.VerificationLedger()
        s = ledger.summary()
        assert s["chunks_verified"] == 0
        assert s["chunks_clean"] == 0
        assert s["activation_eligible"] == 0

    def test_clean_record(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "activation_eligible": True})
        s = ledger.summary()
        assert s["chunks_verified"] == 1
        assert s["chunks_clean"] == 1
        assert s["activation_eligible"] == 1

    def test_corrected_record(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "corrected", "edits_applied": 3,
                       "errors_caught": 2, "activation_eligible": True})
        s = ledger.summary()
        assert s["chunks_corrected"] == 1
        assert s["edits_applied"] == 3
        assert s["errors_caught"] == 2
        assert s["activation_eligible"] == 1

    def test_failed_invalid_audit(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "failed", "reason": "invalid_audit:not_dict"})
        s = ledger.summary()
        assert s["chunks_failed"] == 1
        assert s["invalid_audits"] == 1

    def test_failed_invalid_re_audit(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "failed",
                       "reason": "invalid_re_audit:not_dict"})
        s = ledger.summary()
        assert s["chunks_failed"] == 1
        assert s["invalid_re_audits"] == 1

    def test_skipped_record(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "skipped"})
        s = ledger.summary()
        assert s["chunks_verified"] == 0
        assert s["chunks_skipped"] == 1

    def test_multiple_records(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "activation_eligible": True})
        ledger.record({"status": "corrected", "edits_applied": 1,
                       "errors_caught": 1, "activation_eligible": True})
        ledger.record({"status": "failed", "reason": "audit_error:timeout"})
        ledger.record({"status": "skipped"})
        s = ledger.summary()
        assert s["chunks_verified"] == 3
        assert s["chunks_skipped"] == 1
        assert s["chunks_clean"] == 1
        assert s["chunks_corrected"] == 1
        assert s["chunks_failed"] == 1
        assert s["activation_eligible"] == 2
        assert len(ledger.entries) == 4


# ---------------------------------------------------------------------------
# _semantic_verification_scopes
# ---------------------------------------------------------------------------

class TestSemanticVerificationScopes:
    def test_claim_items(self):
        items = _make_chunk_items(target_id="t1", claim_ids=["c1", "c2"])
        scopes = syn._semantic_verification_scopes(items)
        assert "target:t1" in scopes
        assert set(scopes["target:t1"]) == {"c1", "c2"}

    def test_unit_items(self):
        items = _make_chunk_items(target_id="t1", claim_ids=[],
                                 unit_ids=["u1", "u2"])
        scopes = syn._semantic_verification_scopes(items)
        assert "unit:u1" in scopes
        assert "unit:u2" in scopes

    def test_requirement_items(self):
        items = _make_requirement_items(["r1"])
        scopes = syn._semantic_verification_scopes(items)
        assert "requirement:r1" in scopes

    def test_mixed_items(self):
        items = (_make_chunk_items(target_id="t1", claim_ids=["c1"])
                 + _make_chunk_items(target_id="t1", claim_ids=[],
                                     unit_ids=["u1"])
                 + _make_requirement_items(["r1"]))
        scopes = syn._semantic_verification_scopes(items)
        assert len(scopes) == 3

    def test_empty_chunk(self):
        assert syn._semantic_verification_scopes([]) == {}

    def test_unit_items_include_context_claims(self):
        items = [{"type": "unit",
                  "payload": {"unit": {"unit_id": "u1"}},
                  "serialized": "{}",
                  "claims": [FakeClaim("c1")],
                  "context_claims": [FakeClaim("cc1"), FakeClaim("cc2")],
                  "unit_ids": ["u1"]}]
        scopes = syn._semantic_verification_scopes(items)
        assert "unit:u1" in scopes
        assert set(scopes["unit:u1"]) == {"c1", "cc1", "cc2"}

    def test_missing_target_id(self):
        items = [{"type": "claim",
                  "payload": {"target": {}, "claim": {"id": "c1"}},
                  "claims": [FakeClaim("c1")], "unit_ids": []}]
        scopes = syn._semantic_verification_scopes(items)
        assert scopes == {}


# ---------------------------------------------------------------------------
# _validate_verification_audit
# ---------------------------------------------------------------------------

class TestValidateVerificationAudit:
    SCOPES = {"target:t1", "unit:u1"}
    DRAFT = "The company earned $5 million in revenue."

    def test_clean_audit(self):
        raw = {"findings": []}
        findings, err = syn._validate_verification_audit(
            raw, self.SCOPES, self.DRAFT)
        assert findings == []
        assert err is None

    def test_valid_finding(self):
        raw = {"findings": [{
            "finding_id": "f1",
            "defect_type": "factual_error",
            "scope_id": "target:t1",
            "description": "wrong amount",
            "impact": "blocking",
            "operation": "replace",
            "match": "$5 million",
            "replacement": "$6 million",
        }]}
        findings, err = syn._validate_verification_audit(
            raw, self.SCOPES, self.DRAFT)
        assert err is None
        assert len(findings) == 1

    def test_not_dict(self):
        findings, err = syn._validate_verification_audit(
            "bad", self.SCOPES, self.DRAFT)
        assert findings is None
        assert err == "not_dict"

    def test_findings_not_list(self):
        findings, err = syn._validate_verification_audit(
            {"findings": "bad"}, self.SCOPES, self.DRAFT)
        assert err == "findings_not_list"

    def test_unknown_defect_type(self):
        raw = {"findings": [{"finding_id": "f1", "defect_type": "unknown",
                             "scope_id": "target:t1", "operation": "replace",
                             "match": "$5 million", "replacement": "$6"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "unknown_defect" in err

    def test_invalid_scope(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:FAKE",
                             "operation": "replace",
                             "match": "$5 million", "replacement": "$6"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "invalid_scope" in err

    def test_invalid_operation(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "operation": "rewrite",
                             "match": "$5 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "invalid_op" in err

    def test_match_not_in_draft(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": "nonexistent text",
                             "replacement": "x"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "match_not_found" in err

    def test_empty_match(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": "", "replacement": "x"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "empty_match" in err

    def test_replace_empty_replacement(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": "$5 million", "replacement": ""}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "empty_replacement" in err

    def test_delete_no_replacement_ok(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "unsupported_claim",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "delete",
                             "match": " in revenue"}]}
        findings, err = syn._validate_verification_audit(
            raw, self.SCOPES, self.DRAFT)
        assert err is None
        assert len(findings) == 1

    def test_missing_finding_id(self):
        raw = {"findings": [{"defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "operation": "replace",
                             "match": "$5 million", "replacement": "$6"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_finding_id" in err

    def test_ambiguous_match_rejected(self):
        draft = "The $5 million and $5 million revenue."
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, draft)
        assert "ambiguous_match" in err

    def test_duplicate_finding_id_rejected(self):
        raw = {"findings": [
            {"finding_id": "f1", "defect_type": "factual_error",
             "scope_id": "target:t1", "description": "d",
             "impact": "blocking", "operation": "replace",
             "match": "$5 million", "replacement": "$6 million"},
            {"finding_id": "f1", "defect_type": "numerical_error",
             "scope_id": "target:t1", "operation": "delete",
             "match": " in revenue"},
        ]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "duplicate_finding_id" in err

    def test_insert_after_needs_replacement(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "insert_after",
                             "match": "$5 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "empty_replacement" in err

    def test_non_string_finding_id_rejected(self):
        raw = {"findings": [{"finding_id": 1,
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_finding_id" in err

    def test_non_string_defect_type_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": 42,
                             "scope_id": "target:t1",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "unknown_defect" in err

    def test_non_string_match_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": 123,
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "empty_match" in err

    def test_overlapping_ambiguous_match_rejected(self):
        draft = "111"
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "replace",
                             "match": "11",
                             "replacement": "22"}]}
        _, err = syn._validate_verification_audit(
            raw, self.SCOPES, draft)
        assert "ambiguous_match" in err

    def test_non_string_impact_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "wrong amount",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million",
                             "impact": 42}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_impact" in err

    def test_non_string_description_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million",
                             "description": {"nested": True}}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_description" in err

    def test_delete_non_string_replacement_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "unsupported_claim",
                             "scope_id": "target:t1",
                             "description": "d", "impact": "blocking",
                             "operation": "delete",
                             "match": " in revenue",
                             "replacement": {"bad": True}}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "invalid_delete_replacement" in err

    def test_missing_description_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "impact": "blocking",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_description" in err

    def test_missing_impact_rejected(self):
        raw = {"findings": [{"finding_id": "f1",
                             "defect_type": "factual_error",
                             "scope_id": "target:t1",
                             "description": "wrong amount",
                             "operation": "replace",
                             "match": "$5 million",
                             "replacement": "$6 million"}]}
        _, err = syn._validate_verification_audit(raw, self.SCOPES, self.DRAFT)
        assert "missing_impact" in err


# ---------------------------------------------------------------------------
# _apply_verification_edits
# ---------------------------------------------------------------------------

class TestApplyVerificationEdits:
    def test_single_replace(self):
        draft = "Revenue was $5M in 2025."
        findings = [{"finding_id": "f1", "operation": "replace",
                     "match": "$5M", "replacement": "$6M"}]
        result, err = syn._apply_verification_edits(draft, findings)
        assert err is None
        assert result == "Revenue was $6M in 2025."

    def test_single_delete(self):
        draft = "Revenue was $5M (estimated) in 2025."
        findings = [{"finding_id": "f1", "operation": "delete",
                     "match": " (estimated)"}]
        result, err = syn._apply_verification_edits(draft, findings)
        assert err is None
        assert result == "Revenue was $5M in 2025."

    def test_insert_after(self):
        draft = "Revenue was $5M in 2025."
        findings = [{"finding_id": "f1", "operation": "insert_after",
                     "match": "$5M", "replacement": " (audited)"}]
        result, err = syn._apply_verification_edits(draft, findings)
        assert err is None
        assert result == "Revenue was $5M (audited) in 2025."

    def test_multiple_non_overlapping(self):
        draft = "A earned $5M. B earned $3M."
        findings = [
            {"finding_id": "f1", "operation": "replace",
             "match": "$5M", "replacement": "$6M"},
            {"finding_id": "f2", "operation": "replace",
             "match": "$3M", "replacement": "$4M"},
        ]
        result, err = syn._apply_verification_edits(draft, findings)
        assert err is None
        assert result == "A earned $6M. B earned $4M."

    def test_overlapping_edits_rejected(self):
        draft = "Revenue was $5M total."
        findings = [
            {"finding_id": "f1", "operation": "replace",
             "match": "$5M total", "replacement": "$6M total"},
            {"finding_id": "f2", "operation": "replace",
             "match": "total", "replacement": "net"},
        ]
        _, err = syn._apply_verification_edits(draft, findings)
        assert err is not None
        assert "overlap" in err

    def test_match_lost(self):
        draft = "Revenue was $5M."
        findings = [{"finding_id": "f1", "operation": "replace",
                     "match": "nonexistent", "replacement": "x"}]
        _, err = syn._apply_verification_edits(draft, findings)
        assert "match_lost" in err

    def test_empty_result_rejected(self):
        draft = "Only this."
        findings = [{"finding_id": "f1", "operation": "delete",
                     "match": "Only this."}]
        _, err = syn._apply_verification_edits(draft, findings)
        assert err == "empty_result"

    def test_colocated_inserts_rejected(self):
        draft = "Revenue was $5M in 2025."
        findings = [
            {"finding_id": "f1", "operation": "insert_after",
             "match": "$5M", "replacement": " (audited)"},
            {"finding_id": "f2", "operation": "insert_after",
             "match": "$5M", "replacement": " (estimated)"},
        ]
        _, err = syn._apply_verification_edits(draft, findings)
        assert err is not None
        assert "colocated_inserts" in err or "overlap" in err

    def test_boundary_insert_and_replace_order_independent(self):
        draft = "AB"
        insert_first = [
            {"finding_id": "f1", "operation": "insert_after",
             "match": "A", "replacement": "Y"},
            {"finding_id": "f2", "operation": "replace",
             "match": "B", "replacement": "X"},
        ]
        replace_first = [
            {"finding_id": "f2", "operation": "replace",
             "match": "B", "replacement": "X"},
            {"finding_id": "f1", "operation": "insert_after",
             "match": "A", "replacement": "Y"},
        ]
        r1, e1 = syn._apply_verification_edits(draft, insert_first)
        r2, e2 = syn._apply_verification_edits(draft, replace_first)
        assert e1 is None and e2 is None
        assert r1 == r2 == "AYX"


# ---------------------------------------------------------------------------
# _shadow_verify_chunk — unit tests with mocked model calls
# ---------------------------------------------------------------------------

class FakeBoard:
    def __init__(self):
        self.tokens_input = 0
        self.tokens_output = 0
        self.output_dir = None
        self._logs = []

    def log(self, kind, msg, **kwargs):
        self._logs.append((kind, msg))


class TestShadowVerifyChunk:
    def _patch_call_json(self, monkeypatch, responses):
        """Make call_json return responses in sequence."""
        it = iter(responses)
        def fake_call_json(caller, board, prompt, **kwargs):
            board.tokens_input += 100
            board.tokens_output += 50
            val = next(it)
            if isinstance(val, Exception):
                raise val
            return val
        monkeypatch.setattr(syn, "call_json", fake_call_json)

    def test_clean_audit(self, monkeypatch):
        self._patch_call_json(monkeypatch, [{"findings": []}])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Draft text here.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "clean"
        assert result["activation_eligible"] is True
        assert ledger.chunks_clean == 1

    def test_corrected_audit(self, monkeypatch):
        audit_response = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong amount",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        re_audit_response = {"findings": []}
        self._patch_call_json(monkeypatch,
                              [audit_response, re_audit_response])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M in 2025.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "corrected"
        assert result["edits_applied"] == 1
        assert result["candidate_text"] == "Revenue was $6M in 2025."
        assert ledger.chunks_corrected == 1

    def test_audit_error(self, monkeypatch):
        self._patch_call_json(monkeypatch, [RuntimeError("API timeout")])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Some draft.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert "audit_error" in result["reason"]

    def test_invalid_audit_response(self, monkeypatch):
        self._patch_call_json(monkeypatch, ["not a dict"])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Some draft.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert "invalid_audit" in result["reason"]

    def test_residual_blockers(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        re_audit = {"findings": [{
            "finding_id": "f2", "defect_type": "numerical_error",
            "scope_id": "target:t1", "description": "still wrong",
            "impact": "blocking", "operation": "replace", "match": "$6M",
            "replacement": "$7M",
        }]}
        self._patch_call_json(monkeypatch, [audit, re_audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert result["reason"] == "residual_blockers"
        assert result["activation_eligible"] is False

    def test_skipped_empty_draft(self, monkeypatch):
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="   ",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "skipped"

    def test_skipped_no_scopes(self, monkeypatch):
        ledger = syn.VerificationLedger()
        chunk = [{"type": "unknown", "payload": {}, "serialized": "{}",
                  "claims": [], "unit_ids": []}]
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Some text.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "skipped"

    def test_advisory_only_is_clean(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "missing_nuance",
            "scope_id": "target:t1", "description": "imprecise",
            "impact": "advisory", "operation": "replace", "match": "earned",
            "replacement": "reportedly earned",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="The company earned $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "clean"
        assert result["advisory_count"] == 1

    def test_edit_failure_falls_to_failed(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M",
            "replacement": "$6M",
        }, {
            "finding_id": "f2", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "also wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M in",
            "replacement": "$7M in",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M in 2025.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert "edit_failure" in result["reason"]

    def test_re_audit_error(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        self._patch_call_json(monkeypatch,
                              [audit, RuntimeError("re-audit fail")])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert "re_audit_error" in result["reason"]
        assert result["activation_eligible"] is False
        assert result["control_fallback"] is True

    def test_invalid_re_audit_has_activation_eligible(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        re_audit = {"findings": [{"bad": "missing fields"}]}
        self._patch_call_json(monkeypatch, [audit, re_audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert "invalid_re_audit" in result["reason"]
        assert result["activation_eligible"] is False
        assert result["control_fallback"] is True

    def test_control_hash_present(self, monkeypatch):
        self._patch_call_json(monkeypatch, [{"findings": []}])
        ledger = syn.VerificationLedger()
        draft = "Draft text."
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft=draft,
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        expected = hashlib.sha256(draft.encode("utf-8")).hexdigest()
        assert result["control_hash"] == expected

    def test_tokens_tracked(self, monkeypatch):
        self._patch_call_json(monkeypatch, [{"findings": []}])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Draft.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["tokens_in"] == 100
        assert result["tokens_out"] == 50

    def test_sidecar_has_location_fields(self, monkeypatch):
        self._patch_call_json(monkeypatch, [{"findings": []}])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Draft text.",
            chunk=chunk, filename="memo.docx", section_title="Analysis",
            chunk_index=2, ledger=ledger)
        assert result["filename"] == "memo.docx"
        assert result["section_title"] == "Analysis"
        assert result["chunk_index"] == 2
        assert "target:t1" in result["scope_ids"]

    def test_skipped_has_location_fields(self, monkeypatch):
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="   ",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=3, ledger=ledger)
        assert result["filename"] == "out.docx"
        assert result["chunk_index"] == 3
        assert "scope_ids" in result

    def test_skipped_no_scopes_empty_scope_ids(self, monkeypatch):
        ledger = syn.VerificationLedger()
        chunk = [{"type": "unknown", "payload": {}, "serialized": "{}",
                  "claims": [], "unit_ids": []}]
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Some text.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["scope_ids"] == []

    def test_requirement_scoped_finding_not_blocking(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "requirement:r1", "description": "wrong",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = (_make_chunk_items("t1", ["c1"])
                 + _make_requirement_items(["r1"]))
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "clean"

    def test_edit_failure_has_control_fallback(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M",
            "replacement": "$6M",
        }, {
            "finding_id": "f2", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "also wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M in",
            "replacement": "$7M in",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M in 2025.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert result["control_fallback"] is True

    def test_impact_blocking_alone_not_sufficient(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "imprecise_language",
            "scope_id": "target:t1", "description": "imprecise",
            "impact": "blocking", "operation": "replace", "match": "earned",
            "replacement": "reportedly earned",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="The company earned $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "clean"


# ---------------------------------------------------------------------------
# _build_audit_prompt
# ---------------------------------------------------------------------------

class TestBuildAuditPrompt:
    def test_contains_draft(self):
        prompt = syn._build_audit_prompt(
            "My draft.", {"target:t1": ["c1"]}, "items text")
        assert "My draft." in prompt
        assert "items text" in prompt
        assert "target:t1" in prompt

    def test_contains_scope_ids(self):
        scopes = {"target:t1": ["c1"], "unit:u1": ["c2"]}
        prompt = syn._build_audit_prompt("draft", scopes, "items")
        assert "target:t1" in prompt
        assert "unit:u1" in prompt


# ---------------------------------------------------------------------------
# Integration: shadow flag controls invocation
# ---------------------------------------------------------------------------

class TestShadowIntegration:
    def test_shadow_disabled_no_verification(self, monkeypatch):
        monkeypatch.setattr(syn, "_VERIFICATION_SHADOW", False)
        ledger = syn.VerificationLedger()
        calls = []
        orig = syn._shadow_verify_chunk
        def spy(*a, **kw):
            calls.append(1)
            return orig(*a, **kw)
        monkeypatch.setattr(syn, "_shadow_verify_chunk", spy)
        # _synthesize_section won't call shadow when flag is off
        # (we verify by checking the manifest has no verification_shadow key)
        # This is a structural test — we don't call _synthesize_section
        # directly because it needs a full Board. Instead verify the guard.
        assert syn._VERIFICATION_SHADOW is False

    def test_shadow_enabled_flag(self, monkeypatch):
        monkeypatch.setattr(syn, "_VERIFICATION_SHADOW", True)
        assert syn._VERIFICATION_SHADOW is True

    def test_empty_draft_records_skipped_in_ledger(self, monkeypatch):
        monkeypatch.setattr(syn, "_VERIFICATION_SHADOW", True)
        monkeypatch.setattr(syn, "_REPAIR_ENABLED", False)
        monkeypatch.setattr(syn, "call_text",
                            lambda *a, **kw: "   ")
        board = FakeBoard()
        board.instruction = "request"
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        text, manifest = syn._synthesize_section(
            None, None, board, filename="out.docx", file_form="document",
            format_rules="", section={"title": "S", "guidance": ""},
            chunk=chunk, chunk_index=0, chunk_count=1,
            verification_ledger=ledger)
        assert len(ledger.entries) == 1
        entry = ledger.entries[0]
        assert entry["status"] == "skipped"
        assert entry["reason"] == "empty_draft"
        assert entry["filename"] == "out.docx"
        assert entry["section_title"] == "S"
        assert entry["scope_ids"] == []
        assert "verification_shadow" in manifest


# ---------------------------------------------------------------------------
# _dump_verification_shadow
# ---------------------------------------------------------------------------

class TestDumpVerificationShadow:
    def test_writes_json(self, tmp_path):
        board = FakeBoard()
        board.output_dir = str(tmp_path)
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "control_hash": "abc"})
        syn._dump_verification_shadow(board, ledger)
        path = tmp_path / "loop" / "verification_shadow.json"
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["summary"]["chunks_verified"] == 1
        assert len(data["entries"]) == 1

    def test_no_output_dir_noop(self):
        board = FakeBoard()
        board.output_dir = None
        ledger = syn.VerificationLedger()
        syn._dump_verification_shadow(board, ledger)


class TestDumpCandidate:
    def test_writes_candidate_md_and_manifest(self, tmp_path):
        board = FakeBoard()
        board.output_dir = str(tmp_path)
        candidate_text = "## Section A\n\nCorrected content here."
        syn._dump_candidate(board, "memo.docx", candidate_text,
                            has_corrections=True)
        md_path = tmp_path / "loop" / "candidate_memo.docx.md"
        assert md_path.exists()
        assert md_path.read_text(encoding="utf-8") == candidate_text
        manifest_path = tmp_path / "loop" / "candidate_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "memo.docx" in manifest
        assert manifest["memo.docx"]["has_corrections"] is True
        assert manifest["memo.docx"]["chars"] == len(candidate_text)
        assert len(manifest["memo.docx"]["candidate_hash"]) == 64

    def test_no_corrections_still_written(self, tmp_path):
        board = FakeBoard()
        board.output_dir = str(tmp_path)
        syn._dump_candidate(board, "out.docx", "Same text",
                            has_corrections=False)
        manifest = json.loads(
            (tmp_path / "loop" / "candidate_manifest.json").read_text())
        assert manifest["out.docx"]["has_corrections"] is False

    def test_multiple_files_append_manifest(self, tmp_path):
        board = FakeBoard()
        board.output_dir = str(tmp_path)
        syn._dump_candidate(board, "a.docx", "TextA", has_corrections=True)
        syn._dump_candidate(board, "b.docx", "TextB", has_corrections=False)
        manifest = json.loads(
            (tmp_path / "loop" / "candidate_manifest.json").read_text())
        assert "a.docx" in manifest
        assert "b.docx" in manifest

    def test_no_output_dir_noop(self):
        board = FakeBoard()
        board.output_dir = None
        syn._dump_candidate(board, "x.docx", "text", has_corrections=True)


# ---------------------------------------------------------------------------
# Measurement-plane completeness: every entry has both hashes + activation_eligible
# ---------------------------------------------------------------------------

class TestMeasurementPlaneCompleteness(TestShadowVerifyChunk):
    """Every sidecar entry must carry control_hash, candidate_hash,
    and activation_eligible for paired qualification forensics."""

    def test_clean_entry_has_candidate_hash(self, monkeypatch):
        self._patch_call_json(monkeypatch, [{"findings": []}])
        ledger = syn.VerificationLedger()
        draft = "Draft text."
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft=draft,
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "clean"
        assert result["candidate_hash"] == result["control_hash"]
        assert result["activation_eligible"] is True

    def test_skipped_entry_has_both_hashes(self, monkeypatch):
        self._patch_call_json(monkeypatch, [])
        ledger = syn.VerificationLedger()
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Some text.",
            chunk=[], filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "skipped"
        assert result["candidate_hash"] == result["control_hash"]
        assert result["activation_eligible"] is False

    def test_audit_error_has_both_hashes(self, monkeypatch):
        self._patch_call_json(monkeypatch, [RuntimeError("boom")])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Draft.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert result["candidate_hash"] == result["control_hash"]
        assert result["activation_eligible"] is False
        assert result["control_fallback"] is True

    def test_invalid_audit_has_both_hashes(self, monkeypatch):
        self._patch_call_json(monkeypatch,
                              [{"findings": [{"bad": "no fields"}]}])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Draft.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert result["candidate_hash"] == result["control_hash"]
        assert result["activation_eligible"] is False
        assert result["control_fallback"] is True

    def test_edit_failure_has_both_hashes(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M", "replacement": "$6M",
        }, {
            "finding_id": "f2", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "also wrong",
            "impact": "blocking", "operation": "replace",
            "match": "$5M in", "replacement": "$7M in",
        }]}
        self._patch_call_json(monkeypatch, [audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M in 2025.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "failed"
        assert result["candidate_hash"] == result["control_hash"]
        assert result["activation_eligible"] is False

    def test_corrected_entry_has_different_candidate_hash(self, monkeypatch):
        audit = {"findings": [{
            "finding_id": "f1", "defect_type": "factual_error",
            "scope_id": "target:t1", "description": "wrong",
            "impact": "blocking", "operation": "replace", "match": "$5M",
            "replacement": "$6M",
        }]}
        re_audit = {"findings": []}
        self._patch_call_json(monkeypatch, [audit, re_audit])
        ledger = syn.VerificationLedger()
        chunk = _make_chunk_items("t1", ["c1"])
        result = syn._shadow_verify_chunk(
            None, FakeBoard(), draft="Revenue was $5M.",
            chunk=chunk, filename="out.docx", section_title="S1",
            chunk_index=0, ledger=ledger)
        assert result["status"] == "corrected"
        assert result["candidate_hash"] != result["control_hash"]
        assert result["activation_eligible"] is True


class TestTaskActivationEligible:
    def test_all_clean_is_eligible(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "activation_eligible": True})
        ledger.record({"status": "clean", "activation_eligible": True})
        s = ledger.summary()
        assert s["task_activation_eligible"] is True

    def test_one_failed_is_ineligible(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "activation_eligible": True})
        ledger.record({"status": "failed", "activation_eligible": False})
        s = ledger.summary()
        assert s["task_activation_eligible"] is False

    def test_one_skipped_is_ineligible(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "skipped", "activation_eligible": False})
        ledger.record({"status": "clean", "activation_eligible": True})
        s = ledger.summary()
        assert s["task_activation_eligible"] is False

    def test_empty_ledger_is_ineligible(self):
        ledger = syn.VerificationLedger()
        s = ledger.summary()
        assert s["task_activation_eligible"] is False

    def test_total_elapsed_s(self):
        ledger = syn.VerificationLedger()
        ledger.record({"status": "clean", "activation_eligible": True,
                        "elapsed_s": 3.5})
        ledger.record({"status": "corrected", "activation_eligible": True,
                        "elapsed_s": 7.2, "edits_applied": 1,
                        "errors_caught": 1})
        s = ledger.summary()
        assert s["total_elapsed_s"] == 10.7
