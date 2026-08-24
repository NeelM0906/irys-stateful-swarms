"""Tests for Board.from_snapshot — P2 snapshot replay."""
import json
import tempfile
from pathlib import Path

import pytest

from src.loop.state import (
    Board, Claim, Event, Obligation, Source, Target, Unit,
)


def _make_board() -> Board:
    """Build a board with representative entities of every type."""
    b = Board(instruction="analyze the contract", metadata={"task_id": "t42"})
    b.add_source(Source(id="s1", name="contract.pdf", path="/docs/contract.pdf",
                        kind="document", size_bytes=50000, read_status="read",
                        relevance="definite", relevance_reason="primary doc"))
    b.add_source(Source(id="s2", name="addendum.docx", path="/docs/addendum.docx",
                        kind="document", size_bytes=12000, read_status="unread",
                        relevance="likely"))
    b.add_target(Target(id="t1", need="identify key obligations",
                        materiality="critical", claim_refs=["c1", "c2"],
                        created_iteration=1, proposed_by="seed"))
    b.add_target(Target(id="t2", need="compare pricing terms",
                        materiality="high", status="closed", reason="covered",
                        claim_refs=["c3"], resolved_iteration=3,
                        closure_basis=["c3"]))
    b.add_claim(Claim(id="c1", kind="observation", content="payment due in 30 days",
                      source_doc="s1", source_section="Section 4",
                      evidence="Net 30 terms", source_span=(1200, 1350),
                      target_refs=["t1"], confidence=0.9, iteration=1,
                      created_by="read"))
    b.add_claim(Claim(id="c2", kind="analysis", content="payment terms are standard",
                      support_refs=["c1"], target_refs=["t1"],
                      confidence=0.8, iteration=2, created_by="analyze"))
    b.add_claim(Claim(id="c3", kind="observation", content="price is $100/unit",
                      source_doc="s1", source_section="Section 2",
                      target_refs=["t2"], confidence=0.95, iteration=1,
                      created_by="read", verified=True))
    b.add_obligation(Obligation(id="o1", text="list all payment terms",
                                coverage="exhaustive", mandatory=True))
    b.add_obligation(Obligation(id="o2", text="summarize pricing",
                                coverage="material", mandatory=False,
                                status="satisfied", reason="done"))
    b.add_unit(Unit(id="u1", name="net 30 payment", obligation_ref="o1",
                    claim_refs=["c1"], status="evidenced"))
    b.add_unit(Unit(id="u2", name="late fee", obligation_ref="o1",
                    status="discovered"))
    b.iteration = 4
    b.total_tokens_used = 150000
    b.tokens_input = 120000
    b.tokens_output = 30000
    b.token_budget = 500000
    b.cost_by_model = {"flash": {"input": 100000, "output": 25000, "total": 125000, "calls": 8}}
    b.log("controller", "iter 4: decided", detail={"actions": ["read"]})
    b.log("action_summary", "iter 4: 2 claims", detail={"claims": 2})
    return b


def _snapshot_data(board: Board) -> dict:
    """Produce a snapshot dict matching what Board.snapshot() would write."""
    return {
        "schema_version": 2,
        "instruction": board.instruction,
        "metadata": dict(board.metadata),
        "iteration": board.iteration,
        "stop_reason": board.stop_reason,
        "sources": [s.to_dict() for s in board.sources],
        "claims": [c.to_dict() for c in board.claims],
        "targets": [
            {**t.to_dict(), "blockers": board.target_blockers(t)}
            for t in board.targets
        ],
        "obligations": [o.to_dict() for o in board.obligations],
        "units": [u.to_dict() for u in board.units],
        "events": [e.to_dict() for e in board.events],
        "total_tokens_used": board.total_tokens_used,
        "tokens_input": board.tokens_input,
        "tokens_output": board.tokens_output,
        "token_budget": board.token_budget,
        "budget_used_pct": board.budget_used_pct(),
        "cost_by_model": dict(board.cost_by_model),
    }


# --- from_dict roundtrip tests ---

class TestTargetFromDict:
    def test_roundtrip(self):
        t = Target(id="t1", need="Q", materiality="high", status="closed",
                   reason="done", claim_refs=["c1", "c2"],
                   created_iteration=1, resolved_iteration=3,
                   proposed_by="seed", closure_basis=["c1"])
        t2 = Target.from_dict(t.to_dict())
        assert t2.id == "t1"
        assert t2.need == "Q"
        assert t2.materiality == "high"
        assert t2.status == "closed"
        assert t2.resolved_iteration == 3
        assert t2.closure_basis == ["c1"]

    def test_defaults(self):
        t = Target.from_dict({})
        assert t.status == "open"
        assert t.materiality == "medium"
        assert t.resolved_iteration is None


class TestObligationFromDict:
    def test_roundtrip(self):
        o = Obligation(id="o1", text="cover all", coverage="exhaustive",
                       mandatory=True, status="open")
        o2 = Obligation.from_dict(o.to_dict())
        assert o2.id == "o1"
        assert o2.text == "cover all"
        assert o2.coverage == "exhaustive"
        assert o2.mandatory is True

    def test_defaults(self):
        o = Obligation.from_dict({})
        assert o.mandatory is True
        assert o.status == "open"
        assert o.origin == "instruction"


class TestUnitFromDict:
    def test_roundtrip(self):
        u = Unit(id="u1", name="item", obligation_ref="o1",
                 anchor="sec 3", claim_refs=["c1", "c2"],
                 status="evidenced", reason="found it")
        u2 = Unit.from_dict(u.to_dict())
        assert u2.id == "u1"
        assert u2.claim_refs == ["c1", "c2"]
        assert u2.status == "evidenced"

    def test_defaults(self):
        u = Unit.from_dict({})
        assert u.status == "discovered"


class TestSourceFromDict:
    def test_roundtrip(self):
        s = Source(id="s1", name="doc.pdf", path="/tmp/doc.pdf",
                   kind="document", size_bytes=5000,
                   read_status="read", relevance="definite",
                   relevance_reason="primary")
        s2 = Source.from_dict(s.to_dict())
        assert s2.id == "s1"
        assert s2.size_bytes == 5000
        assert s2.read_status == "read"
        assert s2._doc is None
        assert s2._section_index is None

    def test_defaults(self):
        s = Source.from_dict({})
        assert s.kind == "document"
        assert s.read_status == "unread"
        assert s.relevance == "unknown"


class TestEventFromDict:
    def test_roundtrip(self):
        e = Event(iteration=3, kind="controller", summary="decided",
                  detail={"actions": ["read"]}, model="flash",
                  tokens=500, tokens_in=400, tokens_out=100)
        e2 = Event.from_dict(e.to_dict())
        assert e2.iteration == 3
        assert e2.kind == "controller"
        assert e2.detail == {"actions": ["read"]}
        assert e2.tokens_in == 400

    def test_defaults(self):
        e = Event.from_dict({})
        assert e.kind == ""
        assert e.tokens == 0


# --- Board.from_snapshot tests ---

class TestBoardFromSnapshot:
    def test_full_roundtrip(self):
        """All entities survive a snapshot→from_snapshot cycle."""
        orig = _make_board()
        data = _snapshot_data(orig)
        restored = Board.from_snapshot(data)

        assert restored.instruction == orig.instruction
        assert restored.metadata == orig.metadata
        assert restored.iteration == orig.iteration
        assert restored.total_tokens_used == orig.total_tokens_used
        assert restored.tokens_input == orig.tokens_input
        assert restored.tokens_output == orig.tokens_output
        assert restored.token_budget == orig.token_budget
        assert restored.cost_by_model == orig.cost_by_model
        assert len(restored.sources) == len(orig.sources)
        assert len(restored.claims) == len(orig.claims)
        assert len(restored.targets) == len(orig.targets)
        assert len(restored.obligations) == len(orig.obligations)
        assert len(restored.units) == len(orig.units)
        assert len(restored.events) == len(orig.events)

    def test_indices_rebuilt(self):
        """Runtime indices work after restore."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)

        assert b.find_claim("c1") is not None
        assert b.find_claim("c1").content == "payment due in 30 days"
        assert b.find_target("t1") is not None
        assert b.find_target("t1").need == "identify key obligations"
        assert b.find_source("s1") is not None
        assert b.find_obligation("o1") is not None
        assert b.find_unit("u1") is not None

    def test_counters_reconstructed(self):
        """ID counters pick up after the highest existing ID."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)

        assert b._claim_counter == 3
        assert b._target_counter == 2
        assert b._obligation_counter == 2
        assert b._unit_counter == 2

        new_claim = Claim(kind="analysis", content="unique new insight")
        b.add_claim(new_claim)
        assert new_claim.id == "c4"

        new_target = Target(need="new question")
        b.add_target(new_target)
        assert new_target.id == "t3"

    def test_dedup_preserved(self):
        """Content dedup index rebuilt — duplicates rejected after restore."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)

        dup = Claim(kind="observation", content="payment due in 30 days",
                    source_doc="s1", source_span=(1200, 1350))
        assert b.add_claim(dup) is False

    def test_metadata_override(self):
        """metadata_override replaces snapshot metadata."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data, metadata_override={"new_key": "val"})
        assert b.metadata == {"new_key": "val"}

    def test_strict_mode_clean(self):
        """Strict mode passes on a valid graph."""
        orig = _make_board()
        data = _snapshot_data(orig)
        Board.from_snapshot(data, strict=True)

    def test_strict_mode_bad_claim_ref(self):
        """Strict mode catches dangling claim→target refs."""
        orig = _make_board()
        data = _snapshot_data(orig)
        data["claims"][0]["target_refs"] = ["t_nonexistent"]
        with pytest.raises(ValueError, match="missing target"):
            Board.from_snapshot(data, strict=True)

    def test_strict_mode_bad_target_ref(self):
        """Strict mode catches dangling target→claim refs."""
        orig = _make_board()
        data = _snapshot_data(orig)
        data["targets"][0]["claim_refs"] = ["c_nonexistent"]
        with pytest.raises(ValueError, match="missing claim"):
            Board.from_snapshot(data, strict=True)

    def test_strict_mode_bad_unit_obligation(self):
        """Strict mode catches dangling unit→obligation refs."""
        orig = _make_board()
        data = _snapshot_data(orig)
        data["units"][0]["obligation_ref"] = "o_nonexistent"
        with pytest.raises(ValueError, match="missing obligation"):
            Board.from_snapshot(data, strict=True)

    def test_json_roundtrip(self):
        """Snapshot survives JSON serialization (the real persistence path)."""
        orig = _make_board()
        data = _snapshot_data(orig)
        json_str = json.dumps(data, indent=2, default=str)
        loaded = json.loads(json_str)
        b = Board.from_snapshot(loaded, strict=True)
        assert len(b.claims) == 3
        assert b.find_claim("c1").source_span == (1200, 1350)

    def test_file_roundtrip(self):
        """Full file write→read→restore cycle."""
        orig = _make_board()
        with tempfile.TemporaryDirectory() as td:
            orig.output_dir = td
            orig.snapshot("test")
            path = Path(td) / "loop" / f"board_iter_{orig.iteration}_test.json"
            assert path.exists()
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["schema_version"] == 2
            assert data["metadata"] == {"task_id": "t42"}
            b = Board.from_snapshot(data, strict=True, output_dir=td)
            assert b.output_dir == td
            assert len(b.claims) == 3

    def test_schema_v1_compat(self):
        """Snapshots without schema_version or events still load."""
        data = {
            "instruction": "old task",
            "iteration": 2,
            "stop_reason": "",
            "sources": [],
            "claims": [],
            "targets": [],
            "obligations": [],
            "units": [],
            "total_tokens_used": 1000,
            "tokens_input": 800,
            "tokens_output": 200,
            "token_budget": 100000,
            "cost_by_model": {},
        }
        b = Board.from_snapshot(data)
        assert b.instruction == "old task"
        assert b.iteration == 2
        assert len(b.events) == 0
        assert b.metadata == {}

    def test_bind_claim_works_after_restore(self):
        """bind_claim uses restored indices correctly."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)
        new_claim = Claim(kind="observation", content="unique finding xyz")
        b.add_claim(new_claim)
        changed = b.bind_claim(new_claim.id, ["t1"])
        assert changed is True
        assert new_claim.id in b.find_target("t1").claim_refs

    def test_resolve_target_works_after_restore(self):
        """resolve_target uses restored indices correctly."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)
        assert b.resolve_target("t1", "closed", "fully answered") is True
        assert b.find_target("t1").status == "closed"

    def test_unit_dedup_works_after_restore(self):
        """Unit name dedup index rebuilt — duplicates merge."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)
        dup_unit = Unit(name="net 30 payment", obligation_ref="o1",
                        claim_refs=["c2"])
        result = b.add_unit(dup_unit)
        assert result.id == "u1"
        assert "c2" in result.claim_refs

    def test_events_restored(self):
        """Events roundtrip through snapshot."""
        orig = _make_board()
        data = _snapshot_data(orig)
        b = Board.from_snapshot(data)
        assert len(b.events) == 2
        assert b.events[0].kind == "controller"
        assert b.events[1].kind == "action_summary"

    def test_empty_snapshot(self):
        """Minimal empty snapshot loads without error."""
        b = Board.from_snapshot({})
        assert b.instruction == ""
        assert len(b.claims) == 0
        assert b._claim_counter == 0
