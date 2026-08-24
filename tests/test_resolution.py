"""Tests for P3 canonical resolution data model and resolver."""
import json

import pytest

from src.loop.resolution import (
    Atom, ContradictionEdge, InvalidResolutionResult, ResolutionSet,
    TargetResolution, _derive_provenance, _derive_status, _parse_resolution,
    _inject_board_contradictions,
)


# ── Atom tests ──

class TestAtom:
    def test_frozen(self):
        a = Atom(id="a1", kind="affirmative", content="test")
        with pytest.raises(AttributeError):
            a.content = "changed"

    def test_roundtrip(self):
        a = Atom(
            id="a1", kind="affirmative", content="finding",
            support_claim_ids=("c1", "c2"),
            input_claim_ids=("c3",),
            basis_claim_ids=(),
            is_required=True,
        )
        a2 = Atom.from_dict(a.to_dict())
        assert a2 == a

    def test_defaults(self):
        a = Atom.from_dict({"id": "a1"})
        assert a.kind == "affirmative"
        assert a.is_required is True
        assert a.support_claim_ids == ()


# ── ContradictionEdge tests ──

class TestContradictionEdge:
    def test_canonical_ordering(self):
        e = ContradictionEdge(claim_a="c5", claim_b="c2", disposition="resolved",
                              preference="c2")
        assert e.claim_a == "c2"
        assert e.claim_b == "c5"

    def test_self_edge_rejected(self):
        with pytest.raises(ValueError, match="self-edge"):
            ContradictionEdge(claim_a="c1", claim_b="c1", disposition="noted",
                              preference="")

    def test_roundtrip(self):
        e = ContradictionEdge(
            claim_a="c1", claim_b="c3", disposition="resolved",
            preference="c1", resolution_claim_ids=("c5",),
        )
        e2 = ContradictionEdge.from_dict(e.to_dict())
        assert e2 == e

    def test_frozen(self):
        e = ContradictionEdge(claim_a="c1", claim_b="c2", disposition="noted",
                              preference="")
        with pytest.raises(AttributeError):
            e.disposition = "resolved"


# ── TargetResolution tests ──

class TestTargetResolution:
    def _make(self) -> TargetResolution:
        return TargetResolution(
            target_id="t1",
            atoms=(
                Atom(id="a1", kind="affirmative", content="finding A",
                     support_claim_ids=("c1",)),
                Atom(id="a2", kind="gap", content="missing data",
                     is_required=True),
                Atom(id="a3", kind="affirmative", content="optional note",
                     support_claim_ids=("c2",), is_required=False),
            ),
            contradiction_edges=(
                ContradictionEdge(claim_a="c1", claim_b="c3",
                                  disposition="resolved", preference="c1"),
            ),
            scope_claim_ids=("c1", "c2", "c3"),
            decision_claim_ids=("c1", "c2"),
            considered_claim_ids=("c1", "c2", "c3"),
            status="limited",
        )

    def test_frozen(self):
        r = self._make()
        with pytest.raises(AttributeError):
            r.status = "resolved"

    def test_deep_immutability(self):
        """Mutating input lists after construction doesn't affect the resolution."""
        atoms_list = [Atom(id="a1", kind="affirmative", content="test")]
        scope_list = ["c1", "c2"]
        r = TargetResolution(
            target_id="t1",
            atoms=atoms_list,
            scope_claim_ids=scope_list,
        )
        atoms_list.append(Atom(id="a2", kind="gap", content="added"))
        scope_list.append("c3")
        assert len(r.atoms) == 1
        assert len(r.scope_claim_ids) == 2
        assert isinstance(r.atoms, tuple)
        assert isinstance(r.scope_claim_ids, tuple)

    def test_atom_list_to_tuple_conversion(self):
        """Atom converts list inputs to tuples."""
        a = Atom(id="a1", kind="affirmative", content="test",
                 support_claim_ids=["c1", "c2"])
        assert isinstance(a.support_claim_ids, tuple)
        assert a.support_claim_ids == ("c1", "c2")

    def test_metadata_immutable(self):
        """ResolutionSet.metadata is read-only after construction."""
        from src.loop.resolution import ResolutionSet
        rs = ResolutionSet(metadata={"key": "val"})
        with pytest.raises(TypeError):
            rs.metadata["key"] = "changed"

    def test_metadata_nested_isolation(self):
        """Nested mutable values in metadata are deep-copied."""
        from src.loop.resolution import ResolutionSet
        nested = ["a", "b"]
        rs = ResolutionSet(metadata={"nested": nested})
        nested.append("c")
        assert list(rs.metadata["nested"]) == ["a", "b"]

    def test_resolution_set_targets_list_to_tuple(self):
        """ResolutionSet converts a list of targets to tuple."""
        from src.loop.resolution import ResolutionSet
        targets_list = [TargetResolution(target_id="t1")]
        rs = ResolutionSet(targets=targets_list)
        assert isinstance(rs.targets, tuple)
        targets_list.append(TargetResolution(target_id="t2"))
        assert len(rs.targets) == 1

    def test_input_claim_ids_in_decision(self):
        """input_claim_ids from affirmative atoms enter decision set."""
        atoms = (Atom(id="a1", kind="affirmative", content="F",
                      support_claim_ids=("c1",), input_claim_ids=("c2",)),)
        decision, considered = _derive_provenance(atoms, (), ("c1", "c2"))
        assert "c2" in decision
        assert "c2" in considered

    def test_fingerprint_stable(self):
        r = self._make()
        assert r.fingerprint == r.fingerprint
        assert len(r.fingerprint) == 16

    def test_fingerprint_changes_on_content(self):
        r1 = TargetResolution(
            target_id="t1",
            atoms=(Atom(id="a1", kind="affirmative", content="version A"),),
        )
        r2 = TargetResolution(
            target_id="t1",
            atoms=(Atom(id="a1", kind="affirmative", content="version B"),),
        )
        assert r1.fingerprint != r2.fingerprint

    def test_required_atoms(self):
        r = self._make()
        assert len(r.required_atoms) == 2
        assert all(a.is_required for a in r.required_atoms)

    def test_affirmative_atoms(self):
        r = self._make()
        assert len(r.affirmative_atoms) == 2

    def test_roundtrip(self):
        r = self._make()
        d = r.to_dict()
        r2 = TargetResolution.from_dict(d)
        assert r2.target_id == r.target_id
        assert len(r2.atoms) == len(r.atoms)
        assert r2.status == r.status

    def test_json_roundtrip(self):
        r = self._make()
        s = json.dumps(r.to_dict(), default=str)
        r2 = TargetResolution.from_dict(json.loads(s))
        assert r2.fingerprint == r.fingerprint


# ── Provenance derivation tests (P3-3) ──

class TestDeriveProvenance:
    def test_basic(self):
        atoms = (
            Atom(id="a1", kind="affirmative", content="F",
                 support_claim_ids=("c1", "c2")),
            Atom(id="a2", kind="uncertainty", content="U",
                 basis_claim_ids=("c3",)),
        )
        scope = frozenset(("c1", "c2", "c3", "c4"))
        decision, considered = _derive_provenance(atoms, (), scope)
        assert set(decision) == {"c1", "c2"}
        assert set(considered) == {"c1", "c2", "c3"}

    def test_contradiction_edges_in_considered(self):
        edges = (
            ContradictionEdge(claim_a="c1", claim_b="c2",
                              disposition="resolved", preference="c1",
                              resolution_claim_ids=("c3",)),
        )
        scope = frozenset(("c1", "c2", "c3"))
        decision, considered = _derive_provenance((), edges, scope)
        assert decision == ()
        assert set(considered) == {"c1", "c2", "c3"}

    def test_out_of_scope_filtered(self):
        atoms = (
            Atom(id="a1", kind="affirmative", content="F",
                 support_claim_ids=("c1", "c_outside")),
        )
        scope = frozenset(("c1",))
        decision, considered = _derive_provenance(atoms, (), scope)
        assert set(decision) == {"c1"}
        assert "c_outside" not in considered

    def test_board_order_preserved(self):
        atoms = (
            Atom(id="a1", kind="affirmative", content="F",
                 support_claim_ids=("c3", "c1")),
        )
        scope = ("c1", "c2", "c3")
        decision, _ = _derive_provenance(atoms, (), scope)
        assert decision == ("c1", "c3")


# ── Status derivation tests (P3-4) ──

class TestDeriveStatus:
    def test_resolved(self):
        atoms = (Atom(id="a1", kind="affirmative", content="F",
                      support_claim_ids=("c1",), is_required=True),)
        assert _derive_status(atoms, ()) == "resolved"

    def test_unsupported_affirmative_is_limited(self):
        """Affirmative atom without evidence support is 'limited', not 'resolved'."""
        atoms = (Atom(id="a1", kind="affirmative", content="F"),)
        assert _derive_status(atoms, ()) == "limited"

    def test_optional_affirmative_is_limited(self):
        """Optional affirmative atom is 'limited', not 'resolved'."""
        atoms = (Atom(id="a1", kind="affirmative", content="F",
                      support_claim_ids=("c1",), is_required=False),)
        assert _derive_status(atoms, ()) == "limited"

    def test_limited_with_uncertainty(self):
        atoms = (
            Atom(id="a1", kind="affirmative", content="F"),
            Atom(id="a2", kind="uncertainty", content="U", is_required=True),
        )
        assert _derive_status(atoms, ()) == "limited"

    def test_limited_with_gap(self):
        atoms = (
            Atom(id="a1", kind="affirmative", content="F"),
            Atom(id="a2", kind="gap", content="G", is_required=True),
        )
        assert _derive_status(atoms, ()) == "limited"

    def test_limited_with_unresolved_contradiction(self):
        atoms = (Atom(id="a1", kind="affirmative", content="F"),)
        edges = (ContradictionEdge(claim_a="c1", claim_b="c2",
                                   disposition="unresolved", preference=""),)
        assert _derive_status(atoms, edges) == "limited"

    def test_unsupported_gap_only(self):
        atoms = (Atom(id="a1", kind="gap", content="nothing", is_required=True),)
        assert _derive_status(atoms, ()) == "unsupported"

    def test_unsupported_limitation_only(self):
        atoms = (Atom(id="a1", kind="limitation", content="cannot answer",
                      is_required=True),)
        assert _derive_status(atoms, ()) == "unsupported"

    def test_optional_gap_no_affirmative(self):
        atoms = (Atom(id="a1", kind="gap", content="G", is_required=False),)
        assert _derive_status(atoms, ()) == "unsupported"

    def test_resolved_contradiction_doesnt_block(self):
        atoms = (Atom(id="a1", kind="affirmative", content="F",
                      support_claim_ids=("c1",), is_required=True),)
        edges = (ContradictionEdge(claim_a="c1", claim_b="c2",
                                   disposition="resolved", preference="c1"),)
        assert _derive_status(atoms, edges) == "resolved"


# ── Parse resolution tests ──

class TestParseResolution:
    def test_basic_parse(self):
        raw = {
            "atoms": [
                {"id": "a1", "kind": "affirmative", "content": "finding",
                 "support_claim_ids": ["c1"]},
            ],
        }
        scope = frozenset(("c1", "c2"))
        atoms, edges = _parse_resolution(raw, scope, "t1")
        assert len(atoms) == 1
        assert atoms[0].support_claim_ids == ("c1",)

    def test_out_of_scope_claims_filtered(self):
        raw = {
            "atoms": [
                {"id": "a1", "kind": "affirmative", "content": "F",
                 "support_claim_ids": ["c1", "c_bad"]},
            ],
        }
        scope = frozenset(("c1",))
        atoms, _ = _parse_resolution(raw, scope, "t1")
        assert atoms[0].support_claim_ids == ("c1",)

    def test_duplicate_atom_ids_deduped(self):
        raw = {
            "atoms": [
                {"id": "a1", "kind": "affirmative", "content": "first"},
                {"id": "a1", "kind": "gap", "content": "duplicate"},
            ],
        }
        atoms, _ = _parse_resolution(raw, frozenset(), "t1")
        assert len(atoms) == 1
        assert atoms[0].content == "first"

    def test_empty_produces_gap_atom(self):
        atoms, _ = _parse_resolution({}, frozenset(), "t1")
        assert len(atoms) == 1
        assert atoms[0].kind == "gap"

    def test_self_edges_rejected(self):
        raw = {
            "atoms": [{"id": "a1", "kind": "affirmative", "content": "F"}],
            "contradiction_edges": [
                {"claim_a": "c1", "claim_b": "c1", "disposition": "noted"},
            ],
        }
        scope = frozenset(("c1",))
        _, edges = _parse_resolution(raw, scope, "t1")
        assert len(edges) == 0

    def test_dangling_edge_endpoints_rejected(self):
        raw = {
            "atoms": [{"id": "a1", "kind": "affirmative", "content": "F"}],
            "contradiction_edges": [
                {"claim_a": "c1", "claim_b": "c_bad", "disposition": "noted"},
            ],
        }
        scope = frozenset(("c1",))
        _, edges = _parse_resolution(raw, scope, "t1")
        assert len(edges) == 0

    def test_duplicate_reverse_edges_deduped(self):
        raw = {
            "atoms": [{"id": "a1", "kind": "affirmative", "content": "F"}],
            "contradiction_edges": [
                {"claim_a": "c1", "claim_b": "c2", "disposition": "noted"},
                {"claim_a": "c2", "claim_b": "c1", "disposition": "resolved"},
            ],
        }
        scope = frozenset(("c1", "c2"))
        _, edges = _parse_resolution(raw, scope, "t1")
        assert len(edges) == 1

    def test_invalid_kind_normalized(self):
        raw = {
            "atoms": [
                {"id": "a1", "kind": "bogus", "content": "test"},
            ],
        }
        atoms, _ = _parse_resolution(raw, frozenset(), "t1")
        assert atoms[0].kind == "affirmative"


# ── Board contradiction injection tests ──

class TestBoardContradictionInjection:
    def test_injects_missing_edge(self):
        """Board contradicts_refs create noted edges when LLM missed them."""
        from types import SimpleNamespace as NS
        scope_set = frozenset(("c1", "c2"))
        board = NS(claims=[
            NS(id="c1", active=True, contradicts_refs=["c2"]),
            NS(id="c2", active=True, contradicts_refs=[]),
        ])
        edges = _inject_board_contradictions(board, scope_set, ())
        assert len(edges) == 1
        assert edges[0].disposition == "noted"

    def test_no_duplicate_if_llm_covered(self):
        """If LLM already produced the edge, don't inject a duplicate."""
        from types import SimpleNamespace as NS
        scope_set = frozenset(("c1", "c2"))
        existing = (ContradictionEdge(
            claim_a="c1", claim_b="c2", disposition="resolved",
            preference="c1"),)
        board = NS(claims=[
            NS(id="c1", active=True, contradicts_refs=["c2"]),
            NS(id="c2", active=True, contradicts_refs=[]),
        ])
        edges = _inject_board_contradictions(board, scope_set, existing)
        assert len(edges) == 1
        assert edges[0].disposition == "resolved"

    def test_out_of_scope_ignored(self):
        """Contradictions outside scope are not injected."""
        from types import SimpleNamespace as NS
        scope_set = frozenset(("c1",))
        board = NS(claims=[
            NS(id="c1", active=True, contradicts_refs=["c3"]),
        ])
        edges = _inject_board_contradictions(board, scope_set, ())
        assert len(edges) == 0


# ── ResolutionSet tests ──

class TestResolutionSet:
    def test_frozen(self):
        rs = ResolutionSet()
        with pytest.raises(AttributeError):
            rs.targets = ()

    def test_counts(self):
        rs = ResolutionSet(targets=(
            TargetResolution(target_id="t1", status="resolved"),
            TargetResolution(target_id="t2", status="limited"),
            TargetResolution(target_id="t3", status="unsupported"),
            TargetResolution(target_id="t4", status="resolved"),
        ))
        assert rs.resolved_count == 2
        assert rs.limited_count == 1
        assert rs.unsupported_count == 1

    def test_get(self):
        rs = ResolutionSet(targets=(
            TargetResolution(target_id="t1", status="resolved"),
            TargetResolution(target_id="t2", status="limited"),
        ))
        assert rs.get("t1").status == "resolved"
        assert rs.get("t3") is None

    def test_roundtrip(self):
        rs = ResolutionSet(
            targets=(
                TargetResolution(
                    target_id="t1",
                    atoms=(Atom(id="a1", kind="affirmative", content="F",
                                support_claim_ids=("c1",), is_required=True),),
                    scope_claim_ids=("c1",),
                    status="resolved",
                ),
            ),
            metadata={"board_iteration": 5},
        )
        d = rs.to_dict()
        rs2 = ResolutionSet.from_dict(d)
        assert len(rs2.targets) == 1
        assert rs2.get("t1").status == "resolved"
        assert rs2.metadata["board_iteration"] == 5

    def test_json_roundtrip(self):
        rs = ResolutionSet(targets=(
            TargetResolution(target_id="t1", status="resolved",
                             atoms=(Atom(id="a1", kind="affirmative", content="F",
                                         support_claim_ids=("c1",), is_required=True),),
                             scope_claim_ids=("c1",)),
        ))
        s = json.dumps(rs.to_dict(), default=str)
        rs2 = ResolutionSet.from_dict(json.loads(s))
        assert rs2.resolved_count == 1
