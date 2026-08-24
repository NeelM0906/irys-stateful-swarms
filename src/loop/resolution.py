"""Canonical target resolution — P3 contract implementation.

One semantic dispatch per target produces a deeply immutable
TargetResolution. Code derives provenance, status, and contradiction
edges — the LLM provides atoms and dispositions but never authoritative
provenance or status. The ResolutionSet is the single source of truth
for downstream planning and rendering.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


# ── Atom types ──

@dataclass(frozen=True)
class Atom:
    """A resolved content unit — the smallest piece of the answer."""
    id: str
    kind: str  # affirmative | gap | uncertainty | limitation | contradiction
    content: str
    support_claim_ids: tuple[str, ...] = ()
    input_claim_ids: tuple[str, ...] = ()
    basis_claim_ids: tuple[str, ...] = ()
    is_required: bool = True

    def __post_init__(self):
        for f in ("support_claim_ids", "input_claim_ids", "basis_claim_ids"):
            v = getattr(self, f)
            if not isinstance(v, tuple):
                object.__setattr__(self, f, tuple(v))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "kind": self.kind,
            "content": self.content,
            "support_claim_ids": list(self.support_claim_ids),
            "input_claim_ids": list(self.input_claim_ids),
            "basis_claim_ids": list(self.basis_claim_ids),
            "is_required": self.is_required,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Atom":
        return cls(
            id=str(data.get("id", "")),
            kind=str(data.get("kind", "affirmative")),
            content=str(data.get("content", "")),
            support_claim_ids=tuple(str(r) for r in data.get("support_claim_ids", ())),
            input_claim_ids=tuple(str(r) for r in data.get("input_claim_ids", ())),
            basis_claim_ids=tuple(str(r) for r in data.get("basis_claim_ids", ())),
            is_required=bool(data.get("is_required", True)),
        )


@dataclass(frozen=True)
class ContradictionEdge:
    """An undirected contradiction between two claims, ordered canonically."""
    claim_a: str
    claim_b: str
    disposition: str  # resolved | deferred | noted
    preference: str  # claim_a | claim_b | neither | ""
    resolution_claim_ids: tuple[str, ...] = ()

    def __post_init__(self):
        if self.claim_a == self.claim_b:
            raise ValueError(f"ContradictionEdge self-edge: {self.claim_a}")
        if self.claim_a > self.claim_b:
            a, b = self.claim_a, self.claim_b
            object.__setattr__(self, "claim_a", b)
            object.__setattr__(self, "claim_b", a)
        if not isinstance(self.resolution_claim_ids, tuple):
            object.__setattr__(self, "resolution_claim_ids",
                               tuple(self.resolution_claim_ids))

    @property
    def pair(self) -> tuple[str, str]:
        return (self.claim_a, self.claim_b)

    def to_dict(self) -> dict:
        return {
            "claim_a": self.claim_a,
            "claim_b": self.claim_b,
            "disposition": self.disposition,
            "preference": self.preference,
            "resolution_claim_ids": list(self.resolution_claim_ids),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContradictionEdge":
        return cls(
            claim_a=str(data.get("claim_a", "")),
            claim_b=str(data.get("claim_b", "")),
            disposition=str(data.get("disposition", "noted")),
            preference=str(data.get("preference", "")),
            resolution_claim_ids=tuple(
                str(r) for r in data.get("resolution_claim_ids", ())
            ),
        )


# ── Per-target resolution ──

@dataclass(frozen=True)
class TargetResolution:
    """Immutable resolution of one serialized target."""
    target_id: str
    atoms: tuple[Atom, ...] = ()
    contradiction_edges: tuple[ContradictionEdge, ...] = ()
    scope_claim_ids: tuple[str, ...] = ()
    decision_claim_ids: tuple[str, ...] = ()
    considered_claim_ids: tuple[str, ...] = ()
    status: str = "unsupported"  # resolved | limited | unsupported
    finish_reason: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    provider_attempts: int = 1

    def __post_init__(self):
        for f in ("atoms", "contradiction_edges", "scope_claim_ids",
                  "decision_claim_ids", "considered_claim_ids"):
            v = getattr(self, f)
            if not isinstance(v, tuple):
                object.__setattr__(self, f, tuple(v))

    @property
    def fingerprint(self) -> str:
        """Content hash for divergence detection (P3-12)."""
        data = json.dumps(
            [a.to_dict() for a in self.atoms],
            sort_keys=True,
        )
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    @property
    def required_atoms(self) -> tuple[Atom, ...]:
        return tuple(a for a in self.atoms if a.is_required)

    @property
    def affirmative_atoms(self) -> tuple[Atom, ...]:
        return tuple(a for a in self.atoms if a.kind == "affirmative")

    def to_dict(self) -> dict:
        return {
            "target_id": self.target_id,
            "atoms": [a.to_dict() for a in self.atoms],
            "contradiction_edges": [e.to_dict() for e in self.contradiction_edges],
            "scope_claim_ids": list(self.scope_claim_ids),
            "decision_claim_ids": list(self.decision_claim_ids),
            "considered_claim_ids": list(self.considered_claim_ids),
            "status": self.status,
            "fingerprint": self.fingerprint,
            "finish_reason": self.finish_reason,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "provider_attempts": self.provider_attempts,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TargetResolution":
        atoms = tuple(Atom.from_dict(a) for a in data.get("atoms", ()))
        edges = tuple(
            ContradictionEdge.from_dict(e)
            for e in data.get("contradiction_edges", ())
        )
        scope = tuple(str(r) for r in data.get("scope_claim_ids", ()))
        decision, considered = _derive_provenance(atoms, edges, scope)
        status = _derive_status(atoms, edges)
        return cls(
            target_id=str(data.get("target_id", "")),
            atoms=atoms,
            contradiction_edges=edges,
            scope_claim_ids=scope,
            decision_claim_ids=decision,
            considered_claim_ids=considered,
            status=status,
            finish_reason=str(data.get("finish_reason", "")),
            input_tokens=int(data.get("input_tokens", 0)),
            output_tokens=int(data.get("output_tokens", 0)),
            provider_attempts=int(data.get("provider_attempts", 1)),
        )


# ── Provenance derivation (P3-3) ──

def _derive_provenance(
    atoms: tuple[Atom, ...],
    contradiction_edges: tuple[ContradictionEdge, ...],
    scope_claim_ids: tuple[str, ...] | frozenset[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Code-derive decision and considered claim sets (P3-3).

    Returns (decision_claim_ids, considered_claim_ids) in board order.
    scope_claim_ids should be a board-ordered tuple for stable sorting.
    """
    scope_set = set(scope_claim_ids) if not isinstance(scope_claim_ids, set) else scope_claim_ids
    decision = set()
    considered = set()

    for a in atoms:
        if a.kind == "affirmative":
            decision.update(a.support_claim_ids)
            decision.update(a.input_claim_ids)
        considered.update(a.support_claim_ids)
        considered.update(a.input_claim_ids)
        if a.kind in ("uncertainty", "limitation"):
            considered.update(a.basis_claim_ids)

    for edge in contradiction_edges:
        considered.add(edge.claim_a)
        considered.add(edge.claim_b)
        considered.update(edge.resolution_claim_ids)

    decision &= scope_set
    considered &= scope_set

    if isinstance(scope_claim_ids, (tuple, list)):
        board_order = {cid: i for i, cid in enumerate(scope_claim_ids)}
    else:
        board_order = {cid: i for i, cid in enumerate(sorted(scope_claim_ids))}
    return (
        tuple(sorted(decision, key=lambda c: board_order.get(c, len(board_order)))),
        tuple(sorted(considered, key=lambda c: board_order.get(c, len(board_order)))),
    )


# ── Status derivation (P3-4) ──

def _derive_status(
    atoms: tuple[Atom, ...],
    contradiction_edges: tuple[ContradictionEdge, ...],
) -> str:
    """Code-derive resolution status from atoms and edges (P3-4).

    resolved: at least one required affirmative atom with evidence support,
    no required gaps/uncertainties, no unresolved contradictions.
    limited: some affirmative content but doesn't meet resolved criteria.
    unsupported: no affirmative content at all.
    """
    has_any_affirmative = any(a.kind == "affirmative" for a in atoms)
    has_supported_required = any(
        a.kind == "affirmative" and a.is_required and a.support_claim_ids
        for a in atoms
    )
    has_gap = any(a.kind in ("gap", "limitation") and a.is_required for a in atoms)
    has_uncertainty = any(a.kind == "uncertainty" and a.is_required for a in atoms)
    has_unresolved = any(e.disposition == "unresolved" for e in contradiction_edges)

    if not has_any_affirmative:
        return "unsupported"
    if has_supported_required and not has_unresolved and not has_uncertainty and not has_gap:
        return "resolved"
    return "limited"


# ── Resolution set (P3-1) ──

@dataclass(frozen=True)
class ResolutionSet:
    """Container for all target resolutions — deeply immutable."""
    targets: tuple[TargetResolution, ...] = ()
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.targets, tuple):
            object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "metadata",
                           MappingProxyType(copy.deepcopy(dict(self.metadata))))

    @property
    def resolved_count(self) -> int:
        return sum(1 for r in self.targets if r.status == "resolved")

    @property
    def limited_count(self) -> int:
        return sum(1 for r in self.targets if r.status == "limited")

    @property
    def unsupported_count(self) -> int:
        return sum(1 for r in self.targets if r.status == "unsupported")

    def get(self, target_id: str) -> TargetResolution | None:
        for r in self.targets:
            if r.target_id == target_id:
                return r
        return None

    def to_dict(self) -> dict:
        return {
            "targets": [r.to_dict() for r in self.targets],
            "metadata": dict(self.metadata),
            "resolved": self.resolved_count,
            "limited": self.limited_count,
            "unsupported": self.unsupported_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ResolutionSet":
        return cls(
            targets=tuple(
                TargetResolution.from_dict(r)
                for r in data.get("targets", ())
            ),
            metadata=data.get("metadata", {}),
        )


# ── Resolver ──

class ResolutionInputCapacityError(RuntimeError):
    pass


class ResolutionOutputCapacityError(RuntimeError):
    pass


class InvalidResolutionResult(ValueError):
    pass


def _build_scope(board, target) -> tuple[str, ...]:
    """Board-ordered claim IDs in scope for this target."""
    claim_ids = set(target.claim_refs)
    board_order = []
    for c in board.claims:
        if c.id in claim_ids and c.active:
            board_order.append(c.id)
    return tuple(board_order)


def _parse_resolution(raw: dict, scope_claim_ids: frozenset[str],
                      target_id: str) -> tuple[tuple[Atom, ...], tuple[ContradictionEdge, ...]]:
    """Parse and validate resolver LLM output into atoms and edges."""
    atoms = []
    seen_atom_ids: set[str] = set()
    for i, a in enumerate(raw.get("atoms") or []):
        if not isinstance(a, dict):
            continue
        aid = str(a.get("id", f"atom_{target_id}_{i}"))
        if aid in seen_atom_ids:
            continue
        seen_atom_ids.add(aid)
        kind = str(a.get("kind", "affirmative"))
        if kind not in ("affirmative", "gap", "uncertainty", "limitation", "contradiction"):
            kind = "affirmative"
        support = tuple(str(r) for r in (a.get("support_claim_ids") or ()) if str(r) in scope_claim_ids)
        inputs = tuple(str(r) for r in (a.get("input_claim_ids") or ()) if str(r) in scope_claim_ids)
        basis = tuple(str(r) for r in (a.get("basis_claim_ids") or ()) if str(r) in scope_claim_ids)
        atoms.append(Atom(
            id=aid,
            kind=kind,
            content=str(a.get("content", "")),
            support_claim_ids=support,
            input_claim_ids=inputs,
            basis_claim_ids=basis,
            is_required=bool(a.get("is_required", True)),
        ))

    edges = []
    seen_pairs: set[tuple[str, str]] = set()
    for e in raw.get("contradiction_edges") or []:
        if not isinstance(e, dict):
            continue
        ca = str(e.get("claim_a", ""))
        cb = str(e.get("claim_b", ""))
        if not ca or not cb or ca == cb:
            continue
        if ca not in scope_claim_ids or cb not in scope_claim_ids:
            continue
        pair = (min(ca, cb), max(ca, cb))
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        disp = str(e.get("disposition", "noted"))
        if disp not in ("resolved", "deferred", "noted", "unresolved"):
            disp = "noted"
        edges.append(ContradictionEdge(
            claim_a=ca,
            claim_b=cb,
            disposition=disp,
            preference=str(e.get("preference", "")),
            resolution_claim_ids=tuple(
                str(r) for r in (e.get("resolution_claim_ids") or ())
                if str(r) in scope_claim_ids
            ),
        ))

    if not atoms:
        atoms.append(Atom(
            id=f"gap_{target_id}",
            kind="gap",
            content="no resolution produced",
            is_required=True,
        ))

    return tuple(atoms), tuple(edges)


def _inject_board_contradictions(
    board, scope_set: frozenset[str], edges: tuple[ContradictionEdge, ...]
) -> tuple[ContradictionEdge, ...]:
    """Inject contradiction edges from board-known contradicts_refs (P3-2).

    Any in-scope claim pair where the board records a contradiction gets a
    'noted' edge if the LLM didn't already produce one for that pair.
    """
    existing = {(e.claim_a, e.claim_b) for e in edges}
    injected = list(edges)
    for c in board.claims:
        if c.id not in scope_set or not c.active:
            continue
        for ref in getattr(c, "contradicts_refs", ()):
            if ref not in scope_set:
                continue
            a, b = (c.id, ref) if c.id < ref else (ref, c.id)
            if a == b or (a, b) in existing:
                continue
            existing.add((a, b))
            injected.append(ContradictionEdge(
                claim_a=a, claim_b=b, disposition="noted",
                preference="",
            ))
    return tuple(injected)


def _resolution_packet(board, target, scope: tuple[str, ...]) -> dict:
    """Full-scope claim packet for resolution — no truncation (P3-5)."""
    scope_set = set(scope)
    claims = []
    for c in board.claims:
        if c.id in scope_set and c.active:
            claims.append({
                "id": c.id,
                "kind": c.kind,
                "content": c.content,
                "evidence": c.evidence,
                "source": c.source_doc,
                "section": c.source_section,
                "verified": c.verified,
                "confidence": round(c.confidence, 2),
                "support_refs": c.support_refs,
                "source_span": list(c.source_span) if c.source_span else None,
            })
    return {
        "id": target.id,
        "need": target.need,
        "materiality": target.materiality,
        "status": target.status,
        "reason": target.reason,
        "claims": claims,
    }


_INPUT_CAPACITY_CHARS = 400_000


def resolve_target(caller, board, target, *,
                   output_ceiling: int = 8192) -> TargetResolution:
    """Resolve one target — exactly one semantic dispatch (P3-6)."""
    from .llm import call_json

    scope = _build_scope(board, target)
    scope_set = frozenset(scope)

    packet = _resolution_packet(board, target, scope)
    claims_text = json.dumps(packet["claims"], indent=1, default=str)

    if len(claims_text) > _INPUT_CAPACITY_CHARS:
        raise ResolutionInputCapacityError(
            f"target {target.id}: {len(claims_text)} chars exceeds "
            f"resolution input capacity ({_INPUT_CAPACITY_CHARS})"
        )

    prompt = f"""You are resolving one investigation question from its evidence. Produce a structured resolution with atoms — distinct content units that together answer the question.

QUESTION: {target.need}
STATUS: {target.status} (reason: {target.reason or 'n/a'})
MATERIALITY: {target.materiality}

EVIDENCE ({len(packet['claims'])} claims):
{claims_text}

INSTRUCTIONS:
1. Produce atoms — each is a distinct finding, conclusion, gap, or limitation.
   - "affirmative": a claim-supported finding or conclusion
   - "gap": something the evidence does not cover
   - "uncertainty": a finding with notable caveats
   - "limitation": a structural limitation of the analysis
   - "contradiction": evidence that conflicts
2. Every affirmative atom MUST reference support_claim_ids from the evidence above.
3. If claims contradict each other, add contradiction_edges with disposition.
4. Mark atoms as is_required=true if they must appear in the deliverable, false if optional context.

Return JSON:
{{"atoms": [{{"id": "a1", "kind": "affirmative|gap|uncertainty|limitation|contradiction", "content": "...", "support_claim_ids": ["c1", ...], "input_claim_ids": ["c2", ...], "basis_claim_ids": ["c3", ...], "is_required": true}}],
  "contradiction_edges": [{{"claim_a": "c1", "claim_b": "c2", "disposition": "resolved|deferred|noted", "preference": "c1|c2|neither", "resolution_claim_ids": ["c3"]}}]}}"""

    events_before = len(board.events)
    raw = call_json(caller, board, prompt, kind="resolve_target",
                    max_tokens=output_ceiling)

    tokens_in = tokens_out = 0
    finish = ""
    attempts = 0
    for ev in board.events[events_before:]:
        if ev.kind == "resolve_target":
            tokens_in += ev.tokens_in
            tokens_out += ev.tokens_out
            attempts += ev.detail.get("provider_attempts", 1)
            fr = ev.detail.get("finish_reason", "")
            if fr:
                finish = fr
    if not finish:
        finish = "stop"
    if attempts == 0:
        attempts = 1

    if finish not in ("stop", "end_turn", ""):
        raise ResolutionOutputCapacityError(
            f"target {target.id}: finish_reason={finish}, output likely truncated"
        )

    if not isinstance(raw, dict):
        raise InvalidResolutionResult(
            f"target {target.id}: LLM returned {type(raw).__name__}, not dict"
        )

    tid = target.target_id if hasattr(target, "target_id") else target.id
    atoms, edges = _parse_resolution(raw, scope_set, tid)

    edges = _inject_board_contradictions(board, scope_set, edges)

    decision_ids, considered_ids = _derive_provenance(atoms, edges, scope)
    status = _derive_status(atoms, edges)

    return TargetResolution(
        target_id=target.id,
        atoms=atoms,
        contradiction_edges=edges,
        scope_claim_ids=scope,
        decision_claim_ids=decision_ids,
        considered_claim_ids=considered_ids,
        status=status,
        finish_reason=finish,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        provider_attempts=attempts,
    )


def resolve_all(caller, board, *, output_ceiling: int = 8192) -> ResolutionSet:
    """Resolve every serialized target on the board (P3-5).

    Includes closed, waived, and blocked targets — all get exactly one
    resolution dispatch. The board is read but never mutated.
    """
    seen_ids: set[str] = set()
    resolutions = []
    for target in board.targets:
        if target.id in seen_ids:
            board.log("resolution", f"skipped duplicate target {target.id}")
            continue
        seen_ids.add(target.id)
        try:
            r = resolve_target(caller, board, target,
                               output_ceiling=output_ceiling)
        except (ResolutionInputCapacityError, ResolutionOutputCapacityError,
                InvalidResolutionResult, TypeError, ValueError) as exc:
            board.log("resolution_error",
                      f"target {target.id}: {type(exc).__name__}: {exc}")
            r = TargetResolution(
                target_id=target.id,
                atoms=(Atom(id=f"gap_{target.id}", kind="gap",
                            content=f"resolution failed: {exc}",
                            is_required=True),),
                status="unsupported",
                finish_reason="error",
            )
        resolutions.append(r)
        board.log(
            "resolution",
            f"resolved {target.id}: {r.status} ({len(r.atoms)} atoms, "
            f"{len(r.contradiction_edges)} edges)",
            detail=r.to_dict(),
        )

    return ResolutionSet(
        targets=tuple(resolutions),
        metadata={"board_iteration": board.iteration},
    )
