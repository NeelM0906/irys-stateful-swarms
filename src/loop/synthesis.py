"""Synthesis — planner + per-deliverable generation.

By the time we get here the thinking is done: targets are resolved and
their claims carry the analysis. The planner is the intelligent act of
allocating closed targets to deliverables (the same target can feed a
memo as a summary and a spreadsheet as a full calculation table). Each
deliverable then gets its own synthesis call — editorial work, not
analytical work.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .hydration import build_evidence_context
from .llm import call_json, call_text
from .state import Board, Target

_CLAIMS_PER_TARGET = 48
_CONTENT_CAP = 500
_EVIDENCE_CAP = 220
_REPAIR_ENABLED = os.getenv("LOOP_SYNTHESIS_REPAIR", "1").strip() in (
    "1", "true", "yes",
)
_REPAIR_DRAFT_CAP = int(os.getenv("LOOP_SYNTHESIS_REPAIR_DRAFT_CAP", "120000"))

_VERIFICATION_ENABLED = os.getenv("LOOP_SYNTHESIS_VERIFY", "1").strip() in (
    "1", "true", "yes",
)
_VERIFICATION_BUDGET_RATIO = 0.15
_AUDIT_MAX_FINDINGS = 12
class VerificationLedger:
    """Task-level budget for all verification calls (audit + correction + re-audit).

    Created once at synthesize() entry; shared across every file/section/chunk.
    Budget = 15% of cumulative synthesis drafting tokens (input+output).
    The budget grows as draft calls accumulate; verification calls are charged
    against the running budget.
    """

    def __init__(self) -> None:
        self.draft_tokens = 0
        self.spent = 0
        self.calls = 0
        self.chunks_audited = 0
        self.chunks_corrected = 0
        self.chunks_withheld = 0
        self.chunks_clean = 0

    @property
    def budget(self) -> int:
        return int(self.draft_tokens * _VERIFICATION_BUDGET_RATIO)

    def add_draft_tokens(self, input_tokens: int, output_tokens: int) -> None:
        self.draft_tokens += input_tokens + output_tokens

    def can_reserve(self) -> bool:
        return self.spent < self.budget

    def charge(self, input_tokens: int, output_tokens: int) -> None:
        self.spent += input_tokens + output_tokens
        self.calls += 1

    def summary(self) -> dict:
        return {
            "draft_tokens": self.draft_tokens,
            "budget": self.budget,
            "spent": self.spent,
            "calls": self.calls,
            "chunks_audited": self.chunks_audited,
            "chunks_corrected": self.chunks_corrected,
            "chunks_withheld": self.chunks_withheld,
            "chunks_clean": self.chunks_clean,
        }

_SYNTHESIS_HYDRATE = os.getenv("LOOP_SYNTHESIS_HYDRATE", "0").strip().lower() in (
    "1", "true", "yes",
)
_SYNTHESIS_HYDRATE_MAX = int(os.getenv("LOOP_SYNTHESIS_HYDRATE_MAX_CHARS", "400000"))


def _dedup_claims(claims, cap: int) -> list:
    """Remove near-duplicate claims by content fingerprint, keep highest-confidence."""
    seen: set[str] = set()
    out = []
    for c in claims:
        key = c.content[:200].lower().strip()
        if key not in seen:
            seen.add(key)
            out.append(c)
        if len(out) >= cap:
            break
    return out


def target_packet(board: Board, target: Target,
                  cap: int | None = _CLAIMS_PER_TARGET) -> dict:
    """Everything synthesis may use for one target.

    cap=None exposes every deduplicated active claim — the section-local
    chunker takes all of them and splits across calls instead of dropping
    claims at a count cap.
    """
    bound = board.claims_for_target(target)
    derived = sorted(
        (c for c in bound if c.is_derived),
        key=lambda c: -c.confidence,
    )
    raw = sorted(
        (c for c in bound if not c.is_derived),
        key=lambda c: -c.confidence,
    )
    ordered = derived + raw
    picked = _dedup_claims(ordered, cap if cap is not None else len(ordered))
    return {
        "id": target.id,
        "need": target.need,
        "materiality": target.materiality,
        "status": target.status,
        "reason": target.reason,
        "claims": [
            {
                "id": c.id,
                "kind": c.kind,
                "content": c.content[:_CONTENT_CAP],
                "evidence": c.evidence[:_EVIDENCE_CAP],
                "source": c.source_doc,
                "section": c.source_section,
                "verified": c.verified,
                "confidence": round(c.confidence, 2),
                "support_refs": c.support_refs,
                "source_span": list(c.source_span) if c.source_span else None,
            }
            for c in picked
        ],
        "_claim_objects": picked,
    }


def unit_packets(board: Board, obligation_ids: list[str] | None = None) -> list[dict]:
    """Unit-preserving packets: every non-waived unit survives into
    synthesis. Within-unit summarization is allowed; unit omission is not.
    """
    packets = []
    for ob in board.obligations:
        if not ob.set_valued or ob.status == "waived":
            continue
        if obligation_ids is not None and ob.id not in obligation_ids:
            continue
        units = [u for u in board.units_for(ob.id) if u.status != "waived"]
        if not units:
            continue
        # Per-unit claim budget shrinks as unit count grows — units are
        # never dropped, their evidence is just summarized harder.
        per_unit = max(3, min(8, 240 // max(len(units), 1)))
        rows = []
        for u in units:
            claims = [
                c for c in (board.find_claim(cid) for cid in u.claim_refs)
                if c is not None and c.active
            ]
            claims.sort(key=lambda c: (not c.is_derived, -c.confidence))
            picked = claims[:per_unit]
            rows.append({
                "unit": u.name,
                "unit_id": u.id,
                "anchor": u.anchor,
                "status": u.status,
                "claims": [
                    {"id": c.id, "kind": c.kind, "content": c.content,
                     "evidence": c.evidence, "source": c.source_doc,
                     "support_refs": c.support_refs,
                     "source_span": list(c.source_span) if c.source_span else None}
                    for c in picked
                ] or [{"kind": "gap", "content": "no evidence gathered for this unit"}],
                "_claim_objects": picked,
            })
        packets.append({
            "obligation": ob.text,
            "obligation_id": ob.id,
            "coverage": ob.coverage,
            "units": rows,
        })
    return packets


def requirement_block(board: Board) -> str:
    """All requirement claims — deliverable constraints discovered in sources.

    These bypass packet caps: a requirement is binding regardless of which
    target it is bound to.
    """
    reqs = [c for c in board.claims if c.active and c.kind == "requirement"]
    return "\n".join(
        f"- {c.content}" + (f" (Source: {c.source_doc})" if c.source_doc else "")
        for c in reqs
    )


def plan_synthesis(smart_caller, board: Board) -> dict:
    """Allocate targets to deliverables — form is decided late, by judgment."""
    deliverables = board.metadata.get("deliverables", {})
    files = list(deliverables.values()) if deliverables else ["output.docx"]

    target_lines = "\n".join(
        f"{t.id} [{t.status}/{t.materiality}] {t.need}"
        f" ({len(t.claim_refs)} claims)"
        for t in board.targets
    )
    ob_lines = "\n".join(
        f"{o.id} [{o.status}/{o.coverage}/{'mandatory' if o.mandatory else 'optional'}]"
        f" {o.text} | {len([u for u in board.units_for(o.id) if u.status != 'waived'])} units"
        for o in board.obligations
    )

    prompt = f"""You are planning the final deliverable(s) of a completed investigation. All analytical work is done — your job is allocation and structure: which resolved questions feed which file, in what order, at what depth, in what form.

REQUEST:
{board.instruction}

ANSWER SHAPE: {board.metadata.get('answer_shape', '')}

OUTPUT FILES REQUIRED: {json.dumps(files)}

ANSWER CONTRACT (obligations the deliverables must satisfy; set-valued ones track units):
{ob_lines or '(none)'}

RESOLVED AND OPEN QUESTIONS:
{target_lines}

{f'''DELIVERABLE REQUIREMENTS DISCOVERED IN SOURCES (binding — the plan must satisfy every one):
{requirement_block(board)}
''' if requirement_block(board) else ''}
Rules:
- The same question can feed multiple files DIFFERENTLY (summary in a memo, full table in a spreadsheet, clause edits in a redline). Allocate accordingly.
- .xlsx files need data-shaped sections (tables); .docx files need prose/structured documents. Match form to file type and to what the request actually asks for.
- Closed targets carry the substance. Waived/blocked/open targets with critical/high materiality must appear in a limitations note, never silently dropped.
- COVERAGE PLAN: every mandatory exhaustive/material/native-complete obligation MUST be placed — say where its units are rendered (one row/subsection/clause per unit, in the source's own order/numbering when one exists), and list the required slots each unit must carry IF the obligation demands repeated fields (e.g. identifier, both sources' positions, difference, severity, quantified impact, recommendation). Derive slots from what the obligation's language demands — never invent ceremony for a summary obligation.

Return JSON:
{{"files": [{{
  "filename": "<exact filename>",
  "form": "<what kind of document this is, in plain words>",
  "sections": [{{"title": "...", "target_ids": ["..."], "guidance": "<depth/form for this section>"}}],
  "coverage": [{{"obligation_id": "...", "section": "<which section renders its units>", "unit_mode": "row|subsection|clause|inline", "required_slots": ["..."]}}]
}}]}}
Every required file must appear. Every mandatory set-valued obligation must appear in some file's coverage list."""

    parsed = call_json(smart_caller, board, prompt, kind="synthesis_plan",
                       max_tokens=8192)
    # Coverage guard: a mandatory set-valued obligation with units may not be
    # left unplaced — fail loudly into the plan, never silently.
    if isinstance(parsed, dict) and parsed.get("files"):
        covered = {
            str(c.get("obligation_id", ""))
            for f in parsed["files"] if isinstance(f, dict)
            for c in f.get("coverage", []) if isinstance(c, dict)
        }
        for ob in board.obligations:
            if (ob.set_valued and ob.mandatory and ob.status != "waived"
                    and board.units_for(ob.id) and ob.id not in covered):
                first = parsed["files"][0]
                first.setdefault("coverage", []).append({
                    "obligation_id": ob.id,
                    "section": "Coverage Appendix",
                    "unit_mode": "subsection",
                    "required_slots": [],
                })
                board.log("synthesis_plan",
                          f"coverage guard: {ob.id} unplaced — appended fallback")
    if not isinstance(parsed, dict) or not parsed.get("files"):
        # Fallback: all material targets into each file, flat.
        closed_ids = [t.id for t in board.targets if t.status == "closed"]
        parsed = {"files": [
            {"filename": f, "form": "document",
             "sections": [{"title": "Analysis", "target_ids": closed_ids,
                           "guidance": "complete answer"}]}
            for f in files
        ]}
        board.log("synthesis_plan", "planner fallback: flat allocation")
    return parsed


_SECTION_CHUNK_CAP = 400_000  # existing packet ceiling, now per-call, fixed pre-smoke
_MAX_ITEMS_PER_CHUNK = int(os.getenv("LOOP_SYNTHESIS_MAX_ITEMS_PER_CHUNK", "300"))


def _norm_section_key(name: str) -> str:
    return " ".join(str(name).split()).casefold()


def _claim_row(c) -> dict:
    return {
        "id": c.id, "kind": c.kind,
        "content": c.content[:_CONTENT_CAP],
        "evidence": c.evidence[:_EVIDENCE_CAP],
        "source": c.source_doc, "section": c.source_section,
        "verified": c.verified, "confidence": round(c.confidence, 2),
        "support_refs": c.support_refs,
        "source_span": list(c.source_span) if c.source_span else None,
    }


def _eligible_section_items(board: Board, file_plan: dict) -> list[dict]:
    """Enumerate individually atomic claim/unit/requirement items eligible for
    each planned section, each carrying its target/obligation context.
    Eligible = active claims bound to the section's targets (uncapped,
    deduplicated per section) + units for coverage obligations routed to that
    section + binding requirement claims. Unbound claims stay out of scope."""
    coverage = [c for c in file_plan.get("coverage", []) if isinstance(c, dict)]
    cov_by_section: dict[str, list[dict]] = {}
    for c in coverage:
        cov_by_section.setdefault(
            _norm_section_key(c.get("section", "")), []).append(c)

    req_claims = [c for c in board.claims
                  if c.active and c.kind == "requirement"]

    # Binding requirements are per-call governing context: a distinct role
    # repeated into every chunk of every section (tracked as requirement_ids,
    # never mixed into the serialize-once claim-item stream).
    requirement_items = [
        {"type": "requirement", "payload": {"requirement": _claim_row(rc)},
         "claims": [rc], "unit_ids": []}
        for rc in req_claims
    ]
    req_ids = {rc.id for rc in req_claims}

    def _unit_items(cov_entry: dict, seen_obligations: set[str],
                    seen_claim_ids: set[str],
                    governing_req_ids: set[str]) -> list[dict]:
        ob_id = str(cov_entry.get("obligation_id", ""))
        if ob_id in seen_obligations:
            return []  # an obligation's units render exactly once per file
        seen_obligations.add(ob_id)
        out = []
        for up in unit_packets(board, obligation_ids=[ob_id]):
            header = {"obligation": up["obligation"],
                      "obligation_id": up["obligation_id"],
                      "coverage": up["coverage"],
                      "coverage_plan": cov_entry}
            for row in up.get("units", []):
                claim_objs = row.pop("_claim_objects", [])
                # Section-wide identity distinguishes CANONICAL serialization
                # from per-call resolved CONTEXT: a claim already canonical in
                # this section keeps its full row inside the unit (every unit
                # call must be self-resolvable), but is charged and observed
                # as context, never double-counted in the canonical stream.
                canonical = [c for c in claim_objs
                             if c.id not in seen_claim_ids]
                # A governing requirement is ALREADY fully present in every
                # call via the preamble — a unit referencing it resolves
                # in-call by reference, never by a second serialized copy.
                req_refs = [c.id for c in claim_objs
                            if c.id in governing_req_ids]
                context = [c for c in claim_objs
                           if c.id in seen_claim_ids
                           and c.id not in governing_req_ids]
                for c in canonical:
                    seen_claim_ids.add(c.id)
                row = dict(row)
                if req_refs:
                    row["claims"] = [cr for cr in row.get("claims", [])
                                     if cr.get("id") not in req_refs]
                    row["requirement_claims_in_preamble"] = req_refs
                if context:
                    row["context_claims_canonical_elsewhere"] = [
                        c.id for c in context]
                unit_id = str(row.get("unit_id", "")) or str(row.get("unit", ""))
                out.append({"type": "unit",
                            "payload": {**header, "unit": row},
                            "claims": canonical,
                            "context_claims": context,
                            "unit_ids": [unit_id]})
        return out

    sections_out: list[dict] = []
    seen_obligations: set[str] = set()
    for sec in file_plan.get("sections", []):
        title = str(sec.get("title", ""))
        items: list[dict] = []
        # Requirements govern every call; a requirement claim never repeats
        # as an ordinary claim item within the section.
        seen_claim_ids: set[str] = set(req_ids)
        seen_tids: set[str] = set()
        for tid in [str(t) for t in sec.get("target_ids", [])]:
            if tid in seen_tids:
                continue
            seen_tids.add(tid)
            t = board.find_target(tid)
            if t is None:
                continue
            pkt = target_packet(board, t, cap=None)
            claim_objs = pkt.pop("_claim_objects", [])
            header = {"target_id": pkt["id"], "need": pkt["need"],
                      "materiality": pkt["materiality"],
                      "status": pkt["status"], "reason": pkt["reason"]}
            for c in claim_objs:
                if c.id in seen_claim_ids:
                    continue  # duplicate within the section serializes once
                seen_claim_ids.add(c.id)
                items.append({"type": "claim",
                              "payload": {"target": header,
                                          "claim": _claim_row(c)},
                              "claims": [c], "unit_ids": []})
        for c in cov_by_section.get(_norm_section_key(title), []):
            items.extend(_unit_items(c, seen_obligations, seen_claim_ids, req_ids))
        sections_out.append({"title": title,
                             "guidance": str(sec.get("guidance", "")),
                             "requirements": requirement_items,
                             "items": items})

    # Coverage routed to an unknown section name uses the existing Coverage
    # Appendix fallback — deterministic, never dropped, never misrouted, and
    # governed by the same binding requirements as every other section.
    known = {_norm_section_key(s["title"]) for s in sections_out}
    appendix_items: list[dict] = []
    appendix_seen: set[str] = set(req_ids)
    for c in coverage:
        if _norm_section_key(c.get("section", "")) in known:
            continue
        appendix_items.extend(
            _unit_items(c, seen_obligations, appendix_seen, req_ids))
    if appendix_items:
        sections_out.append({"title": "Coverage Appendix",
                             "guidance": "render every unit exactly once",
                             "requirements": requirement_items,
                             "items": appendix_items})
    return sections_out


_CHUNK_SEP = "\n"


def _chunk_section_items(items: list[dict], char_cap: int,
                         requirements: list[dict] | None = None
                         ) -> list[list[dict]]:
    """Deterministic atomic chunks: each item (one claim, one unit) is
    serialized whole; packing accounts for the exact joined payload length
    including separators; the split is always BETWEEN items. Requirements are
    per-call governing context prepended to EVERY chunk, and their length is
    charged against each chunk's cap. Only a genuinely indivisible single
    item (or the requirement preamble itself) may exceed the cap."""
class EmptyAssemblyError(RuntimeError):
    """Every section of a required deliverable produced no synthesized body.
    A structurally empty assembly must fail loudly — it is never represented
    as a completed work product."""


class ChunkCapacityError(ValueError):
    """The governing requirement preamble plus one bounded item cannot fit
    the per-call ceiling — a structural failure, never an over-cap call.
    Carries a `detail` dict identifying the offending composition."""

    def __init__(self, message: str, detail: dict | None = None):
        super().__init__(message)
        self.detail = detail or {}


def _chunk_section_items(items: list[dict], char_cap: int,
                         requirements: list[dict] | None = None,
                         max_items: int = 0,
                         ) -> list[list[dict]]:
    """Deterministic atomic chunks: each item (one claim, one unit) is
    serialized whole; packing accounts for the exact joined payload length
    including separators; the split is always BETWEEN items. Requirements are
    per-call governing context prepended to EVERY chunk, and their length is
    charged against each chunk's cap — including the first item. Only a
    genuinely indivisible single item with NO governing preamble may exceed
    the cap alone; a preamble+item that cannot fit fails explicitly.
    max_items caps items per chunk regardless of character budget."""
    if not items:
        return [[]]  # empty-section semantics take precedence over preamble
    req = [{**r, "serialized": json.dumps(r["payload"], indent=1, default=str)}
           for r in (requirements or [])]
    req_len = (sum(len(r["serialized"]) for r in req)
               + len(_CHUNK_SEP) * len(req))
    if req and req_len >= char_cap:
        raise ChunkCapacityError(
            f"requirement preamble ({req_len} chars) exceeds the per-call "
            f"ceiling ({char_cap})",
            detail={"reason": "preamble_exceeds_cap",
                    "preamble_chars": req_len, "cap": char_cap,
                    "requirement_ids": [c.id for r in req
                                        for c in r.get("claims", [])]})
    chunks: list[list[dict]] = []
    cur: list[dict] = []
    cur_len = req_len
    for it in items:
        s = json.dumps(it["payload"], indent=1, default=str)
        length = len(s) + (len(_CHUNK_SEP) if cur else 0)
        if cur and (cur_len + length > char_cap
                    or (max_items and len(cur) >= max_items)):
            chunks.append(req + cur)
            cur = []
            cur_len = req_len
            length = len(s)
        if not cur and req and req_len + length > char_cap:
            raise ChunkCapacityError(
                f"requirement preamble ({req_len}) + item ({length}) exceeds "
                f"the per-call ceiling ({char_cap})",
                detail={"reason": "preamble_plus_item_exceeds_cap",
                        "preamble_chars": req_len, "item_chars": length,
                        "cap": char_cap, "item_type": it.get("type"),
                        "item_claim_ids": [c.id for c in it.get("claims", [])],
                        "item_unit_ids": it.get("unit_ids", [])})
        cur.append({**it, "serialized": s})
        cur_len += length
    if cur:
        chunks.append(req + cur)
    return chunks or [[]]


def _synthesize_section(caller, repairer, board: Board, *, filename: str,
                        file_form: str, format_rules: str,
                        section: dict, chunk: list[dict], chunk_index: int,
                        chunk_count: int,
                        sec_chunks: list | None = None,
                        ledger: VerificationLedger | None = None,
                        ) -> tuple[str, dict]:
    """Draft (and, when enabled, verify) exactly one section chunk against
    exactly its serialized items. Returns (text, chunk_manifest).

    The chunk manifest is registered into sec_chunks BEFORE the model call
    and mutated in place — a thrown call still persists the exact payload
    that reached the model."""
    payloads = _CHUNK_SEP.join(it["serialized"] for it in chunk)
    requirement_ids = [c.id for it in chunk if it["type"] == "requirement"
                       for c in it.get("claims", [])]
    claim_ids = [c.id for it in chunk if it["type"] != "requirement"
                 for c in it.get("claims", [])]
    context_claim_ids = [c.id for it in chunk
                         for c in it.get("context_claims", [])]
    unit_ids = [u for it in chunk for u in it.get("unit_ids", [])]

    seen_cov: set[str] = set()
    cov_entries = []
    for it in chunk:
        if it["type"] != "unit":
            continue
        cp = it["payload"].get("coverage_plan", {})
        ob = str(cp.get("obligation_id"))
        if ob in seen_cov:
            continue
        seen_cov.add(ob)
        cov_entries.append(cp)
    coverage_lines = "\n".join(
        f"- {cp.get('obligation_id')}: render every unit exactly once as "
        f"{cp.get('unit_mode', 'subsection')}"
        + (", each unit carrying: "
           + ", ".join(str(s) for s in cp.get('required_slots', []))
           if cp.get("required_slots") else "")
        for cp in cov_entries
    )

    hydration_stats: dict = {}
    source_text_block = ""
    # Hydration covers every claim object physically present in this call:
    # ordinary claims, requirement claims, and resolved context copies.
    chunk_claims = [c for it in chunk for c in it.get("claims", [])]
    chunk_claims += [c for it in chunk for c in it.get("context_claims", [])]
    if _SYNTHESIS_HYDRATE and chunk_claims:
        evidence_context, hydration_stats = build_evidence_context(
            board, chunk_claims, max_chars=_SYNTHESIS_HYDRATE_MAX,
        )
        board.log(
            "synthesis_hydrate",
            f"{filename} / {section['title']} chunk {chunk_index + 1}: "
            f"{hydration_stats.get('merged_windows', 0)} windows, "
            f"{hydration_stats.get('chars', 0)} chars"
            + (f", {hydration_stats.get('dropped_windows')} dropped"
               if hydration_stats.get("dropped_windows") else ""),
            detail={"filename": filename, "section": section["title"],
                    "chunk_index": chunk_index, **hydration_stats},
        )
        if evidence_context:
            source_text_block = (
                "\n\nPRIMARY SOURCE TEXT backing the items above — use it for "
                "exact wording, numbers, dates, parties, and citations. Do not "
                "import facts not reflected in the items.\n" + evidence_context
            )

    chunk_note = (f" (part {chunk_index + 1} of {chunk_count} for this section — "
                  "write ONLY this part's content; other parts are handled "
                  "separately; do not summarize or introduce them)"
                  if chunk_count > 1 else "")

    prompt = f"""You are writing ONE SECTION of the final deliverable of a completed expert investigation. The items below are your ONLY knowledge — write from them, never invent.

ORIGINAL REQUEST:
{board.instruction}

FILE: {filename} - {file_form}
{format_rules}

SECTION: {section['title']}{chunk_note}
SECTION GUIDANCE: {section['guidance']}

{f'''COVERAGE PLAN for units in this chunk (binding structure — fill it, do not reorganize):
{coverage_lines}
''' if coverage_lines else ''}
ITEMS (binding requirements, resolved questions with their claims, and coverage units — a "requirement" item is a constraint on the work product and must be satisfied wherever it applies):
{payloads if payloads.strip() else '(no packet-supported content for this section)'}
{source_text_block}

NUMERICAL FIDELITY: every specific number, amount, percentage, date, count, dollar figure, ratio, or calculation in the items MUST appear verbatim in the output.

Write ONLY this section's content (no document title, no other sections, no meta-commentary). Professional, specific, decision-ready."""

    # Register the manifest BEFORE any model call — a thrown call still
    # persists the exact payload it received.
    manifest = {
        "chunk_index": chunk_index, "chunk_count": chunk_count,
        "items": len(chunk), "claim_ids": claim_ids,
        "requirement_ids": requirement_ids,
        "context_claim_ids": context_claim_ids, "unit_ids": unit_ids,
        "serialized_chars": len(payloads),  # exact joined payload length
        "hydration_stats": hydration_stats,
        "hydration_chars": hydration_stats.get("chars", 0),
        "hydration_sha256": (
            hashlib.sha256(source_text_block.encode("utf-8")).hexdigest()
            if source_text_block else None
        ),
        "serialized_payload": payloads,
        "result": "raised",
        "repair": "off",
    }
    if sec_chunks is not None:
        sec_chunks.append(manifest)

    tin0, tout0 = board.tokens_input, board.tokens_output
    try:
        text = call_text(caller, board, prompt, kind="synthesize",
                         max_tokens=32768, temperature=0.25)
    except Exception as exc:
        # A thrown section call is an explicit assembly failure — stamp the
        # accounting, log the exact context, then keep fail-loud behavior.
        manifest["tokens_in"] = board.tokens_input - tin0
        manifest["tokens_out"] = board.tokens_output - tout0
        board.log(
            "assembly_failure",
            f"{filename} / {section['title']}: chunk {chunk_index + 1}/"
            f"{chunk_count} draft call raised: {str(exc)[:160]}",
            detail={"filename": filename, "section": section["title"],
                    "chunk_index": chunk_index, "claim_ids": claim_ids,
                    "requirement_ids": requirement_ids,
                    "error": str(exc)[:300]},
        )
        raise
    manifest["result"] = "ok" if text.strip() else "empty"
    draft_in = board.tokens_input - tin0
    draft_out = board.tokens_output - tout0
    if ledger is not None:
        ledger.add_draft_tokens(draft_in, draft_out)

    is_xlsx = filename.lower().endswith(".xlsx")

    if text and _VERIFICATION_ENABLED and ledger is not None:
        verified, v_record = _verify_chunk(
            repairer, board, ledger,
            draft=text, payloads=payloads, claim_ids=claim_ids,
            section_title=section["title"], filename=filename,
            is_xlsx=is_xlsx,
        )
        manifest["verify"] = v_record.get("status", "unknown")
        manifest["verify_record"] = {
            k: v for k, v in v_record.items() if k != "pre_barrier_draft"
        }
        text = verified
    elif text and _REPAIR_ENABLED:
        try:
            repaired = _repair_section(
                repairer, board, filename=filename, format_rules=format_rules,
                section=section, payloads=payloads,
                coverage_lines=coverage_lines,
                source_text_block=source_text_block, draft=text,
            )
        except Exception:
            manifest["repair"] = "raised"
            manifest["tokens_in"] = board.tokens_input - tin0
            manifest["tokens_out"] = board.tokens_output - tout0
            raise
        if _usable_repair(text, repaired):
            text = repaired
            manifest["repair"] = "kept"
        else:
            manifest["repair"] = "discarded"

    manifest["tokens_in"] = board.tokens_input - tin0
    manifest["tokens_out"] = board.tokens_output - tout0
    return text, manifest


def _repair_section(caller, board: Board, *, filename: str, format_rules: str,
                    section: dict, payloads: str, coverage_lines: str,
                    source_text_block: str, draft: str) -> str:
    """Scoped coverage editor: repair one section chunk against exactly its
    own items. Never sees or rewrites other sections."""
    prompt = f"""You are the coverage editor for ONE SECTION of an expert work product. Compare the draft against its items and rewrite the COMPLETE section so every item-supported material fact survives.

FILE: {filename}
{format_rules}

SECTION: {section['title']}
SECTION GUIDANCE: {section['guidance']}

EDITORIAL RULES:
- Preserve correct draft content, names, dates, amounts, citations, structure.
- Do not invent outside the items. Exact numbers, dates, thresholds, parties, defined terms, and section names are high-risk facts — include them verbatim when material.
- If the items contain a material fact the draft missed, add it in place; never append a generic note.
{f'''- COVERAGE: {coverage_lines}''' if coverage_lines else ''}

ITEMS (the exact population the draft was written from — atomic, never truncated):
{payloads}
{source_text_block}

DRAFT SECTION TO REPAIR:
{draft[:_REPAIR_DRAFT_CAP]}

Return only the complete revised section. No commentary."""
    return call_text(caller, board, prompt, kind="synthesis_repair",
                     max_tokens=32768, temperature=0.15)


# ---------------------------------------------------------------------------
# Verification Barrier V2
# ---------------------------------------------------------------------------

_AUDIT_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "defect": {"type": "string", "enum": [
                        "factual_error", "computation_error",
                        "wrong_entity", "unsupported_claim",
                        "contradiction", "omission",
                    ]},
                    "severity": {"type": "string", "enum": [
                        "blocking", "advisory",
                    ]},
                    "span": {"type": "string"},
                    "explanation": {"type": "string"},
                },
                "required": ["id", "defect", "severity", "span", "explanation"],
            },
        },
    },
    "required": ["findings"],
}

_CORRECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "finding_id": {"type": "string"},
                    "operation": {"type": "string", "enum": [
                        "replace", "insert_after", "delete",
                    ]},
                    "span": {"type": "string"},
                    "replacement": {"type": "string"},
                },
                "required": ["finding_id", "operation", "span"],
            },
        },
    },
    "required": ["edits"],
}


def _validate_audit(audit: dict, claim_ids: list[str]) -> bool:
    """Return True if the audit response is structurally valid."""
    if not isinstance(audit, dict):
        return False
    findings = audit.get("findings")
    if not isinstance(findings, list):
        return False
    if len(findings) > _AUDIT_MAX_FINDINGS:
        return False
    seen_ids: set[str] = set()
    claim_id_set = set(claim_ids)
    for f in findings:
        if not isinstance(f, dict):
            return False
        fid = f.get("id", "")
        if not fid or fid in seen_ids:
            return False
        seen_ids.add(fid)
        if f.get("defect") not in (
            "factual_error", "computation_error", "wrong_entity",
            "unsupported_claim", "contradiction", "omission",
        ):
            return False
        if f.get("severity") not in ("blocking", "advisory"):
            return False
        span = f.get("span", "")
        if f.get("defect") != "omission" and not span:
            return False
    return True


def _blocking_findings(audit: dict) -> list[dict]:
    return [f for f in audit.get("findings", []) if f.get("severity") == "blocking"]


def _validate_correction(correction: dict, blocking_ids: set[str]) -> bool:
    """Return True if the correction response is structurally valid."""
    if not isinstance(correction, dict):
        return False
    edits = correction.get("edits")
    if not isinstance(edits, list):
        return False
    for e in edits:
        if not isinstance(e, dict):
            return False
        if e.get("operation") not in ("replace", "insert_after", "delete"):
            return False
        if e.get("finding_id") not in blocking_ids:
            return False
        if e.get("operation") != "delete" and not e.get("replacement"):
            return False
    return True


def _apply_edits(draft: str, edits: list[dict]) -> str | None:
    """Apply correction edits to the draft. Returns None if any edit fails."""
    result = draft
    applied: set[str] = set()
    for edit in edits:
        fid = edit.get("finding_id", "")
        if fid in applied:
            continue
        op = edit.get("operation", "")
        span = edit.get("span", "")
        replacement = edit.get("replacement", "")

        if op == "replace" and span:
            if span not in result:
                return None
            result = result.replace(span, replacement, 1)
        elif op == "insert_after" and span:
            idx = result.find(span)
            if idx < 0:
                return None
            insert_at = idx + len(span)
            result = result[:insert_at] + replacement + result[insert_at:]
        elif op == "delete" and span:
            if span not in result:
                return None
            result = result.replace(span, "", 1)
        else:
            return None
        applied.add(fid)
    return result


def _withhold_text(section_title: str, is_xlsx: bool) -> str:
    """Format-appropriate placeholder for a withheld chunk."""
    if is_xlsx:
        return (f"## Sheet: {section_title}\n"
                "| Verification limitation |\n| --- |\n"
                f"| This section's content could not be verified against "
                f"source materials and has been withheld for accuracy. |")
    return (f"[Verification limitation for \"{section_title}\": this section's "
            "content could not be verified against source materials and has "
            "been withheld for accuracy.]")


def _verify_chunk(
    caller, board: Board, ledger: VerificationLedger,
    *, draft: str, payloads: str, claim_ids: list[str],
    section_title: str, filename: str, is_xlsx: bool,
) -> tuple[str, dict]:
    """Run the verification barrier on one chunk.

    Returns (final_text, verification_record).
    final_text is either the original draft (clean), corrected text, or
    a withhold placeholder.
    """
    record: dict = {"status": "skipped", "pre_barrier_draft": draft}

    if not ledger.can_reserve():
        record["status"] = "budget_exhausted"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    ledger.chunks_audited += 1

    audit_prompt = f"""You are a factual auditor. Compare the DRAFT against its ITEMS and report defects.

ITEMS (the sole source of truth — everything the draft should contain):
{payloads[:_REPAIR_DRAFT_CAP]}

DRAFT TO AUDIT:
{draft[:_REPAIR_DRAFT_CAP]}

Report up to {_AUDIT_MAX_FINDINGS} findings. Each finding must have:
- id: unique string (f1, f2, ...)
- defect: one of factual_error, computation_error, wrong_entity, unsupported_claim, contradiction, omission
- severity: "blocking" (would mislead the reader) or "advisory" (style/minor)
- span: the exact text from the draft containing the defect (empty string for omissions)
- explanation: why this is wrong, citing the specific item that contradicts it

If the draft is faithful to its items, return {{"findings": []}}."""

    tin0, tout0 = board.tokens_input, board.tokens_output
    try:
        audit = call_json(caller, board, audit_prompt, kind="verify_audit",
                          max_tokens=4096, temperature=0.0)
    except Exception:
        record["status"] = "audit_raised"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    audit_in = board.tokens_input - tin0
    audit_out = board.tokens_output - tout0
    ledger.charge(audit_in, audit_out)
    record["audit_tokens"] = {"input": audit_in, "output": audit_out}

    if not _validate_audit(audit, claim_ids):
        record["status"] = "audit_invalid"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    record["audit_findings"] = len(audit.get("findings", []))
    blockers = _blocking_findings(audit)
    record["audit_blockers"] = len(blockers)

    if not blockers:
        record["status"] = "audit_clean"
        ledger.chunks_clean += 1
        return draft, record

    if not ledger.can_reserve():
        record["status"] = "budget_exhausted"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    blocking_ids = {f["id"] for f in blockers}
    blocker_text = "\n".join(
        f"- {f['id']}: [{f['defect']}] span=\"{f['span'][:100]}\" — {f['explanation'][:200]}"
        for f in blockers
    )

    correction_prompt = f"""You are a precision editor. Fix ONLY the blocking defects listed below by producing targeted edit operations on the draft.

BLOCKING DEFECTS:
{blocker_text}

ITEMS (source of truth):
{payloads[:_REPAIR_DRAFT_CAP]}

DRAFT:
{draft[:_REPAIR_DRAFT_CAP]}

For each blocking defect, produce one edit:
- finding_id: the defect id
- operation: "replace" (swap span with replacement), "insert_after" (add after span), or "delete" (remove span)
- span: exact text from the draft to target
- replacement: the corrected text (omit for delete)

Return {{"edits": [...]}}. Fix ONLY the blocking defects — do not rewrite other content."""

    tin0, tout0 = board.tokens_input, board.tokens_output
    try:
        correction = call_json(caller, board, correction_prompt,
                               kind="verify_correct", max_tokens=4096,
                               temperature=0.0)
    except Exception:
        record["status"] = "correct_raised"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    corr_in = board.tokens_input - tin0
    corr_out = board.tokens_output - tout0
    ledger.charge(corr_in, corr_out)
    record["correct_tokens"] = {"input": corr_in, "output": corr_out}

    if not _validate_correction(correction, blocking_ids):
        record["status"] = "correct_invalid"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    corrected = _apply_edits(draft, correction.get("edits", []))
    if corrected is None:
        record["status"] = "edit_failed"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    if not ledger.can_reserve():
        record["status"] = "budget_exhausted"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    reaudit_prompt = f"""You are a factual auditor performing a re-audit after corrections. Compare the CORRECTED DRAFT against its ITEMS.

ITEMS (sole source of truth):
{payloads[:_REPAIR_DRAFT_CAP]}

CORRECTED DRAFT:
{corrected[:_REPAIR_DRAFT_CAP]}

Report any remaining defects using the same format. If all blocking defects are resolved, return {{"findings": []}}."""

    tin0, tout0 = board.tokens_input, board.tokens_output
    try:
        reaudit = call_json(caller, board, reaudit_prompt,
                            kind="verify_reaudit", max_tokens=4096,
                            temperature=0.0)
    except Exception:
        record["status"] = "reaudit_raised"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    reaudit_in = board.tokens_input - tin0
    reaudit_out = board.tokens_output - tout0
    ledger.charge(reaudit_in, reaudit_out)
    record["reaudit_tokens"] = {"input": reaudit_in, "output": reaudit_out}

    if not _validate_audit(reaudit, claim_ids):
        record["status"] = "reaudit_invalid"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    reaudit_blockers = _blocking_findings(reaudit)
    record["reaudit_blockers"] = len(reaudit_blockers)

    if reaudit_blockers:
        record["status"] = "reaudit_blockers"
        ledger.chunks_withheld += 1
        return _withhold_text(section_title, is_xlsx), record

    record["status"] = "corrected"
    ledger.chunks_corrected += 1
    return corrected, record


def _assemble_sections(filename: str, section_outputs: list[tuple[str, list[str]]],
                       residual_note: str) -> str:
    """Deterministic assembly: code supplies headings and concatenates in plan
    order. No model call rewrites, summarizes, or drops sections."""
    is_xlsx = filename.lower().endswith(".xlsx")
    parts: list[str] = []
    if is_xlsx:
        # Sheets are the structure; injected headings would collide.
        for _title, texts in section_outputs:
            parts.extend(t for t in texts if t.strip())
        if residual_note:
            rows = "\n".join(
                f"| {line.lstrip('- ').strip()} |"
                for line in residual_note.splitlines() if line.strip()
            )
            parts.append("## Sheet: Limitations\n| Unresolved material "
                         "question |\n| --- |\n" + rows)
    else:
        for title, texts in section_outputs:
            body = "\n\n".join(t for t in texts if t.strip())
            if not body:
                body = "(no packet-supported content for this section)"
            if title:
                parts.append(f"## {title}\n\n{body}")
            else:
                parts.append(body)
        if residual_note:
            parts.append("## Limitations\n\nThe following material questions "
                         "were not fully resolved during the investigation:\n"
                         + residual_note)
    return "\n\n".join(parts)


def synthesize(smart_caller, board: Board, plan: dict,
               repair_caller=None) -> dict[str, str]:
    """Generate each deliverable section-locally: atomic section-scoped packet
    chunks, section-scoped drafting/repair, deterministic assembly.

    smart_caller is used for the drafting calls (the premium model).
    repair_caller, if provided, handles scoped repair passes (cheaper model).
    """
    repairer = repair_caller or smart_caller
    results: dict[str, str] = {}
    ledger: VerificationLedger | None = None
    if _VERIFICATION_ENABLED:
        ledger = VerificationLedger()
    deliverables = board.metadata.get("deliverables", {})
    required = list(deliverables.values()) if deliverables else ["output.docx"]

    # Guard: every required file must have a plan entry with its EXACT name.
    planned_names = {str(f.get("filename", "")) for f in plan.get("files", [])}
    closed_ids = [t.id for t in board.targets if t.status == "closed"]
    for name in required:
        if name in planned_names:
            continue
        # Try fuzzy match (planner drifted on the name) — rename in place.
        fuzzy = next(
            (f for f in plan.get("files", [])
             if str(f.get("filename", "")) not in required
             and Path(str(f.get("filename", ""))).suffix == Path(name).suffix),
            None,
        )
        if fuzzy is not None:
            board.log("synthesis_plan", f"renamed plan file {fuzzy.get('filename')} -> {name}")
            fuzzy["filename"] = name
            planned_names.add(name)
        else:
            plan.setdefault("files", []).append({
                "filename": name, "form": "document",
                "sections": [{"title": "Analysis", "target_ids": closed_ids,
                              "guidance": "complete answer"}],
            })
            board.log("synthesis_plan", f"added missing plan entry for {name}")
    # Drop hallucinated extras not in the required set.
    plan["files"] = [
        f for f in plan.get("files", [])
        if str(f.get("filename", "")) in required
    ]
    residuals = [
        t for t in board.targets
        if t.rank >= 2 and t.status in ("open", "blocked", "waived")
        and not t.reason.startswith("merged into")
    ]
    residual_note = "\n".join(
        f"- [{t.status}] {t.need}" + (f" — {t.reason}" if t.reason else "")
        for t in residuals
    )

    for file_plan in plan.get("files", []):
        filename = str(file_plan.get("filename", "output.docx"))
        is_xlsx = filename.lower().endswith(".xlsx")

        format_rules = (
            "FORMAT: Spreadsheet content. Use '## Sheet: <name>' to start each "
            "sheet, then markdown pipe tables (| col | col |). Every row of "
            "data the analysis supports — spreadsheets are for completeness, "
            "not summaries. No prose paragraphs."
            if is_xlsx else
            "FORMAT: Markdown that converts to a professional document. "
            "'##'/'###' for headings within this section, '-' for bullets, "
            "plain paragraphs for prose. Concrete numbers, exact names, "
            "citations to source documents inline like (Source: <doc>, <section>)."
        )
        sections = _eligible_section_items(board, file_plan)
        file_manifest: dict = {"filename": filename, "sections": []}
        section_outputs: list[tuple[str, list[str]]] = []
        total_calls = 0

        def _persist_artifacts() -> None:
            """Every exit path — success or failure — persists exactly what
            happened: the assembly manifest and the funnel-searchable packet
            record of every chunk that actually reached a model call."""
            _dump_assembly(board, filename, file_manifest)
            _dump_packets(board, filename, {
                "sections": [
                    {"section": s["title"],
                     "chunks": [c.get("serialized_payload", "")
                                for c in s.get("chunks", [])]}
                    for s in file_manifest["sections"]
                ],
            })

        try:
          for section in sections:
            section_req_ids = [c.id for it in section.get("requirements", [])
                               for c in it.get("claims", [])]
            # Register the section manifest up front and mutate it in place —
            # a mid-section failure still persists every executed chunk.
            sec_manifest: dict = {"title": section["title"], "chunks": []}
            file_manifest["sections"].append(sec_manifest)
            try:
                chunks = _chunk_section_items(
                    section["items"], _SECTION_CHUNK_CAP,
                    requirements=section.get("requirements"),
                    max_items=_MAX_ITEMS_PER_CHUNK,
                )
            except ChunkCapacityError as exc:
                board.log(
                    "assembly_failure",
                    f"{filename} / {section['title']}: {exc}",
                    detail={"filename": filename, "section": section["title"],
                            "reason": "chunk_capacity",
                            "requirement_ids": section_req_ids,
                            "error": str(exc)[:300], **exc.detail},
                )
                # The failure entry identifies the offending composition; it
                # never claims a chunk was sent. Persistence happens in the
                # finally below for every exit path.
                sec_manifest["failure"] = {
                    "reason": "chunk_capacity",
                    "requirement_ids": section_req_ids,
                    "error": str(exc)[:300], **exc.detail,
                }
                raise
            texts: list[str] = []
            for i, chunk in enumerate(chunks):
                if not chunk:
                    sec_manifest["chunks"].append(
                        {"chunk_index": 0, "chunk_count": 1, "items": 0,
                         "requirement_ids": section_req_ids,
                         "result": "empty_section"})
                    # A section with no eligible items is a structural
                    # failure surfaced explicitly, never a silent omission.
                    board.log(
                        "assembly_failure",
                        f"{filename} / {section['title']}: no eligible items",
                        detail={"filename": filename,
                                "section": section["title"],
                                "reason": "empty_section",
                                "requirement_ids": section_req_ids},
                    )
                    continue
                text, m = _synthesize_section(
                    smart_caller, repairer, board,
                    filename=filename,
                    file_form=str(file_plan.get("form", "document")),
                    format_rules=format_rules,
                    section=section, chunk=chunk,
                    chunk_index=i, chunk_count=len(chunks),
                    sec_chunks=sec_manifest["chunks"],  # registered pre-call
                    ledger=ledger,
                )
                texts.append(text)
                total_calls += 1
                if m["result"] == "empty":
                    # An empty/failed section call is an explicit assembly
                    # failure, never a silent omission.
                    board.log(
                        "assembly_failure",
                        f"{filename} / {section['title']}: chunk "
                        f"{i + 1}/{len(chunks)} produced no content",
                        detail={"filename": filename,
                                "section": section["title"],
                                "chunk_index": i,
                                "claim_ids": m["claim_ids"]},
                    )
            section_outputs.append((section["title"], texts))
        finally:
            # Success or failure, persist exactly what happened: the assembly
            # manifest and the funnel-searchable record of every chunk that
            # actually reached a model call.
            _persist_artifacts()

        if not sections:
            board.log(
                "assembly_failure",
                f"{filename}: plan produced no renderable sections",
                detail={"filename": filename, "reason": "empty_plan"},
            )
        # Failure-honesty guard: a deliverable whose every section produced no
        # synthesized body is a failed task, never a completed one. The
        # assembly evidence is already persisted by the finally above; fail
        # loudly so no `completed` status can be written. Score-neutral: this
        # path cannot trigger when any section produced content.
        if not any(t.strip() for _title, texts in section_outputs
                   for t in texts):
            board.log(
                "assembly_failure",
                f"{filename}: every section empty — refusing to represent an "
                "empty assembly as a completed deliverable",
                detail={"filename": filename, "reason": "all_sections_empty",
                        "sections": [t for t, _ in section_outputs]},
            )
            raise EmptyAssemblyError(
                f"{filename}: all {len(section_outputs)} sections produced "
                "no synthesized content")
        final = _assemble_sections(filename, section_outputs, residual_note)
        results[filename] = final or "(synthesis produced no content)"
        board.log(
            "synthesize",
            f"{filename}: {len(final)} chars from {total_calls} section calls "
            f"across {len(sections)} sections",
            detail={"filename": filename, "section_calls": total_calls,
                    "sections": [
                        {"title": s["title"],
                         "chunks": [{k: v for k, v in c.items()
                                     if k != "serialized_payload"}
                                    for c in s["chunks"]]}
                        for s in file_manifest["sections"]
                    ]},
        )

    if ledger is not None:
        board.log(
            "verification_ledger",
            f"V2 barrier: {ledger.chunks_audited} audited, "
            f"{ledger.chunks_clean} clean, {ledger.chunks_corrected} corrected, "
            f"{ledger.chunks_withheld} withheld, "
            f"{ledger.spent}/{ledger.budget} tokens ({ledger.calls} calls)",
            detail=ledger.summary(),
        )

    return results


def _dump_assembly(board: Board, filename: str, manifest: dict) -> None:
    """Persist the assembly manifest and each exact serialized chunk supplied
    to synthesis — funnel analysis reads what the model actually saw."""
    if not board.output_dir:
        return
    try:
        d = Path(board.output_dir) / "loop"
        d.mkdir(parents=True, exist_ok=True)
        safe = filename.replace("/", "_").replace("\\", "_")
        (d / f"assembly_{safe}.json").write_text(
            json.dumps(manifest, indent=1, default=str), encoding="utf-8",
        )
    except OSError:
        pass


def _usable_repair(draft: str, repaired: str | None) -> bool:
    """Reject parse failures and obvious truncation from the repair pass."""
    if not repaired:
        return False
    cleaned = repaired.strip()
    if not cleaned:
        return False
    if len(draft) < 1200:
        return len(cleaned) >= len(draft) * 0.5
    return len(cleaned) >= max(1200, int(len(draft) * 0.6))


def _dump_packets(board: Board, filename: str, packet_blocks) -> None:
    """Persist exactly what synthesis saw — the funnel analyzer needs this
    to answer 'did this claim survive packet selection?' without inference."""
    if not board.output_dir:
        return
    try:
        d = Path(board.output_dir) / "loop"
        d.mkdir(parents=True, exist_ok=True)
        safe = filename.replace("/", "_").replace("\\", "_")
        (d / f"packets_{safe}.json").write_text(
            json.dumps(packet_blocks, indent=1, default=str), encoding="utf-8",
        )
    except OSError:
        pass


def write_final_state(board: Board) -> None:
    """Stop-reason + residual ledger — the run must explain itself."""
    if not board.output_dir:
        return
    d = Path(board.output_dir) / "loop"
    d.mkdir(parents=True, exist_ok=True)
    summary = {
        "stop_reason": board.stop_reason,
        "iterations": board.iteration,
        "targets": {
            "total": len(board.targets),
            "closed": sum(1 for t in board.targets if t.status == "closed"),
            "waived": sum(1 for t in board.targets if t.status == "waived"),
            "blocked": sum(1 for t in board.targets if t.status == "blocked"),
            "open_at_stop": [
                {"id": t.id, "materiality": t.materiality, "need": t.need,
                 "blockers": board.target_blockers(t)}
                for t in board.open_targets()
            ],
        },
        "claims": {
            "total": len(board.claims),
            "derived": sum(1 for c in board.claims if c.is_derived),
            "unbound": len(board.unbound_claims()),
        },
        "contract": {
            "obligations": len(board.obligations),
            "satisfied": sum(1 for o in board.obligations if o.status == "satisfied"),
            "waived": sum(1 for o in board.obligations if o.status == "waived"),
            "open_mandatory_at_stop": [
                {"id": o.id, "text": o.text, "coverage": o.coverage}
                for o in board.open_mandatory_obligations()
            ],
            "units": len(board.units),
            "units_evidenced": sum(
                1 for u in board.units if u.status in ("evidenced", "analyzed")
            ),
        },
        "tokens": board.total_tokens_used,
        "cost_by_model": board.cost_by_model,
    }
    (d / "final_state.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )
