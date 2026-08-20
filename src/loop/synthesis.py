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
import time
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

_SYNTHESIS_HYDRATE = os.getenv("LOOP_SYNTHESIS_HYDRATE", "0").strip().lower() in (
    "1", "true", "yes",
)
_SYNTHESIS_HYDRATE_MAX = int(os.getenv("LOOP_SYNTHESIS_HYDRATE_MAX_CHARS", "400000"))

_VERIFICATION_SHADOW = os.getenv(
    "LOOP_SYNTHESIS_VERIFICATION_SHADOW", "0"
).strip().lower() in ("1", "true", "yes")


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
                        verification_ledger: "VerificationLedger | None" = None,
                        ) -> tuple[str, dict]:
    """Draft (and, when enabled, repair) exactly one section chunk against
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

    if text and _REPAIR_ENABLED:
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

    if _VERIFICATION_SHADOW and verification_ledger is not None and text.strip():
        try:
            shadow = _shadow_verify_chunk(
                caller, board, draft=text, chunk=chunk,
                filename=filename, section_title=section["title"],
                chunk_index=chunk_index, ledger=verification_ledger,
            )
        except Exception:
            shadow = {"status": "failed",
                      "reason": "exception_containment",
                      "control_hash": hashlib.sha256(
                          text.encode("utf-8")).hexdigest(),
                      "control_fallback": True,
                      "activation_eligible": False,
                      "filename": filename,
                      "section_title": section["title"],
                      "chunk_index": chunk_index,
                      "scope_ids": []}
            verification_ledger.record(shadow)
        manifest["verification_shadow"] = shadow

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
    verification_ledger = VerificationLedger() if _VERIFICATION_SHADOW else None
    results: dict[str, str] = {}
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
                    sec_chunks=sec_manifest["chunks"],
                    verification_ledger=verification_ledger,
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

    if verification_ledger is not None:
        _dump_verification_shadow(board, verification_ledger)
        board.log("verification_shadow",
                  f"V3 shadow complete: {verification_ledger.summary()}",
                  detail=verification_ledger.summary())

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


# --- Shadow Verification V3 ---

_BLOCKING_DEFECTS = frozenset({
    "factual_error", "numerical_error", "unsupported_claim",
    "contradicts_source", "fabricated_detail",
})
_ADVISORY_DEFECTS = frozenset({
    "missing_nuance", "imprecise_language", "style",
})
_KNOWN_DEFECTS = _BLOCKING_DEFECTS | _ADVISORY_DEFECTS
_VERIFICATION_AUDIT_MAX_TOKENS = 8192


class VerificationUnavailableError(RuntimeError):
    pass


class VerificationBlockedError(RuntimeError):
    pass


class VerificationLedger:
    def __init__(self):
        self.chunks_verified = 0
        self.chunks_skipped = 0
        self.chunks_clean = 0
        self.chunks_corrected = 0
        self.chunks_failed = 0
        self.invalid_audits = 0
        self.invalid_re_audits = 0
        self.edits_applied = 0
        self.errors_caught = 0
        self.activation_eligible = 0
        self.entries: list[dict] = []

    def record(self, entry: dict) -> None:
        status = entry.get("status", "failed")
        if status == "skipped":
            self.chunks_skipped += 1
        else:
            self.chunks_verified += 1
            if status == "clean":
                self.chunks_clean += 1
            elif status == "corrected":
                self.chunks_corrected += 1
            else:
                self.chunks_failed += 1
                reason = entry.get("reason", "")
                if "invalid_audit" in reason:
                    self.invalid_audits += 1
                if "invalid_re_audit" in reason:
                    self.invalid_re_audits += 1
        self.edits_applied += entry.get("edits_applied", 0)
        self.errors_caught += entry.get("errors_caught", 0)
        if entry.get("activation_eligible"):
            self.activation_eligible += 1
        self.entries.append(entry)

    def summary(self) -> dict:
        return {
            "chunks_verified": self.chunks_verified,
            "chunks_skipped": self.chunks_skipped,
            "chunks_clean": self.chunks_clean,
            "chunks_corrected": self.chunks_corrected,
            "chunks_failed": self.chunks_failed,
            "invalid_audits": self.invalid_audits,
            "invalid_re_audits": self.invalid_re_audits,
            "edits_applied": self.edits_applied,
            "errors_caught": self.errors_caught,
            "activation_eligible": self.activation_eligible,
        }


def _semantic_verification_scopes(chunk: list[dict]) -> dict[str, list[str]]:
    scopes: dict[str, list[str]] = {}
    for it in chunk:
        tp = it.get("type", "")
        if tp == "claim":
            tid = it["payload"].get("target", {}).get("target_id", "")
            if tid:
                scopes.setdefault(f"target:{tid}", []).extend(
                    c.id for c in it.get("claims", []))
        elif tp == "unit":
            for uid in it.get("unit_ids", []):
                ids = [c.id for c in it.get("claims", [])]
                ids += [c.id for c in it.get("context_claims", [])]
                scopes.setdefault(f"unit:{uid}", []).extend(ids)
        elif tp == "requirement":
            for c in it.get("claims", []):
                scopes.setdefault(f"requirement:{c.id}", [c.id])
    return scopes


def _build_audit_prompt(draft: str, scopes: dict[str, list[str]],
                        items_text: str) -> str:
    scope_lines = json.dumps(scopes, indent=1)
    return f"""You are a factual accuracy auditor. Examine the draft against its source items. Report only defects provable from the items.

VERIFICATION SCOPES (semantic units of the analysis):
{scope_lines}

SOURCE ITEMS (ground truth):
{items_text}

DRAFT TO AUDIT:
{draft}

Return JSON. If no defects: {{"findings": []}}

Otherwise:
{{"findings": [{{
  "finding_id": "f1",
  "defect_type": "<factual_error|numerical_error|unsupported_claim|contradicts_source|fabricated_detail|missing_nuance|imprecise_language>",
  "scope_id": "<scope from above>",
  "impact": "<blocking|advisory>",
  "description": "<what is wrong>",
  "operation": "<replace|delete|insert_after>",
  "match": "<exact substring of draft>",
  "replacement": "<corrected text>"
}}]}}

Rules: "match" must be exact substring of draft. scope_id must be from the scopes above. For delete, omit replacement. Only blocking types (factual_error, numerical_error, unsupported_claim, contradicts_source, fabricated_detail) trigger correction."""


def _count_overlapping(haystack: str, needle: str) -> int:
    count = start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx == -1:
            return count
        count += 1
        start = idx + 1


def _validate_verification_audit(raw, valid_scopes: set[str],
                                 draft: str
                                 ) -> tuple[list[dict] | None, str | None]:
    if not isinstance(raw, dict):
        return None, "not_dict"
    findings = raw.get("findings")
    if not isinstance(findings, list):
        return None, "findings_not_list"
    seen_fids: set[str] = set()
    for f in findings:
        if not isinstance(f, dict):
            return None, "finding_not_dict"
        fid = f.get("finding_id")
        if not isinstance(fid, str) or not fid:
            return None, "missing_finding_id"
        if fid in seen_fids:
            return None, f"duplicate_finding_id:{fid}"
        seen_fids.add(fid)
        dt = f.get("defect_type")
        if not isinstance(dt, str) or dt not in _KNOWN_DEFECTS:
            return None, f"unknown_defect:{dt}"
        sid = f.get("scope_id")
        if not isinstance(sid, str) or sid not in valid_scopes:
            return None, f"invalid_scope:{sid}"
        op = f.get("operation")
        if not isinstance(op, str) or op not in ("replace", "delete",
                                                  "insert_after"):
            return None, f"invalid_op:{op}"
        desc = f.get("description")
        if desc is not None and not isinstance(desc, str):
            return None, f"invalid_description:{fid}"
        impact = f.get("impact")
        if impact is not None and not isinstance(impact, str):
            return None, f"invalid_impact:{fid}"
        match = f.get("match")
        if not isinstance(match, str) or not match:
            return None, f"empty_match:{fid}"
        if match not in draft:
            return None, f"match_not_found:{fid}:{match[:60]}"
        if _count_overlapping(draft, match) > 1:
            return None, f"ambiguous_match:{fid}:{match[:60]}"
        if op in ("replace", "insert_after"):
            repl = f.get("replacement")
            if not isinstance(repl, str) or not repl:
                return None, f"empty_replacement:{fid}"
        elif op == "delete":
            repl = f.get("replacement")
            if repl is not None and not isinstance(repl, str):
                return None, f"invalid_delete_replacement:{fid}"
    return findings, None


def _apply_verification_edits(draft: str, findings: list[dict]
                              ) -> tuple[str | None, str | None]:
    intervals: list[tuple[int, int, str, str]] = []
    for f in findings:
        match_text = f["match"]
        op = f["operation"]
        replacement = f.get("replacement", "")
        idx = draft.find(match_text)
        if idx == -1:
            return None, f"match_lost:{f['finding_id']}"
        if op == "replace":
            intervals.append((idx, idx + len(match_text), replacement,
                              f["finding_id"]))
        elif op == "delete":
            intervals.append((idx, idx + len(match_text), "",
                              f["finding_id"]))
        elif op == "insert_after":
            end = idx + len(match_text)
            intervals.append((end, end, replacement, f["finding_id"]))
    intervals.sort(key=lambda x: x[0])
    for i in range(1, len(intervals)):
        prev_start, prev_end, _, prev_fid = intervals[i - 1]
        cur_start, cur_end, _, cur_fid = intervals[i]
        if cur_start < prev_end:
            return None, f"overlap:{prev_fid}/{cur_fid}"
        if cur_start == prev_end and prev_start == prev_end and cur_start == cur_end:
            return None, f"colocated_inserts:{prev_fid}/{cur_fid}"
    parts: list[str] = []
    cursor = 0
    for start, end, repl, _ in intervals:
        parts.append(draft[cursor:start])
        parts.append(repl)
        cursor = end
    parts.append(draft[cursor:])
    result = "".join(parts)
    if not result.strip():
        return None, "empty_result"
    return result, None


def _shadow_verify_chunk(caller, board: Board, *, draft: str,
                         chunk: list[dict], filename: str,
                         section_title: str, chunk_index: int,
                         ledger: "VerificationLedger") -> dict:
    t0 = time.monotonic()
    control_hash = hashlib.sha256(draft.encode("utf-8")).hexdigest()
    scopes = _semantic_verification_scopes(chunk)
    if not scopes or not draft.strip():
        entry = {"status": "skipped",
                 "reason": "no_scopes" if not scopes else "empty_draft",
                 "control_hash": control_hash,
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()) if scopes else []}
        ledger.record(entry)
        return entry

    items_text = _CHUNK_SEP.join(it["serialized"] for it in chunk)
    prompt = _build_audit_prompt(draft, scopes, items_text)
    scope_set = set(scopes.keys())

    tin0, tout0 = board.tokens_input, board.tokens_output
    try:
        audit_raw = call_json(caller, board, prompt,
                              kind="verification_audit",
                              max_tokens=_VERIFICATION_AUDIT_MAX_TOKENS,
                              temperature=0.1)
    except Exception as exc:
        entry = {"status": "failed",
                 "reason": f"audit_error:{str(exc)[:100]}",
                 "control_hash": control_hash,
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    findings, err = _validate_verification_audit(audit_raw, scope_set, draft)
    if err:
        entry = {"status": "failed",
                 "reason": f"invalid_audit:{err}",
                 "control_hash": control_hash,
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    blocking = [f for f in findings
                if f.get("defect_type") in _BLOCKING_DEFECTS
                and not f.get("scope_id", "").startswith("requirement:")]
    if not blocking:
        entry = {"status": "clean", "control_hash": control_hash,
                 "advisory_count": len(findings),
                 "activation_eligible": True,
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    candidate, edit_err = _apply_verification_edits(draft, blocking)
    if edit_err:
        entry = {"status": "failed",
                 "reason": f"edit_failure:{edit_err}",
                 "control_hash": control_hash,
                 "control_fallback": True,
                 "blocking_count": len(blocking),
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()

    re_prompt = _build_audit_prompt(candidate, scopes, items_text)
    try:
        re_raw = call_json(caller, board, re_prompt,
                           kind="verification_re_audit",
                           max_tokens=_VERIFICATION_AUDIT_MAX_TOKENS,
                           temperature=0.1)
    except Exception as exc:
        entry = {"status": "failed",
                 "reason": f"re_audit_error:{str(exc)[:100]}",
                 "control_hash": control_hash,
                 "control_fallback": True,
                 "candidate_hash": candidate_hash,
                 "edits_applied": len(blocking),
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    re_findings, re_err = _validate_verification_audit(
        re_raw, scope_set, candidate)
    if re_err:
        entry = {"status": "failed",
                 "reason": f"invalid_re_audit:{re_err}",
                 "control_hash": control_hash,
                 "control_fallback": True,
                 "candidate_hash": candidate_hash,
                 "edits_applied": len(blocking),
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    re_blocking = [f for f in re_findings
                   if f.get("defect_type") in _BLOCKING_DEFECTS
                   and not f.get("scope_id", "").startswith("requirement:")]
    if re_blocking:
        entry = {"status": "failed",
                 "reason": "residual_blockers",
                 "control_hash": control_hash,
                 "control_fallback": True,
                 "candidate_hash": candidate_hash,
                 "candidate_text": candidate,
                 "edits_applied": len(blocking),
                 "residual_count": len(re_blocking),
                 "activation_eligible": False,
                 "filename": filename, "section_title": section_title,
                 "chunk_index": chunk_index,
                 "scope_ids": list(scopes.keys()),
                 "tokens_in": board.tokens_input - tin0,
                 "tokens_out": board.tokens_output - tout0,
                 "elapsed_s": round(time.monotonic() - t0, 2)}
        ledger.record(entry)
        return entry

    entry = {"status": "corrected", "control_hash": control_hash,
             "candidate_hash": candidate_hash,
             "candidate_text": candidate,
             "edits_applied": len(blocking),
             "errors_caught": len(blocking),
             "activation_eligible": True,
             "filename": filename, "section_title": section_title,
             "chunk_index": chunk_index,
             "scope_ids": list(scopes.keys()),
             "tokens_in": board.tokens_input - tin0,
             "tokens_out": board.tokens_output - tout0,
             "elapsed_s": round(time.monotonic() - t0, 2)}
    ledger.record(entry)
    return entry


def _dump_verification_shadow(board: Board,
                              ledger: "VerificationLedger") -> None:
    if not board.output_dir:
        return
    try:
        d = Path(board.output_dir) / "loop"
        d.mkdir(parents=True, exist_ok=True)
        data = {"summary": ledger.summary(), "entries": ledger.entries}
        (d / "verification_shadow.json").write_text(
            json.dumps(data, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass


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
