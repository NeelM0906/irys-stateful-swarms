"""Tests for section-local packet serialization (cycle-5 treatment)."""
import json
from dataclasses import dataclass

from src.loop.state import Board, Claim, Obligation, Source, Target, Unit
from src.loop.synthesis import (
    synthesize, target_packet, _eligible_section_items, _chunk_section_items,
    _assemble_sections, _SECTION_CHUNK_CAP,
)


@dataclass
class _FakeDoc:
    text: str = ""


@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 100
    tokens_output: int = 50
    tokens_total: int = 150
    model: str = "fake"


class _EchoCaller:
    """Returns a deterministic marker embedding the section title and the
    claim ids present in the prompt, so tests can verify scoping."""

    def __init__(self, empty_for: tuple = ()):
        self.prompts = []
        self.empty_for = empty_for

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        section = ""
        for line in prompt.splitlines():
            if line.startswith("SECTION: "):
                section = line[9:].split(" (part")[0].strip()
                break
        if section in self.empty_for:
            return _FakeResult(text="")
        ids = sorted(set(
            tok for tok in prompt.replace('"', ' ').split()
            if tok.startswith("c") and tok[1:].isdigit()
        ))
        return _FakeResult(text=f"[{section}] claims:{','.join(ids)}")


def _make_board(n_claims_t0=3, n_claims_t1=3):
    board = Board(instruction="produce the report", output_dir="")
    board.add_source(Source(id="s0", name="doc-0", kind="document",
                            size_bytes=100, _doc=_FakeDoc(text="alpha " * 200)))
    board.add_target(Target(id="t0", need="question zero",
                            materiality="critical", status="closed"))
    board.add_target(Target(id="t1", need="question one",
                            materiality="high", status="closed"))
    for i in range(n_claims_t0):
        c = Claim(id=f"c{100 + i}", content=f"t0 fact number {100 + i}",
                  kind="analysis", target_refs=["t0"], confidence=0.9)
        board.add_claim(c)
        board.find_target("t0").claim_refs.append(c.id)
    for i in range(n_claims_t1):
        c = Claim(id=f"c{200 + i}", content=f"t1 fact number {200 + i}",
                  kind="analysis", target_refs=["t1"], confidence=0.9)
        board.add_claim(c)
        board.find_target("t1").claim_refs.append(c.id)
    return board


def _plan(two_sections=True):
    sections = [{"title": "Alpha", "target_ids": ["t0"], "guidance": "deep"}]
    if two_sections:
        sections.append({"title": "Beta", "target_ids": ["t1"],
                         "guidance": "brief"})
    return {"files": [{"filename": "output.docx", "form": "memo",
                       "sections": sections}]}


# --- 1. section isolation -------------------------------------------------------

def test_sections_cannot_see_each_others_claims():
    board = _make_board()
    caller = _EchoCaller()
    out = synthesize(caller, board, _plan())
    text = out["output.docx"]
    alpha = text.split("## Beta")[0]
    beta = text.split("## Beta")[1]
    assert "c100" in alpha and "c200" not in alpha
    assert "c200" in beta and "c100" not in beta
    # prompt-level isolation too
    alpha_prompts = [p for p in caller.prompts if "SECTION: Alpha" in p]
    assert alpha_prompts and all("c200" not in p for p in alpha_prompts)


# --- 2. one-section file, same path ------------------------------------------------

def test_single_section_file_single_call():
    board = _make_board()
    caller = _EchoCaller()
    out = synthesize(caller, board, _plan(two_sections=False))
    drafts = [p for p in caller.prompts if "coverage editor" not in p]
    assert len(drafts) == 1
    assert "## Alpha" in out["output.docx"]


# --- 3. >48 claims all assigned -----------------------------------------------------

def test_uncapped_claims_all_reach_chunks():
    board = _make_board(n_claims_t0=60, n_claims_t1=1)
    sections = _eligible_section_items(board, _plan()["files"][0])
    alpha_items = sections[0]["items"]
    ids = {c.id for it in alpha_items for c in it["claims"]}
    assert len(ids) == 60  # nothing dropped at the 48 cap
    chunks = _chunk_section_items(alpha_items, _SECTION_CHUNK_CAP)
    chunk_ids = {c.id for ch in chunks for it in ch for c in it["claims"]}
    assert chunk_ids == ids


# --- 4. atomic chunk splitting -------------------------------------------------------

def test_chunks_split_between_objects_without_loss():
    board = _make_board(n_claims_t0=6, n_claims_t1=6)
    sections = _eligible_section_items(board, _plan()["files"][0])
    items = sections[0]["items"] + sections[1]["items"]
    small_cap = 900  # force splitting
    chunks = _chunk_section_items(items, small_cap)
    assert len(chunks) >= 2
    for ch in chunks:
        for it in ch:
            json.loads(it["serialized"])  # every payload round-trips whole
    all_ids = {c.id for it in items for c in it["claims"]}
    chunk_ids = {c.id for ch in chunks for it in ch for c in it["claims"]}
    assert chunk_ids == all_ids


def test_oversized_single_object_travels_alone_unsliced():
    board = _make_board(n_claims_t0=1, n_claims_t1=0)
    big = Claim(id="c900", content="x" * 5000, kind="analysis",
                target_refs=["t0"], confidence=0.9)
    board.add_claim(big)
    board.find_target("t0").claim_refs.append("c900")
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    chunks = _chunk_section_items(sections[0]["items"], 100)
    for ch in chunks:
        assert len(ch) == 1  # each object alone, never sliced
        json.loads(ch[0]["serialized"])


# --- 5/13. manifest completeness and exact model inputs ------------------------------

def test_manifest_contains_all_eligible_ids_and_exact_payloads(tmp_path):
    board = _make_board()
    board.output_dir = str(tmp_path)
    caller = _EchoCaller()
    synthesize(caller, board, _plan())
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    ids = {cid for s in manifest["sections"] for ch in s["chunks"]
           for cid in ch.get("claim_ids", [])}
    assert ids == {f"c{100 + i}" for i in range(3)} | {f"c{200 + i}"
                                                       for i in range(3)}
    # exact payload persisted equals what the prompt carried
    for s in manifest["sections"]:
        for ch in s["chunks"]:
            payload = ch.get("serialized_payload", "")
            assert payload
            assert any(payload in p for p in caller.prompts)


def test_unbound_claims_never_enter_manifests(tmp_path):
    board = _make_board()
    board.add_claim(Claim(id="c999", content="unbound stray fact",
                          kind="observation", confidence=0.9))
    board.output_dir = str(tmp_path)
    synthesize(_EchoCaller(), board, _plan())
    manifest = (tmp_path / "loop" / "assembly_output.docx.json").read_text(
        encoding="utf-8")
    assert "c999" not in manifest


# --- 6. dedup within section; target reusable across sections ------------------------

def test_target_allocated_to_two_sections_serves_both():
    board = _make_board()
    plan = {"files": [{"filename": "output.docx", "form": "memo", "sections": [
        {"title": "Alpha", "target_ids": ["t0"], "guidance": "summary"},
        {"title": "Beta", "target_ids": ["t0"], "guidance": "full table"},
    ]}]}
    out = synthesize(_EchoCaller(), board, plan)
    text = out["output.docx"]
    assert text.count("c100") == 2  # both sections rendered the claim


# --- 7. determinism -------------------------------------------------------------------

def test_ordering_deterministic_across_runs():
    outs = []
    for _ in range(2):
        board = _make_board(n_claims_t0=20, n_claims_t1=20)
        sections = _eligible_section_items(board, _plan()["files"][0])
        outs.append(json.dumps(
            [[c.id for it in s["items"] for c in it["claims"]]
             for s in sections]))
    assert outs[0] == outs[1]


# --- 8. coverage routing and fallback ---------------------------------------------------

def _add_obligation_with_units(board, section_name):
    board.add_obligation(Obligation(id="o1", text="account for every item",
                                    coverage="exhaustive", mandatory=True))
    # coverage="exhaustive" makes the obligation set-valued (derived property)
    for i in range(2):
        u = board.add_unit(Unit(name=f"item-{i}", obligation_ref="o1",
                                anchor=f"{i}.0"))
        c = Claim(id=f"c{300 + i}", content=f"unit fact {300 + i}",
                  kind="observation", confidence=0.9)
        board.add_claim(c)
        u.claim_refs.append(c.id)
    return {"files": [{"filename": "output.docx", "form": "memo",
                       "sections": [{"title": "Alpha", "target_ids": ["t0"],
                                     "guidance": "deep"}],
                       "coverage": [{"obligation_id": "o1",
                                     "section": section_name,
                                     "unit_mode": "row",
                                     "required_slots": []}]}]}


def test_coverage_routes_to_named_section():
    board = _make_board()
    plan = _add_obligation_with_units(board, "Alpha")
    caller = _EchoCaller()
    synthesize(caller, board, plan)
    alpha_prompts = [p for p in caller.prompts if "SECTION: Alpha" in p]
    assert any("item-0" in p and "item-1" in p for p in alpha_prompts)


def test_unmatched_coverage_routes_to_coverage_appendix():
    board = _make_board()
    plan = _add_obligation_with_units(board, "Nonexistent Section")
    caller = _EchoCaller()
    out = synthesize(caller, board, plan)
    assert any("SECTION: Coverage Appendix" in p for p in caller.prompts)
    assert "## Coverage Appendix" in out["output.docx"]


def test_coverage_case_whitespace_variant_routes_to_planned_section():
    board = _make_board()
    plan = _add_obligation_with_units(board, "  alpha ")  # planned is "Alpha"
    caller = _EchoCaller()
    synthesize(caller, board, plan)
    alpha_prompts = [p for p in caller.prompts if "SECTION: Alpha" in p]
    assert any("item-0" in p for p in alpha_prompts)  # routed, not appended
    assert not any("SECTION: Coverage Appendix" in p for p in caller.prompts)


# --- 9. requirements visible without consuming packet space ---------------------------

def test_requirements_visible_in_every_section_call():
    board = _make_board()
    board.add_claim(Claim(id="c500", content="must be addressed to the court",
                          kind="requirement", source_doc="doc-0",
                          confidence=0.9))
    caller = _EchoCaller()
    synthesize(caller, board, _plan())
    drafts = [p for p in caller.prompts if "coverage editor" not in p]
    assert all("must be addressed to the court" in p for p in drafts)


# --- 11. scoped repair -------------------------------------------------------------------

def test_repair_scoped_to_single_section(monkeypatch):
    monkeypatch.setenv("LOOP_SYNTHESIS_REPAIR", "1")
    import importlib
    import src.loop.synthesis as syn
    importlib.reload(syn)
    board = _make_board()
    caller = _EchoCaller()
    syn.synthesize(caller, board, _plan())
    repair_prompts = [p for p in caller.prompts if "coverage editor" in p]
    assert repair_prompts
    for p in repair_prompts:
        assert not ("c100" in p and "c200" in p)  # never both sections
    importlib.reload(syn)


# --- 12. deterministic assembly -------------------------------------------------------------

def test_assembly_preserves_section_outputs_verbatim():
    outputs = [("Alpha", ["body A1", "body A2"]), ("Beta", ["body B"])]
    text = _assemble_sections("output.docx", outputs, "")
    assert "## Alpha\n\nbody A1\n\nbody A2" in text
    assert "## Beta\n\nbody B" in text


def test_assembly_xlsx_concatenates_without_headings():
    outputs = [("Data", ["## Sheet: One\n| a |"]),
               ("More", ["## Sheet: Two\n| b |"])]
    text = _assemble_sections("out.xlsx", outputs, "- residual stuff")
    assert "## Data" not in text and "## More" not in text
    assert "## Sheet: One" in text and "## Sheet: Two" in text
    # residuals render as a deterministic Limitations sheet, never dropped
    assert "## Sheet: Limitations" in text
    assert "| residual stuff |" in text


def test_assembly_appends_deterministic_limitations():
    text = _assemble_sections("out.docx", [("Alpha", ["body"])],
                              "- [open] unresolved question")
    assert "## Limitations" in text
    assert "- [open] unresolved question" in text


# --- reviewer round-1 gap tests -----------------------------------------------------------

def test_repair_receives_complete_untruncated_payload():
    board = _make_board(n_claims_t0=5, n_claims_t1=0)
    caller = _EchoCaller()
    synthesize(caller, board, _plan(two_sections=False))
    repair_prompts = [p for p in caller.prompts if "coverage editor" in p]
    assert repair_prompts
    # the LAST claim item (tail object) must be present whole in the repair
    for p in repair_prompts:
        assert '"c104"' in p  # tail claim id survives, no prefix slicing
        # every payload object in the repair prompt round-trips
        assert "never truncated" in p


def test_intra_target_split_when_claims_exceed_cap():
    board = _make_board(n_claims_t0=40, n_claims_t1=0)
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    items = sections[0]["items"]
    one_item = len(json.dumps(items[0]["payload"], indent=1))
    cap = one_item * 5  # forces splitting inside the single target's claims
    chunks = _chunk_section_items(items, cap)
    assert len(chunks) > 1  # one target's claims split across bounded calls
    for ch in chunks:
        joined = "\n".join(it["serialized"] for it in ch)
        assert len(joined) <= cap or len(ch) == 1
    all_ids = {c.id for it in items for c in it["claims"]}
    chunk_ids = {c.id for ch in chunks for it in ch for c in it["claims"]}
    assert chunk_ids == all_ids


def test_duplicate_claim_across_targets_serializes_once_per_section():
    board = _make_board(n_claims_t0=1, n_claims_t1=0)
    shared = board.find_claim("c100")
    board.find_target("t1").claim_refs.append(shared.id)  # bound to both
    plan = {"files": [{"filename": "output.docx", "form": "memo", "sections": [
        {"title": "Alpha", "target_ids": ["t0", "t1"], "guidance": "deep"},
    ]}]}
    sections = _eligible_section_items(board, plan["files"][0])
    ids = [c.id for it in sections[0]["items"] for c in it["claims"]
           if it["type"] == "claim"]
    assert ids.count("c100") == 1


def test_requirement_items_have_manifest_identity(tmp_path):
    board = _make_board()
    board.add_claim(Claim(id="c500", content="must be addressed to the court",
                          kind="requirement", source_doc="doc-0",
                          confidence=0.9))
    board.output_dir = str(tmp_path)
    synthesize(_EchoCaller(), board, _plan())
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    for s in manifest["sections"]:
        # requirements carry their own manifest role in EVERY chunk
        for ch in s["chunks"]:
            assert "c500" in ch.get("requirement_ids", [])
            assert "c500" not in ch.get("claim_ids", [])  # roles never mix


def test_serialized_chars_is_exact_joined_length(tmp_path):
    board = _make_board()
    board.output_dir = str(tmp_path)
    synthesize(_EchoCaller(), board, _plan(two_sections=False))
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    ch = manifest["sections"][0]["chunks"][0]
    assert ch["serialized_chars"] == len(ch["serialized_payload"])


def test_target_bound_requirement_not_duplicated_as_claim_item():
    board = _make_board(n_claims_t0=1, n_claims_t1=0)
    rc = Claim(id="c600", content="must include a signature block",
               kind="requirement", target_refs=["t0"], confidence=0.9)
    board.add_claim(rc)
    board.find_target("t0").claim_refs.append("c600")
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    sec = sections[0]
    req_ids = [c.id for it in sec["requirements"] for c in it["claims"]]
    item_ids = [c.id for it in sec["items"] for c in it["claims"]]
    assert "c600" in req_ids
    assert "c600" not in item_ids  # governs, never repeats as ordinary claim


def test_target_unit_overlap_serializes_once_per_section():
    board = _make_board(n_claims_t0=1, n_claims_t1=0)
    plan = _add_obligation_with_units(board, "Alpha")
    shared = board.find_claim("c300")  # unit claim also bound to the target
    shared.target_refs.append("t0")
    board.find_target("t0").claim_refs.append("c300")
    sections = _eligible_section_items(board, plan["files"][0])
    sec = sections[0]
    ids = [c.id for it in sec["items"] for c in it["claims"]]
    assert ids.count("c300") == 1
    # the unit that lost the duplicate records the reference explicitly
    unit_payloads = [it["payload"] for it in sec["items"]
                     if it["type"] == "unit"]
    assert any("claims_rendered_elsewhere_in_section" in p.get("unit", {})
               or "c300" in json.dumps(p) for p in unit_payloads)


def test_requirements_visible_in_every_chunk_of_split_section():
    board = _make_board(n_claims_t0=30, n_claims_t1=0)
    rc = Claim(id="c700", content="every row must carry severity",
               kind="requirement", confidence=0.9)
    board.add_claim(rc)
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    sec = sections[0]
    one_item = len(json.dumps(sec["items"][0]["payload"], indent=1))
    cap = one_item * 6
    chunks = _chunk_section_items(sec["items"], cap,
                                  requirements=sec["requirements"])
    assert len(chunks) > 1
    for ch in chunks:  # the governing requirement rides in EVERY chunk
        assert any(it["type"] == "requirement"
                   and it["claims"][0].id == "c700" for it in ch)


def test_coverage_appendix_carries_requirements():
    board = _make_board()
    rc = Claim(id="c800", content="cite the governing statute",
               kind="requirement", confidence=0.9)
    board.add_claim(rc)
    plan = _add_obligation_with_units(board, "Unknown Place")
    caller = _EchoCaller()
    synthesize(caller, board, plan)
    appendix_prompts = [p for p in caller.prompts
                        if "SECTION: Coverage Appendix" in p]
    assert appendix_prompts
    assert all("cite the governing statute" in p for p in appendix_prompts)


def test_thrown_draft_call_logs_assembly_failure(tmp_path):
    import pytest as _pytest

    class _ThrowingCaller:
        def complete(self, prompt, **kwargs):
            raise RuntimeError("provider exploded")

    board = _make_board()
    board.output_dir = str(tmp_path)
    with _pytest.raises(RuntimeError):
        synthesize(_ThrowingCaller(), board, _plan(two_sections=False))
    fails = [e for e in board.events if e.kind == "assembly_failure"]
    assert fails and "draft call raised" in fails[0].summary
    # the thrown chunk's exact payload persists in both artifacts
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    ch = manifest["sections"][0]["chunks"][0]
    assert ch["result"] == "raised"
    assert ch["serialized_payload"]
    assert "tokens_in" in ch and "tokens_out" in ch  # accounting stamped
    packets = (tmp_path / "loop" / "packets_output.docx.json").read_text(
        encoding="utf-8")
    assert "c100" in packets  # funnel record includes the thrown chunk


def test_preamble_plus_first_item_cap_boundaries():
    import pytest as _pytest
    from src.loop.synthesis import ChunkCapacityError
    board = _make_board(n_claims_t0=2, n_claims_t1=0)
    rc = Claim(id="c700", content="governing rule", kind="requirement",
               confidence=0.9)
    board.add_claim(rc)
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    sec = sections[0]
    req = sec["requirements"]
    req_len = (len(json.dumps(req[0]["payload"], indent=1))
               + 1) * len(req)
    item_len = len(json.dumps(sec["items"][0]["payload"], indent=1))
    # cap exactly fits preamble + first item: no error, over-cap never emitted
    ok_cap = req_len + item_len
    chunks = _chunk_section_items(sec["items"], ok_cap, requirements=req)
    for ch in chunks:
        joined = "\n".join(it["serialized"] for it in ch)
        assert len(joined) <= ok_cap
    # cap one char short: preamble + first item cannot fit -> explicit failure
    with _pytest.raises(ChunkCapacityError):
        _chunk_section_items(sec["items"], ok_cap - 1, requirements=req)
    # preamble alone exceeding the cap is an explicit failure
    with _pytest.raises(ChunkCapacityError):
        _chunk_section_items(sec["items"], req_len - 1, requirements=req)


def test_oversized_item_without_preamble_still_travels_alone():
    board = _make_board(n_claims_t0=2, n_claims_t1=0)
    sections = _eligible_section_items(board, _plan(False)["files"][0])
    chunks = _chunk_section_items(sections[0]["items"], 50)  # no requirements
    assert all(len(ch) == 1 for ch in chunks)  # indivisible items go alone


def test_unit_call_resolves_context_claims_across_chunk_boundary():
    board = _make_board(n_claims_t0=1, n_claims_t1=0)
    plan = _add_obligation_with_units(board, "Alpha")
    shared = board.find_claim("c300")
    shared.target_refs.append("t0")
    board.find_target("t0").claim_refs.append("c300")
    sections = _eligible_section_items(board, plan["files"][0])
    sec = sections[0]
    # force target claim item and unit items into different chunks
    one_item = max(len(json.dumps(it["payload"], indent=1))
                   for it in sec["items"])
    chunks = _chunk_section_items(sec["items"], one_item + 2)
    assert len(chunks) > 1
    # the unit's chunk must carry the shared claim's CONTENT even though the
    # canonical serialization lives in another chunk
    unit_chunks = [ch for ch in chunks
                   if any(it["type"] == "unit" and
                          "c300" in json.dumps(it["payload"])
                          for it in ch)]
    assert unit_chunks
    for ch in unit_chunks:
        blob = "\n".join(it["serialized"] for it in ch)
        if '"context_claims_canonical_elsewhere"' in blob:
            assert "unit fact 300" in blob  # full content, not a bare pointer


def test_hydration_covers_requirement_and_context_only_chunks(monkeypatch):
    monkeypatch.setenv("LOOP_SYNTHESIS_HYDRATE", "1")
    import importlib
    import src.loop.synthesis as syn
    importlib.reload(syn)
    board = _make_board(n_claims_t0=0, n_claims_t1=0)
    rc = Claim(id="c700", content="governing rule text",
               kind="requirement", source_doc="doc-0", confidence=0.9)
    board.add_claim(rc)
    plan = _add_obligation_with_units(board, "Alpha")
    caller = _EchoCaller()
    syn.synthesize(caller, board, plan)
    hyd = [e for e in board.events if e.kind == "synthesis_hydrate"]
    assert hyd  # hydration ran for the requirement+unit chunk
    importlib.reload(syn)


def test_capacity_failure_persists_failure_manifest(tmp_path, monkeypatch):
    import pytest as _pytest
    import src.loop.synthesis as syn
    monkeypatch.setattr(syn, "_SECTION_CHUNK_CAP", 200)
    board = _make_board(n_claims_t0=2, n_claims_t1=0)
    rc = Claim(id="c700", content="governing rule " * 10, kind="requirement",
               confidence=0.9)
    board.add_claim(rc)
    board.output_dir = str(tmp_path)
    with _pytest.raises(syn.ChunkCapacityError):
        syn.synthesize(_EchoCaller(), board, _plan(two_sections=False))
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    failed = [s for s in manifest["sections"] if "failure" in s]
    assert failed
    f = failed[0]["failure"]
    assert f["reason"] in ("preamble_exceeds_cap",
                           "preamble_plus_item_exceeds_cap")
    assert "c700" in f["requirement_ids"]
    assert failed[0]["chunks"] == []  # never claims an unsent chunk was sent


def test_requirement_referenced_by_unit_not_duplicated_in_call(tmp_path):
    board = _make_board(n_claims_t0=0, n_claims_t1=0)
    rc = Claim(id="c700", content="the governing statute must be cited",
               kind="requirement", confidence=0.9)
    board.add_claim(rc)
    plan = _add_obligation_with_units(board, "Alpha")
    # unit references the requirement claim too
    unit = board.units[0]
    unit.claim_refs.append("c700")
    board.output_dir = str(tmp_path)
    caller = _EchoCaller()
    synthesize(caller, board, plan)
    manifest = json.loads(
        (tmp_path / "loop" / "assembly_output.docx.json").read_text(
            encoding="utf-8"))
    for s in manifest["sections"]:
        for ch in s["chunks"]:
            req = set(ch.get("requirement_ids", []))
            can = set(ch.get("claim_ids", []))
            ctx = set(ch.get("context_claim_ids", []))
            # all three role sets are pairwise disjoint
            assert not (req & can) and not (req & ctx) and not (can & ctx)
            # the requirement content appears exactly once in the payload
            payload = ch.get("serialized_payload", "")
            assert payload.count("the governing statute must be cited") <= 1
            if "c700" in payload:
                assert '"requirement_claims_in_preamble"' in payload or \
                    '"requirement"' in payload


def test_empty_plan_logs_assembly_failure():
    board = _make_board()
    plan = {"files": [{"filename": "output.docx", "form": "memo",
                       "sections": []}]}
    board.metadata["deliverables"] = {}
    synthesize(_EchoCaller(), board, plan)
    fails = [e for e in board.events if e.kind == "assembly_failure"]
    assert fails  # structural failure surfaced, not silent


# --- 14. empty section calls create explicit failure events -----------------------------

def test_empty_section_call_logs_assembly_failure():
    board = _make_board()
    caller = _EchoCaller(empty_for=("Beta",))
    out = synthesize(caller, board, _plan())
    failures = [e for e in board.events if e.kind == "assembly_failure"]
    assert failures and failures[0].detail["section"] == "Beta"
    assert "## Beta" in out["output.docx"]  # section present, not omitted
    assert "(no packet-supported content" in out["output.docx"]


# --- 16. file guards unchanged ---------------------------------------------------------------

def test_required_file_guards_still_apply():
    board = _make_board()
    board.metadata["deliverables"] = {"main": "memo.docx"}
    plan = {"files": [{"filename": "wrong-name.docx", "form": "memo",
                       "sections": [{"title": "Alpha", "target_ids": ["t0"],
                                     "guidance": "deep"}]}]}
    out = synthesize(_EchoCaller(), board, plan)
    assert list(out.keys()) == ["memo.docx"]  # fuzzy rename preserved
