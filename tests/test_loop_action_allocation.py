"""Tests for the frontier read-lane action allocation in controller_decide."""
import json
from dataclasses import dataclass

from src.loop.state import Board, Source, Target
from src.loop.control import controller_decide, MAX_ACTIONS_PER_ITERATION
from src.loop.triage import triage_sources, catalog_summary


# --- fakes -------------------------------------------------------------------

@dataclass
class _FakeResult:
    text: str = ""
    tokens_input: int = 10
    tokens_output: int = 10
    tokens_total: int = 20
    model: str = "fake"


class _FakeCaller:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.prompts = []

    def complete(self, prompt, **kwargs):
        self.prompts.append(prompt)
        payload = self.payloads.pop(0) if self.payloads else {}
        return _FakeResult(text=json.dumps(payload))


def _make_board(n_docs, n_targets=2):
    board = Board(instruction="find the facts")
    for i in range(n_docs):
        board.add_source(Source(
            id=f"s{i}", name=f"doc{i}.txt", path=f"corpus/a/doc{i}.txt",
            kind="document", size_bytes=2048,
        ))
    for i in range(n_targets):
        board.add_target(Target(
            id=f"t{i}", need=f"need {i}",
            materiality="critical" if i == 0 else "medium",
        ))
    board.iteration = 1
    return board


def _controller_payload(n_reads, n_other, extra_actions=None):
    actions = [{"kind": "read", "source_id": f"s{i}", "focus": "x",
                "target_ids": ["t0"]} for i in range(n_reads)]
    actions += [{"kind": "bind"} for _ in range(n_other)]
    if extra_actions:
        actions += extra_actions
    return {"reasoning": "r", "target_updates": [], "obligation_updates": [],
            "actions": actions, "converge": False}


def _activate_frontier(board):
    """Run real frontier triage + render one page for the current iteration."""
    caller = _FakeCaller([{"candidates": [
        {"id": "s1", "target_ids": ["t0"], "priority": "definite", "reason": "r"},
        {"id": "s2", "target_ids": ["t1"], "priority": "definite", "reason": "r"},
    ]}])
    triage_sources(caller, board)


# --- 1. inactive/small-corpus keeps existing behavior ------------------------

def test_inactive_path_keeps_six_action_cap():
    board = _make_board(10)
    payload = _controller_payload(5, 5)  # 10 valid actions offered
    result = controller_decide(_FakeCaller([payload]), board, {})
    assert len(result["actions"]) == MAX_ACTIONS_PER_ITERATION
    prompt = board.events and [e for e in board.events if e.kind == "controller"]
    ctrl = prompt[-1].detail
    assert ctrl["frontier_read_lane_active"] is False


def test_inactive_prompt_carries_legacy_budget_line():
    board = _make_board(10)
    caller = _FakeCaller([_controller_payload(1, 1)])
    controller_decide(caller, board, {})
    assert f"Max {MAX_ACTIONS_PER_ITERATION} actions." in caller.prompts[0]
    assert "read actions" not in caller.prompts[0].split("converge_reason")[-1]


# --- 2-3. metadata/failed events cannot activate ------------------------------

def test_colliding_metadata_without_page_cannot_activate():
    board = _make_board(10)
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = {"t0": []}
    board.metadata["retrieval_fallback"] = []
    result = controller_decide(_FakeCaller([_controller_payload(4, 4)]), board, {})
    assert len(result["actions"]) == MAX_ACTIONS_PER_ITERATION
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["frontier_read_lane_active"] is False


def test_validation_failed_event_cannot_activate():
    board = _make_board(100)
    board.metadata["retrieval_frontier_enabled"] = True
    board.metadata["retrieval_frontier"] = {"ghost": [{"source_id": "s1"}]}
    board.metadata["retrieval_fallback"] = []
    result = controller_decide(_FakeCaller([_controller_payload(4, 4)]), board, {})
    assert len(result["actions"]) == MAX_ACTIONS_PER_ITERATION
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["frontier_read_lane_active"] is False


# --- 4. valid current-iteration page activates the two-lane contract ---------

def test_valid_frontier_page_activates_lane():
    board = _make_board(100)
    _activate_frontier(board)
    caller = _FakeCaller([_controller_payload(1, 1)])
    controller_decide(caller, board, {})
    assert "read actions" in caller.prompts[0]
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["frontier_read_lane_active"] is True


# --- 5. six reads + six non-reads accepted in response order -----------------

def test_six_plus_six_accepted_in_order():
    board = _make_board(100)
    _activate_frontier(board)
    payload = _controller_payload(6, 6)
    result = controller_decide(_FakeCaller([payload]), board, {})
    kinds = [a["kind"] for a in result["actions"]]
    assert kinds.count("read") == 6
    assert kinds.count("bind") == 6
    # response order preserved: all reads came first in the payload
    assert kinds[:6] == ["read"] * 6


# --- 6. seventh read and seventh non-read independently rejected -------------

def test_seventh_read_and_seventh_nonread_rejected():
    board = _make_board(100)
    _activate_frontier(board)
    payload = _controller_payload(7, 7)
    result = controller_decide(_FakeCaller([payload]), board, {})
    kinds = [a["kind"] for a in result["actions"]]
    assert kinds.count("read") == 6
    assert kinds.count("bind") == 6
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["reads_proposed"] == 7 and ctrl["reads_accepted"] == 6
    assert ctrl["nonread_proposed"] == 7 and ctrl["nonread_accepted"] == 6


# --- 7. invalid kinds consume no lane ----------------------------------------

def test_invalid_action_kinds_consume_no_lane():
    board = _make_board(100)
    _activate_frontier(board)
    junk = [{"kind": "explode"}, {"kind": "read_all"}, "not-a-dict"]
    payload = _controller_payload(6, 6, extra_actions=junk)
    result = controller_decide(_FakeCaller([payload]), board, {})
    kinds = [a["kind"] for a in result["actions"]]
    assert kinds.count("read") == 6 and kinds.count("bind") == 6


# --- 8. fewer reads are not padded --------------------------------------------

def test_fewer_reads_not_padded():
    board = _make_board(100)
    _activate_frontier(board)
    payload = _controller_payload(2, 3)
    result = controller_decide(_FakeCaller([payload]), board, {})
    kinds = [a["kind"] for a in result["actions"]]
    assert kinds.count("read") == 2 and kinds.count("bind") == 3


# --- 9. forced-analysis gate still runs ---------------------------------------

def test_forced_analysis_gate_still_runs():
    from src.loop.state import Claim
    board = _make_board(100)
    _activate_frontier(board)
    # Give t0 three raw observations and zero derived claims — the gate's
    # actual trigger — then have the controller waive it prematurely.
    t0 = board.find_target("t0")
    for i in range(3):
        c = Claim(content=f"fact {i}", kind="observation",
                  source_doc="doc0.txt", target_refs=["t0"])
        board.add_claim(c)
        t0.claim_refs.append(c.id)
    payload = _controller_payload(1, 1)
    payload["target_updates"] = [
        {"target_id": "t0", "status": "waived", "reason": "premature"},
    ]
    result = controller_decide(_FakeCaller([payload]), board, {})
    # Gate reopened t0 and injected the analyze action after allocation
    assert board.find_target("t0").status == "open"
    injected = [a for a in result["actions"]
                if a.get("kind") == "analyze" and a.get("target_id") == "t0"]
    assert injected


def test_stale_same_iteration_success_cannot_mask_current_failure():
    board = _make_board(100)
    _activate_frontier(board)
    # First controller call renders a valid page (lane active)
    controller_decide(_FakeCaller([_controller_payload(1, 1)]), board, {})
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["frontier_read_lane_active"] is True
    # Corrupt the frontier WITHOUT advancing the iteration; the next render
    # fails validation and the stale success must not activate the lane
    board.metadata["retrieval_frontier"] = {"ghost": [{"source_id": "s1"}]}
    result = controller_decide(_FakeCaller([_controller_payload(4, 4)]), board, {})
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["frontier_read_lane_active"] is False
    assert len(result["actions"]) == MAX_ACTIONS_PER_ITERATION


# --- 10. observability fields present, no evaluator data ----------------------

def test_observability_fields_and_no_evaluator_data():
    board = _make_board(100)
    _activate_frontier(board)
    result = controller_decide(_FakeCaller([_controller_payload(3, 2)]), board, {})
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    for key in ("frontier_read_lane_active", "reads_proposed", "reads_accepted",
                "nonread_proposed", "nonread_accepted",
                "accepted_read_source_ids", "accepted_read_target_ids"):
        assert key in ctrl
    blob = json.dumps(ctrl).lower()
    for banned in ("criteria", "rubric", "score", "match_criteria"):
        assert banned not in blob
    assert ctrl["accepted_read_source_ids"] == ["s0", "s1", "s2"]


# --- 11. legacy read counting on inactive path ---------------------------------

def test_inactive_lane_counts_reads_for_observability():
    board = _make_board(10)
    result = controller_decide(_FakeCaller([_controller_payload(2, 2)]), board, {})
    ctrl = [e for e in board.events if e.kind == "controller"][-1].detail
    assert ctrl["reads_accepted"] == 2 and ctrl["nonread_accepted"] == 2
