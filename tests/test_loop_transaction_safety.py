"""P1 transaction-safety invariant tests.

Verifies that every admitted controller action envelope executes before any
stop predicate is honored.  Stop checks (convergence, max_iterations, budget)
are post-dispatch; entry-admission gates prevent wasted controller calls.
Shadow telemetry records what the legacy pre-dispatch ordering would have done.
"""
from __future__ import annotations

import types
from dataclasses import dataclass
from unittest.mock import MagicMock, patch, call

import pytest

from src.loop.state import Board, Source, Target
from src.loop import (
    run_loop, MAX_ITERATIONS, BUDGET_STOP_PCT, DIMINISHING_ROUNDS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _FakeDoc:
    id: str = "d0"
    name: str = "doc.txt"
    size_bytes: int = 100
    text: str = "test document content"
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {"path": "corpus/test/doc.txt"}


@dataclass
class _FakeTask:
    instruction: str = "analyze the documents"
    documents: list = None
    metadata: dict = None
    output_dir: str = ""

    def __post_init__(self):
        if self.documents is None:
            self.documents = [_FakeDoc()]
        if self.metadata is None:
            self.metadata = {}


def _make_board(**kwargs) -> Board:
    board = Board(instruction="test", **kwargs)
    board.add_source(Source(id="s0", name="d0", path="test.txt", size_bytes=100))
    board.add_target(Target(id="t0", need="find facts", materiality="critical"))
    return board


def _stub_caller(response=None):
    return MagicMock(return_value=response or {})


_NOOP_PATCHES = {
    "src.loop.seed_targets": lambda *a, **kw: None,
    "src.loop.triage_sources": lambda *a, **kw: None,
    "src.loop.bulk_extract_frontier": lambda *a, **kw: {},
    "src.loop.finalize_bulk_extraction": lambda *a, **kw: None,
    "src.loop.plan_synthesis": lambda *a, **kw: {"sections": []},
    "src.loop.synthesize": lambda *a, **kw: "deliverable",
    "src.loop.write_final_state": lambda *a, **kw: None,
    "src.loop.reframe_ledger": lambda *a, **kw: None,
    "src.loop.maintain_ledger": lambda *a, **kw: None,
    "src.loop.blackboard_audit": lambda *a, **kw: None,
    "src.loop._force_analysis_gate": lambda board, forced: None,
}


def _run_with_patches(controller_decisions, execute_side_effect=None,
                      auto_bind_side_effect=None, board_setup=None,
                      budget_pcts=None):
    """Run the loop with controlled controller decisions and action execution.

    controller_decisions: list of dicts, one per iteration.
    execute_side_effect: callable(actions, board, ...) -> dict summary.
    auto_bind_side_effect: callable(board, ...) -> dict.
    board_setup: callable(board) to mutate board before loop starts.
    budget_pcts: list of floats — budget_used_pct() returns for each call.
    """
    decision_iter = iter(controller_decisions)
    call_log = {"controller_calls": 0, "execute_calls": [], "bind_calls": 0}

    def fake_controller(caller, board, last_summary, *, max_iterations=12,
                        closeout=False):
        call_log["controller_calls"] += 1
        try:
            return next(decision_iter)
        except StopIteration:
            return {"converge": True, "converge_reason": "exhausted",
                    "actions": []}

    def fake_execute(actions, board, worker_caller, smart_caller=None):
        call_log["execute_calls"].append(list(actions))
        if execute_side_effect:
            return execute_side_effect(actions, board)
        return {"claims": 0}

    def fake_auto_bind(board, caller, budget_stop_pct=85.0):
        call_log["bind_calls"] += 1
        if auto_bind_side_effect:
            return auto_bind_side_effect(board, caller)
        return {"bound": 0}

    budget_idx = [0]
    _original_budget_pct = Board.budget_used_pct

    def fake_budget_pct(self):
        if budget_pcts is not None and budget_idx[0] < len(budget_pcts):
            val = budget_pcts[budget_idx[0]]
            budget_idx[0] += 1
            return val
        return _original_budget_pct(self)

    patches = dict(_NOOP_PATCHES)
    patches["src.loop.controller_decide"] = fake_controller
    patches["src.loop.execute_actions"] = fake_execute
    patches["src.loop.auto_bind"] = fake_auto_bind

    task = _FakeTask()

    budget_ctx = (
        patch.object(Board, "budget_used_pct", fake_budget_pct)
        if budget_pcts is not None
        else _nullcontext()
    )

    with (
        patch.multiple("src.loop", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }),
        budget_ctx,
        patch.object(Board, "snapshot", lambda self, label="": None),
    ):
        if board_setup:
            old_init = Board.__post_init__

            def patched_init(self):
                old_init(self)
                board_setup(self)

            with patch.object(Board, "__post_init__", patched_init):
                result = run_loop(task, _stub_caller(), smart_caller=_stub_caller())
        else:
            result = run_loop(task, _stub_caller(), smart_caller=_stub_caller())

    return result, call_log


class _nullcontext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# ---------------------------------------------------------------------------
# Test 1: Convergence with actions — actions execute before stop
# ---------------------------------------------------------------------------

def test_convergence_with_actions_dispatches_before_stop():
    """Controller returns converge=True plus actions while post-action open
    sets are empty.  Assert execute_actions is called once before the
    convergence stop."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}

    def execute_and_resolve(actions, board):
        for t in board.targets:
            if t.is_open:
                board.resolve_target(t.id, "closed", "test closure")
        return {"claims": 1}

    decisions = [
        {"converge": True, "converge_reason": "all done",
         "actions": [action]},
    ]

    (deliverable, board), log = _run_with_patches(
        decisions, execute_side_effect=execute_and_resolve,
    )

    assert len(log["execute_calls"]) == 1
    assert log["execute_calls"][0] == [action]
    assert board.stop_reason.startswith("converged:")


# ---------------------------------------------------------------------------
# Test 2: Max boundary with actions — envelope executes, then max stop
# ---------------------------------------------------------------------------

def test_max_boundary_dispatches_before_stop():
    """Set MAX_ITERATIONS=1; controller returns actions.  Assert envelope
    executes once, then max stop.  No second controller call."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}
    decisions = [
        {"converge": False, "converge_reason": "", "actions": [action]},
        {"converge": True, "converge_reason": "fallback", "actions": []},
    ]

    with patch("src.loop.MAX_ITERATIONS", 1):
        (deliverable, board), log = _run_with_patches(decisions)

    assert len(log["execute_calls"]) == 1
    assert log["controller_calls"] == 1
    assert "max_iterations" in board.stop_reason


# ---------------------------------------------------------------------------
# Test 3: Budget crossed by controller — actions still execute
# ---------------------------------------------------------------------------

def test_budget_crossed_by_controller_dispatches_actions():
    """Entry budget is below threshold, controller call raises it above.
    Actions are returned and must execute before budget stop."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}
    decisions = [
        {"converge": False, "converge_reason": "", "actions": [action]},
    ]

    # budget_used_pct() calls:
    # 1. entry gate check: 50% (below threshold)
    # 2. legacy hazard check: 90% (above threshold — controller spent tokens)
    # 3. shadow log: 90%
    # 4. post-transaction check: 90% (stop)
    budget_sequence = [50.0, 90.0, 90.0, 90.0]

    (deliverable, board), log = _run_with_patches(
        decisions, budget_pcts=budget_sequence,
    )

    assert len(log["execute_calls"]) == 1
    assert "budget" in board.stop_reason


# ---------------------------------------------------------------------------
# Test 4: Budget exhausted on entry — zero controller calls
# ---------------------------------------------------------------------------

def test_budget_exhausted_on_entry_skips_controller():
    """Entry budget is at/above threshold.  Assert zero controller calls
    and zero action dispatches."""
    decisions = [
        {"converge": False, "converge_reason": "",
         "actions": [{"kind": "analyze"}]},
    ]

    # First budget check (entry gate) already at threshold
    budget_sequence = [90.0]

    (deliverable, board), log = _run_with_patches(
        decisions, budget_pcts=budget_sequence,
    )

    assert log["controller_calls"] == 0
    assert len(log["execute_calls"]) == 0
    assert "budget_entry" in board.stop_reason
    assert board.iteration == 0, "entry rejection must not mutate iteration"


# ---------------------------------------------------------------------------
# Test 5: Max exhausted on entry — zero controller calls
# ---------------------------------------------------------------------------

def test_max_exhausted_on_entry_skips_controller():
    """Start with board.iteration == MAX_ITERATIONS.  Assert zero controller
    calls and zero action dispatches."""
    decisions = [
        {"converge": False, "converge_reason": "",
         "actions": [{"kind": "analyze"}]},
    ]

    # MAX_ITERATIONS defaults to 12.  board.iteration starts at 0.
    # With MAX_ITERATIONS=0, entry gate 0 >= 0 triggers immediately.
    with patch("src.loop.MAX_ITERATIONS", 0):
        (deliverable, board), log = _run_with_patches(decisions)

    assert log["controller_calls"] == 0
    assert len(log["execute_calls"]) == 0
    assert "max_iterations_entry" in board.stop_reason
    assert board.iteration == 0, "entry rejection must not mutate iteration"


# ---------------------------------------------------------------------------
# Test 6: Overlapping hazards — convergence wins by precedence
# ---------------------------------------------------------------------------

def test_overlapping_hazards_convergence_wins():
    """Convergence is eligible on the max boundary and budget is also
    exhausted.  Assert one dispatch and convergence stop."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}

    def execute_and_resolve(actions, board):
        for t in board.targets:
            if t.is_open:
                board.resolve_target(t.id, "closed", "test")
        return {"claims": 1}

    decisions = [
        {"converge": True, "converge_reason": "all resolved",
         "actions": [action]},
    ]

    # budget: entry=50 (pass gate), legacy=90 (above threshold),
    # shadow=90, post-transaction=90
    budget_sequence = [50.0, 90.0, 90.0, 90.0]

    with patch("src.loop.MAX_ITERATIONS", 1):
        (deliverable, board), log = _run_with_patches(
            decisions, execute_side_effect=execute_and_resolve,
            budget_pcts=budget_sequence,
        )

    assert len(log["execute_calls"]) == 1
    assert board.stop_reason.startswith("converged:")


# ---------------------------------------------------------------------------
# Test 7: Post-action convergence recheck denies if targets opened
# ---------------------------------------------------------------------------

def test_convergence_denied_when_actions_open_targets():
    """Controller requests convergence with empty pre-action open set.
    Action causes a material target to become open.  Assert convergence
    is denied and the loop continues."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}

    call_count = [0]

    def execute_and_open_target(actions, board):
        call_count[0] += 1
        if call_count[0] == 1:
            board.add_target(Target(
                id="t_new", need="new finding", materiality="critical",
            ))
        return {"claims": 0}

    def board_resolve_all(board):
        for t in board.targets:
            if t.is_open:
                board.resolve_target(t.id, "closed", "pre-resolved")

    decisions = [
        {"converge": True, "converge_reason": "all done",
         "actions": [action]},
        {"converge": True, "converge_reason": "really done",
         "actions": []},
    ]

    with patch("src.loop.MAX_ITERATIONS", 3):
        (deliverable, board), log = _run_with_patches(
            decisions, execute_side_effect=execute_and_open_target,
            board_setup=board_resolve_all,
        )

    assert log["controller_calls"] >= 2
    converge_denied = any(
        e.kind == "converge_denied" for e in board.events
    )
    assert converge_denied


# ---------------------------------------------------------------------------
# Test 8: Controller-only semantic progress resets quiet_rounds
# ---------------------------------------------------------------------------

def test_controller_only_resolution_is_semantic_progress():
    """Controller closes/resolves a target but returns no actions.
    Assert resolved_delta > 0, quiet_rounds resets, and diminishing
    returns does not fire for that transaction."""

    call_count = [0]

    def seed_with_target(caller, board):
        board.add_target(Target(id="t0", need="test", materiality="critical"))

    def fake_controller(caller, board, last_summary, *, max_iterations=12,
                        closeout=False):
        call_count[0] += 1
        if call_count[0] == 1:
            return {"converge": False, "converge_reason": "", "actions": []}
        if call_count[0] == 2:
            board.resolve_target("t0", "closed", "controller judgment")
            return {"converge": False, "converge_reason": "", "actions": []}
        return {"converge": True, "converge_reason": "done", "actions": []}

    patches = dict(_NOOP_PATCHES)
    patches["src.loop.seed_targets"] = seed_with_target
    patches["src.loop.controller_decide"] = fake_controller
    patches["src.loop.execute_actions"] = lambda *a, **kw: {"claims": 0}
    patches["src.loop.auto_bind"] = lambda *a, **kw: {"bound": 0}

    task = _FakeTask()

    with (
        patch.multiple("src.loop", **{
            k.split(".")[-1]: v for k, v in patches.items()
        }),
        patch.object(Board, "snapshot", lambda self, label="": None),
        patch("src.loop.MAX_ITERATIONS", 10),
    ):
        deliverable, board = run_loop(
            task, _stub_caller(), smart_caller=_stub_caller(),
        )

    assert "diminishing" not in board.stop_reason
    shadow_events = [
        e for e in board.events if e.kind == "action_transaction_shadow"
    ]
    assert len(shadow_events) >= 2
    iter2_event = shadow_events[1]
    assert iter2_event.detail.get("semantic_progress") is True


# ---------------------------------------------------------------------------
# Test 9: Genuinely quiet transaction increments quiet_rounds
# ---------------------------------------------------------------------------

def test_genuinely_quiet_transaction_increments_quiet():
    """No actions, no new claims, no derived claim, no resolution delta.
    Assert exactly one quiet-round increment leading to diminishing stop."""
    decisions = [
        {"converge": False, "converge_reason": "", "actions": []},
        {"converge": False, "converge_reason": "", "actions": []},
    ]

    with patch("src.loop.DIMINISHING_ROUNDS", 2):
        (deliverable, board), log = _run_with_patches(decisions)

    assert "diminishing_returns" in board.stop_reason
    shadow_events = [
        e for e in board.events if e.kind == "action_transaction_shadow"
    ]
    for ev in shadow_events:
        assert ev.detail.get("semantic_progress") is False


# ---------------------------------------------------------------------------
# Test 10: No double dispatch
# ---------------------------------------------------------------------------

def test_no_double_dispatch():
    """On a normal nonterminal iteration, each selected action envelope
    reaches execute_actions exactly once."""
    actions = [
        {"kind": "read", "target_id": "t0", "source_id": "s0"},
        {"kind": "analyze", "target_id": "t0", "instruction": "test"},
    ]
    decisions = [
        {"converge": False, "converge_reason": "", "actions": actions},
        {"converge": True, "converge_reason": "done", "actions": []},
    ]

    def execute_and_resolve(acts, board):
        for t in board.targets:
            if t.is_open:
                board.resolve_target(t.id, "closed", "test")
        return {"claims": 1}

    (deliverable, board), log = _run_with_patches(
        decisions, execute_side_effect=execute_and_resolve,
    )

    assert len(log["execute_calls"]) == 1
    assert len(log["execute_calls"][0]) == 2


# ---------------------------------------------------------------------------
# Test 11: Diminishing-return ordering — envelope dispatched before stop
# ---------------------------------------------------------------------------

def test_diminishing_return_dispatches_envelope_first():
    """On the transaction that reaches DIMINISHING_ROUNDS, assert any
    nonempty envelope was already dispatched before the stop."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}
    decisions = [
        {"converge": False, "converge_reason": "", "actions": []},
        {"converge": False, "converge_reason": "", "actions": [action]},
    ]

    with patch("src.loop.DIMINISHING_ROUNDS", 2):
        (deliverable, board), log = _run_with_patches(decisions)

    assert len(log["execute_calls"]) == 1
    assert "diminishing_returns" in board.stop_reason


# ---------------------------------------------------------------------------
# Test 12: Final snapshot contains committed work
# ---------------------------------------------------------------------------

def test_final_snapshot_observes_committed_mutations():
    """A terminal-boundary action mutates the board; assert the board
    state after stop reflects that mutation."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}

    def execute_and_add_claim(actions, board):
        from src.loop.state import Claim
        board.add_claim(Claim(
            kind="analysis", content="derived fact",
            source_doc="s0",
            target_refs=["t0"], created_by="test",
        ))
        return {"claims": 1}

    decisions = [
        {"converge": False, "converge_reason": "", "actions": [action]},
    ]

    with patch("src.loop.MAX_ITERATIONS", 1):
        (deliverable, board), log = _run_with_patches(
            decisions, execute_side_effect=execute_and_add_claim,
        )

    assert any(c.content == "derived fact" for c in board.claims)
    assert "max_iterations" in board.stop_reason


# ---------------------------------------------------------------------------
# Test 13: Shadow record correctness
# ---------------------------------------------------------------------------

def test_shadow_telemetry_records_legacy_hazards():
    """For convergence, max, and budget legacy predicates, assert
    legacy_first_stop, selected count, dispatcher-envelope count,
    and selected_but_undispatched == 0 are emitted correctly."""
    action = {"kind": "analyze", "target_id": "t0", "instruction": "test"}

    def execute_and_resolve(actions, board):
        for t in board.targets:
            if t.is_open:
                board.resolve_target(t.id, "closed", "test")
        return {"claims": 1}

    decisions = [
        {"converge": True, "converge_reason": "done", "actions": [action]},
    ]

    # At max boundary with budget also exhausted
    budget_sequence = [50.0, 90.0, 90.0, 90.0]

    with patch("src.loop.MAX_ITERATIONS", 1):
        (deliverable, board), log = _run_with_patches(
            decisions, execute_side_effect=execute_and_resolve,
            budget_pcts=budget_sequence,
        )

    shadow_events = [
        e for e in board.events if e.kind == "action_transaction_shadow"
    ]
    assert len(shadow_events) == 1
    detail = shadow_events[0].detail
    assert detail["selected_action_count"] == 1
    assert detail["dispatcher_envelope_count"] == 1
    assert detail["selected_but_undispatched"] == 0
    # All three predicates must be present as booleans
    hazards = detail["legacy_hazards"]
    assert isinstance(hazards["convergence"], bool)
    assert isinstance(hazards["max_iterations"], bool)
    assert isinstance(hazards["budget"], bool)
    assert hazards["convergence"] is True
    assert hazards["max_iterations"] is True
    assert hazards["budget"] is True
    assert detail["legacy_first_stop"] == "convergence"
    # Enriched telemetry fields
    assert "derived_added" in detail
    assert "resolved_delta" in detail
    assert "budget_pct_before_dispatch" in detail


# ---------------------------------------------------------------------------
# Test 14: No model-call regression away from hazards
# ---------------------------------------------------------------------------

def test_no_regression_without_hazards():
    """For a trace that never hits a legacy pre-dispatch hazard, assert
    controller and action-dispatch counts are unchanged from expected."""
    actions_1 = [{"kind": "analyze", "target_id": "t0", "instruction": "iter1"}]
    actions_2 = [{"kind": "analyze", "target_id": "t0", "instruction": "iter2"}]

    def execute_resolve_on_second(actions, board):
        if actions[0].get("instruction") == "iter2":
            for t in board.targets:
                if t.is_open:
                    board.resolve_target(t.id, "closed", "done")
        return {"claims": 1}

    decisions = [
        {"converge": False, "converge_reason": "", "actions": actions_1},
        {"converge": False, "converge_reason": "", "actions": actions_2},
        {"converge": True, "converge_reason": "all resolved", "actions": []},
    ]

    (deliverable, board), log = _run_with_patches(
        decisions, execute_side_effect=execute_resolve_on_second,
    )

    assert log["controller_calls"] == 3
    assert len(log["execute_calls"]) == 2

    shadow_events = [
        e for e in board.events if e.kind == "action_transaction_shadow"
    ]
    for ev in shadow_events:
        if ev.detail.get("selected_action_count", 0) > 0:
            assert ev.detail.get("legacy_first_stop") is None
        assert ev.detail.get("selected_but_undispatched") == 0
