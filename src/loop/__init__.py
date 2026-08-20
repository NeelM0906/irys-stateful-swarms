"""The target-closing loop — one loop over one state model.

    seed → triage → [controller → execute → maintain]* → plan → synthesize

Every iteration the controller sees compact state and decides; executors
do bounded work in parallel; convergence is explicit closure of material
targets with a recorded stop reason — never a silent timer expiry.
"""
from __future__ import annotations

import os

from .actions import auto_bind, bulk_extract_frontier, execute_actions, finalize_bulk_extraction
from .control import (
    _force_analysis_gate,
    blackboard_audit, controller_decide, maintain_ledger, reframe_ledger,
    seed_targets, should_maintain,
)
from .state import Board, Source
from .synthesis import plan_synthesis, synthesize, write_final_state
from .triage import triage_sources

MAX_ITERATIONS = int(os.getenv("LOOP_MAX_ITERATIONS", "12"))
# Contract plane (obligations/units/coverage plan/gate). Default OFF: v5/v6
# showed macro regressions; re-enable per-experiment until validated on its
# target class with paired runs.
CONTRACT_ENABLED = os.getenv("LOOP_CONTRACT", "0").strip() in ("1", "true", "yes")
BUDGET_STOP_PCT = float(os.getenv("LOOP_BUDGET_STOP_PCT", "85"))
DIMINISHING_ROUNDS = 2
# Iterations at which the blackboard is rebuilt (reframe pass): the ledger is
# re-derived from accumulated understanding — splits, new questions, reopens.
REFRAME_ITERATIONS = tuple(
    int(x) for x in os.getenv("LOOP_REFRAME_ITERATIONS", "3,7").split(",") if x.strip()
)
AUDIT_EVERY = int(os.getenv("LOOP_AUDIT_EVERY", "3"))


def run_loop(task, worker_caller, smart_caller=None, synthesis_caller=None,
             audit_caller=None):
    """Run the loop on a task. Returns (deliverable, board).

    worker_caller: cheap tier — read/search/bind/verify executors.
    smart_caller: judgment tier — seed/triage/controller/maintenance + analyze.
    synthesis_caller: optional dedicated tier for plan_synthesis/synthesize.
                      Falls back to smart_caller.
    audit_caller: optional stronger model for periodic blackboard audit.
    """
    smart = smart_caller or worker_caller
    synth = synthesis_caller or smart
    auditor = audit_caller or synth
    board = Board(
        instruction=task.instruction,
        metadata=dict(task.metadata or {}),
        output_dir=task.output_dir,
        token_budget=int(os.getenv("LOOP_TOKEN_BUDGET", "3000000")),
    )
    board.metadata["contract_enabled"] = CONTRACT_ENABLED
    for doc in task.documents:
        board.add_source(Source(
            id=doc.id, name=doc.name,
            path=str(doc.metadata.get("path", "")),
            size_bytes=doc.size_bytes,
            _doc=doc,
        ))

    # Think before reading; triage before thinking about everything.
    seed_targets(synth, board)
    triage_sources(smart, board)
    # Large validated frontiers get one bulk extraction pass over every
    # definite candidate before the controller loop (dormant elsewhere).
    bulk = bulk_extract_frontier(board, worker_caller)
    finalize_bulk_extraction(board, bulk, BUDGET_STOP_PCT)
    board.snapshot("seed")

    last_summary: dict = {}
    quiet_rounds = 0
    open_history: list[int] = []
    closeout = False

    while True:
        board.iteration += 1

        decision = controller_decide(
            smart, board, last_summary,
            max_iterations=MAX_ITERATIONS, closeout=closeout,
        )

        # --- convergence policy ---
        material_open = board.material_open_targets()
        open_mandatory = board.open_mandatory_obligations()
        open_history.append(len(material_open) + len(open_mandatory))
        closeout = (
            board.iteration >= MAX_ITERATIONS // 2
            and len(open_history) >= 2
            and open_history[-1] >= open_history[-2] > 0
        )
        if closeout:
            board.log(
                "closeout",
                f"{len(material_open)} material targets + "
                f"{len(open_mandatory)} mandatory obligations not shrinking",
            )
        if decision["converge"]:
            if not material_open and not open_mandatory:
                board.stop_reason = f"converged: {decision['converge_reason']}"
                break
            board.log(
                "converge_denied",
                f"{len(material_open)} material targets, "
                f"{len(open_mandatory)} mandatory obligations still open",
                detail={"targets": [t.id for t in material_open],
                        "obligations": [o.id for o in open_mandatory]},
            )
        if board.iteration >= MAX_ITERATIONS:
            board.stop_reason = (
                f"max_iterations ({MAX_ITERATIONS}); "
                f"{len(material_open)} material targets open"
            )
            break
        if board.budget_used_pct() >= BUDGET_STOP_PCT:
            board.stop_reason = (
                f"budget ({board.budget_used_pct()}%); "
                f"{len(material_open)} material targets open"
            )
            break

        if not decision["actions"]:
            quiet_rounds += 1
        else:
            derived_before = sum(1 for c in board.claims if c.is_derived)
            resolved_before = len(board.resolved_targets())
            last_summary = execute_actions(decision["actions"], board, worker_caller,
                                              smart_caller=smart)
            derived_added = sum(1 for c in board.claims if c.is_derived) - derived_before
            resolved_delta = len(board.resolved_targets()) - resolved_before
            last_summary["derived_added"] = derived_added
            if derived_added == 0 and resolved_delta == 0 and last_summary.get("claims", 0) == 0:
                quiet_rounds += 1
            else:
                quiet_rounds = 0

            new_claims = (derived_added > 0) or (last_summary.get("claims", 0) > 0)
            if new_claims:
                bind_result = auto_bind(board, worker_caller,
                                        budget_stop_pct=BUDGET_STOP_PCT)
                last_summary["auto_bind"] = bind_result

        if quiet_rounds >= DIMINISHING_ROUNDS:
            board.stop_reason = (
                f"diminishing_returns ({quiet_rounds} quiet rounds); "
                f"{len(board.material_open_targets())} material targets open"
            )
            break

        if board.iteration in REFRAME_ITERATIONS:
            reframe_ledger(smart, board)
            open_history.clear()
            closeout = False
        elif should_maintain(board) or closeout:
            maintain_ledger(smart, board, closeout=closeout)

        if (AUDIT_EVERY > 0
                and board.iteration > 0
                and board.iteration % AUDIT_EVERY == 0
                and board.open_targets()):
            try:
                blackboard_audit(auditor, board)
            except Exception as exc:
                board.log("blackboard_audit", f"audit failed: {exc}",
                          detail={"error": str(exc)})

        # Maintenance/reframe/audit can waive targets — catch unanalyzed ones.
        forced = []
        _force_analysis_gate(board, forced)
        if forced:
            extra = execute_actions(forced, board, worker_caller,
                                       smart_caller=smart)
            board.log("force_analyze_exec",
                      f"executed {len(forced)} forced analyze actions",
                      detail=extra)

        board.snapshot()

    board.log("stop", board.stop_reason)
    board.snapshot("final")

    plan = plan_synthesis(smart, board)
    deliverable = synthesize(synth, board, plan, repair_caller=smart)
    write_final_state(board)
    return deliverable, board
