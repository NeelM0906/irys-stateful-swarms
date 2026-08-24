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
CANONICAL_RESOLUTION = os.getenv("LOOP_CANONICAL_RESOLUTION", "0").strip() in ("1", "true", "yes")
# Contract plane (obligations/units/coverage plan/gate). Default OFF: v5/v6
# showed macro regressions; re-enable per-experiment until validated on its
# target class with paired runs.
CONTRACT_ENABLED = os.getenv("LOOP_CONTRACT", "0").strip() in ("1", "true", "yes")
BUDGET_STOP_PCT = float(os.getenv("LOOP_BUDGET_STOP_PCT", "85"))
DIMINISHING_ROUNDS = 2
ANALYSIS_ENRICHMENT = os.getenv("LOOP_ANALYSIS_ENRICHMENT", "0").strip() in ("1", "true", "yes")
_ENRICHMENT_MAX_CALLS = int(os.getenv("LOOP_ENRICHMENT_MAX_CALLS", "15"))
_ENRICHMENT_MIN_OBS = int(os.getenv("LOOP_ENRICHMENT_MIN_OBS", "3"))
_ENRICHMENT_MAX_DERIVED = int(os.getenv("LOOP_ENRICHMENT_MAX_DERIVED", "1"))
# Iterations at which the blackboard is rebuilt (reframe pass): the ledger is
# re-derived from accumulated understanding — splits, new questions, reopens.
REFRAME_ITERATIONS = tuple(
    int(x) for x in os.getenv("LOOP_REFRAME_ITERATIONS", "3,7").split(",") if x.strip()
)
AUDIT_EVERY = int(os.getenv("LOOP_AUDIT_EVERY", "3"))


def _analysis_enrichment(board, worker_caller, smart_caller):
    """Post-convergence pass: run analyze on under-analyzed targets.

    Fires only when LOOP_ANALYSIS_ENRICHMENT=1. Identifies targets with
    many raw observations but few derived claims and dispatches analysis
    calls to increase the derive ratio before synthesis.
    """
    if board.budget_used_pct() >= BUDGET_STOP_PCT:
        board.log("analysis_enrichment", "skipped: budget exhausted")
        return

    candidates = []
    for t in board.targets:
        if t.status == "waived":
            continue
        bound = board.claims_for_target(t)
        obs = [c for c in bound if c.active and not c.is_derived]
        derived = [c for c in bound if c.active and c.is_derived]
        if len(obs) >= _ENRICHMENT_MIN_OBS and len(derived) <= _ENRICHMENT_MAX_DERIVED:
            candidates.append((t, len(obs), len(derived)))

    candidates.sort(key=lambda x: (-x[1], x[0].rank))
    candidates = candidates[:_ENRICHMENT_MAX_CALLS]

    if not candidates:
        board.log("analysis_enrichment", "no under-analyzed targets found")
        return

    board.log(
        "analysis_enrichment",
        f"enriching {len(candidates)} under-analyzed targets "
        f"(min_obs={_ENRICHMENT_MIN_OBS}, max_derived={_ENRICHMENT_MAX_DERIVED})",
        detail={"targets": [c[0].id for c in candidates]},
    )

    actions = []
    for t, obs_count, derived_count in candidates:
        actions.append({
            "kind": "analyze",
            "target_id": t.id,
            "instruction": (
                f"This target has {obs_count} raw observations but only "
                f"{derived_count} derived conclusions. Derive calculations, "
                f"comparisons, cross-document connections, risk assessments, "
                f"and actionable conclusions from the bound evidence."
            ),
        })

    derived_before = sum(1 for c in board.claims if c.is_derived)
    summary = execute_actions(actions, board, worker_caller,
                              smart_caller=smart_caller)
    derived_added = sum(1 for c in board.claims if c.is_derived) - derived_before

    if derived_added > 0 or summary.get("claims", 0) > 0:
        bind_result = auto_bind(board, worker_caller,
                                budget_stop_pct=BUDGET_STOP_PCT)
        board.log("analysis_enrichment_bind", "post-enrichment auto_bind",
                  detail=bind_result)

    board.log(
        "analysis_enrichment_done",
        f"enrichment complete: {derived_added} derived claims added from "
        f"{len(candidates)} targets",
        detail={**summary, "derived_added": derived_added},
    )
    board.snapshot("enriched")


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
        # ── Entry-admission gates ──
        # Do not pay for a controller call when no transaction can begin.
        # Gates fire before the iteration increment so rejected entries
        # leave board.iteration unchanged (Codex R8 finding 1).
        if board.iteration >= MAX_ITERATIONS:
            board.stop_reason = f"max_iterations_entry ({MAX_ITERATIONS})"
            break
        if board.budget_used_pct() >= BUDGET_STOP_PCT:
            board.stop_reason = (
                f"budget_entry ({board.budget_used_pct()}%); "
                f"{len(board.material_open_targets())} material targets open"
            )
            break

        board.iteration += 1

        derived_before = sum(1 for c in board.claims if c.is_derived)
        resolved_before = len(board.resolved_targets())

        decision = controller_decide(
            smart, board, last_summary,
            max_iterations=MAX_ITERATIONS, closeout=closeout,
        )

        # ── Legacy hazard telemetry (shadow measurement of old ordering) ──
        # Record what the pre-P1 stop ordering would have done at this point,
        # before dispatch. All three predicates are always present as booleans.
        pre_dispatch_open = board.material_open_targets()
        pre_dispatch_mandatory = board.open_mandatory_obligations()
        budget_pct_before_dispatch = board.budget_used_pct()
        converge_would_fire = bool(
            decision["converge"]
            and not pre_dispatch_open
            and not pre_dispatch_mandatory
        )
        legacy_hazards = {
            "convergence": converge_would_fire,
            "max_iterations": board.iteration >= MAX_ITERATIONS,
            "budget": budget_pct_before_dispatch >= BUDGET_STOP_PCT,
        }
        legacy_first_stop = None
        for hazard_key in ("convergence", "max_iterations", "budget"):
            if legacy_hazards[hazard_key]:
                legacy_first_stop = hazard_key
                break

        # ── Committed transaction: dispatch the full admitted envelope ──
        actions = tuple(decision.get("actions") or ())
        action_summary: dict = {}
        if actions:
            action_summary = execute_actions(
                list(actions), board, worker_caller, smart_caller=smart,
            )

        derived_added = sum(1 for c in board.claims if c.is_derived) - derived_before
        resolved_delta = len(board.resolved_targets()) - resolved_before
        action_summary["derived_added"] = derived_added

        new_claims = (derived_added > 0) or (action_summary.get("claims", 0) > 0)
        if new_claims:
            bind_result = auto_bind(board, worker_caller,
                                    budget_stop_pct=BUDGET_STOP_PCT)
            action_summary["auto_bind"] = bind_result

        last_summary = action_summary
        semantic_progress = bool(
            derived_added > 0
            or resolved_delta > 0
            or action_summary.get("claims", 0) > 0
        )
        quiet_rounds = 0 if semantic_progress else quiet_rounds + 1

        board.log(
            "action_transaction_shadow",
            "admitted controller transaction completed",
            detail={
                "iteration": board.iteration,
                "selected_action_count": len(actions),
                "dispatcher_envelope_count": len(actions),
                "selected_but_undispatched": 0,
                "derived_added": derived_added,
                "resolved_delta": resolved_delta,
                "legacy_hazards": legacy_hazards,
                "legacy_first_stop": legacy_first_stop,
                "semantic_progress": semantic_progress,
                "budget_pct_before_dispatch": budget_pct_before_dispatch,
                "budget_pct_after_transaction": board.budget_used_pct(),
            },
        )

        # ── Post-transaction stop predicates (read committed state) ──
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
                detail={
                    "targets": [t.id for t in material_open],
                    "obligations": [o.id for o in open_mandatory],
                },
            )

        if board.iteration >= MAX_ITERATIONS:
            board.stop_reason = (
                f"max_iterations ({MAX_ITERATIONS}); "
                f"{len(material_open)} material targets open"
            )
            break

        post_budget_pct = board.budget_used_pct()
        if post_budget_pct >= BUDGET_STOP_PCT:
            board.stop_reason = (
                f"budget ({post_budget_pct}%); "
                f"{len(material_open)} material targets open"
            )
            break

        if quiet_rounds >= DIMINISHING_ROUNDS:
            board.stop_reason = (
                f"diminishing_returns ({quiet_rounds} quiet rounds); "
                f"{len(material_open)} material targets open"
            )
            break

        # ── Reframe / maintenance / audit / forced-analysis / snapshot ──
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

        forced = []
        _force_analysis_gate(board, forced)
        if forced:
            derived_before_forced = sum(1 for c in board.claims if c.is_derived)
            extra = execute_actions(forced, board, worker_caller,
                                       smart_caller=smart)
            derived_added_forced = (
                sum(1 for c in board.claims if c.is_derived) - derived_before_forced
            )
            board.log("force_analyze_exec",
                      f"executed {len(forced)} forced analyze actions",
                      detail=extra)
            if derived_added_forced > 0 or extra.get("claims", 0) > 0:
                bind_result = auto_bind(board, worker_caller,
                                        budget_stop_pct=BUDGET_STOP_PCT)
                board.log("force_analyze_bind", "post-forced auto_bind",
                          detail=bind_result)

        board.snapshot()

    board.log("stop", board.stop_reason)
    board.snapshot("final")

    if ANALYSIS_ENRICHMENT:
        _analysis_enrichment(board, worker_caller, smart)

    resolutions = None
    if CANONICAL_RESOLUTION:
        from .resolution import resolve_all
        resolutions = resolve_all(smart, board)
        board.snapshot("resolved")

    plan = plan_synthesis(smart, board, resolutions=resolutions)
    deliverable = synthesize(synth, board, plan, repair_caller=smart,
                             resolutions=resolutions)
    write_final_state(board)
    return deliverable, board
