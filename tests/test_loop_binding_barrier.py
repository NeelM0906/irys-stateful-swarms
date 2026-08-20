"""Tests for cycle-6 same-iteration binding barrier (auto_bind wiring)."""
from unittest.mock import patch, MagicMock

from src.loop.state import Board, Claim, Source, Target
from src.loop.actions import auto_bind


def _board_with_claims_and_targets(n_claims=3, n_targets=2, bound=False):
    board = Board(instruction="test")
    board.add_source(Source(id="s0", name="doc", path="a.txt", size_bytes=100))
    for i in range(n_targets):
        board.add_target(Target(id=f"t{i}", need=f"need {i}", materiality="critical"))
    for i in range(n_claims):
        c = Claim(id=f"c{i}", kind="observation", content=f"fact {i}",
                  source_doc="doc", evidence=f"evidence {i}",
                  target_refs=[f"t0"] if bound else [])
        board.add_claim(c)
    return board


def test_dormant_no_unbound():
    board = _board_with_claims_and_targets(bound=True)
    assert len(board.unbound_claims()) == 0
    result = auto_bind(board, MagicMock())
    assert result == {"offered": 0, "bound": 0, "unbound_after": 0,
                      "calls": 0, "failures": 0, "invalid": 0}


def test_dormant_no_open_targets_reports_actual_unbound():
    board = Board(instruction="test")
    board.add_source(Source(id="s0", name="doc", path="a.txt", size_bytes=100))
    c = Claim(id="c0", kind="observation", content="fact", source_doc="doc",
              evidence="ev")
    board.add_claim(c)
    result = auto_bind(board, MagicMock())
    assert result["offered"] == 0
    assert result["calls"] == 0
    assert result["unbound_after"] == 1


def test_activates_with_unbound_and_open_targets():
    board = _board_with_claims_and_targets(n_claims=2, bound=False)
    assert len(board.unbound_claims()) == 2
    assert len(board.open_targets()) == 2

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = {"bound": 2}
        result = auto_bind(board, MagicMock())

    assert result["offered"] == 2
    assert result["bound"] == 2
    assert result["calls"] == 1
    assert result["failures"] == 0
    mock_bind.assert_called_once()


def test_handles_batch_failure_gracefully():
    board = _board_with_claims_and_targets(n_claims=2, bound=False)

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.side_effect = RuntimeError("provider error")
        result = auto_bind(board, MagicMock())

    assert result["offered"] == 2
    assert result["bound"] == 0
    assert result["failures"] == 1
    assert result["calls"] == 1


def test_batches_correctly():
    board = _board_with_claims_and_targets(n_claims=130, bound=False)
    assert len(board.unbound_claims()) == 130

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = {"bound": 10}
        result = auto_bind(board, MagicMock())

    assert result["offered"] == 130
    assert mock_bind.call_count == 3  # 60+60+10
    assert result["bound"] == 30  # 10 per batch * 3 batches
    assert result["calls"] == 3


def test_reports_unbound_after():
    board = _board_with_claims_and_targets(n_claims=5, bound=False)

    def partial_bind(job, board, caller):
        for c in job["claims"][:2]:
            board.bind_claim(c.id, ["t0"])
        return {"bound": 2}

    with patch("src.loop.actions._run_bind_batch", side_effect=partial_bind):
        result = auto_bind(board, MagicMock())

    assert result["offered"] == 5
    assert result["bound"] == 2
    assert result["unbound_after"] == 3


def test_does_not_rebind_already_bound():
    board = _board_with_claims_and_targets(n_claims=5, bound=False)
    board.bind_claim("c0", ["t0"])
    board.bind_claim("c1", ["t0"])
    assert len(board.unbound_claims()) == 3

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = {"bound": 3}
        result = auto_bind(board, MagicMock())

    assert result["offered"] == 3


def test_budget_stop_prevents_overrun():
    board = _board_with_claims_and_targets(n_claims=130, bound=False)
    board.token_budget = 1000
    board.total_tokens_used = 900  # 90% used, above default 85% stop

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = {"bound": 10}
        result = auto_bind(board, MagicMock())

    assert result["calls"] == 0
    assert result["offered"] == 130
    assert result["bound"] == 0


def test_budget_stop_mid_batch():
    board = _board_with_claims_and_targets(n_claims=130, bound=False)
    board.token_budget = 10000
    board.total_tokens_used = 0

    call_count = 0
    def side_effect(job, board, caller):
        nonlocal call_count
        call_count += 1
        board.total_tokens_used = 9000  # push to 90% after first call
        return {"bound": 5}

    with patch("src.loop.actions._run_bind_batch", side_effect=side_effect):
        result = auto_bind(board, MagicMock())

    assert result["calls"] == 1
    assert result["bound"] == 5


def test_invalid_response_counted():
    board = _board_with_claims_and_targets(n_claims=2, bound=False)

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = {}  # empty dict = malformed
        result = auto_bind(board, MagicMock())

    assert result["invalid"] == 1
    assert result["bound"] == 0
    assert result["failures"] == 0


def test_none_response_counted_as_invalid():
    board = _board_with_claims_and_targets(n_claims=2, bound=False)

    with patch("src.loop.actions._run_bind_batch") as mock_bind:
        mock_bind.return_value = None
        result = auto_bind(board, MagicMock())

    assert result["invalid"] == 1
    assert result["bound"] == 0


def test_empty_llm_response_counted_as_invalid_e2e():
    """End-to-end: LLM returns {} (no bindings key) through real _run_bind_batch."""
    board = _board_with_claims_and_targets(n_claims=2, bound=False)

    with patch("src.loop.actions.call_json") as mock_json:
        mock_json.return_value = {}
        result = auto_bind(board, MagicMock())

    assert result["invalid"] == 1
    assert result["bound"] == 0
    assert result["calls"] == 1
