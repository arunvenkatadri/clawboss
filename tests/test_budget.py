"""Tests for clawboss.budget — BudgetTracker and BudgetSnapshot."""

import threading

import pytest

from clawboss.budget import BudgetSnapshot, BudgetTracker
from clawboss.errors import ClawbossError
from clawboss.policy import Policy


# ---------------------------------------------------------------------------
# BudgetSnapshot properties
# ---------------------------------------------------------------------------


class TestBudgetSnapshot:
    def test_tokens_remaining_with_limit(self):
        snap = BudgetSnapshot(tokens_used=300, token_limit=1000, iterations=1, iteration_limit=5)
        assert snap.tokens_remaining == 700

    def test_tokens_remaining_no_limit(self):
        snap = BudgetSnapshot(tokens_used=300, token_limit=None, iterations=1, iteration_limit=5)
        assert snap.tokens_remaining is None

    def test_tokens_remaining_over_budget_clamps_to_zero(self):
        snap = BudgetSnapshot(tokens_used=1500, token_limit=1000, iterations=1, iteration_limit=5)
        assert snap.tokens_remaining == 0

    def test_iterations_remaining(self):
        snap = BudgetSnapshot(tokens_used=0, token_limit=None, iterations=2, iteration_limit=5)
        assert snap.iterations_remaining == 3

    def test_iterations_remaining_over_limit_clamps_to_zero(self):
        snap = BudgetSnapshot(tokens_used=0, token_limit=None, iterations=7, iteration_limit=5)
        assert snap.iterations_remaining == 0

    def test_is_over_token_budget_true(self):
        snap = BudgetSnapshot(tokens_used=1000, token_limit=1000, iterations=0, iteration_limit=5)
        assert snap.is_over_token_budget is True

    def test_is_over_token_budget_false(self):
        snap = BudgetSnapshot(tokens_used=500, token_limit=1000, iterations=0, iteration_limit=5)
        assert snap.is_over_token_budget is False

    def test_is_over_token_budget_no_limit(self):
        snap = BudgetSnapshot(tokens_used=999999, token_limit=None, iterations=0, iteration_limit=5)
        assert snap.is_over_token_budget is False

    def test_is_over_iteration_limit_true(self):
        snap = BudgetSnapshot(tokens_used=0, token_limit=None, iterations=5, iteration_limit=5)
        assert snap.is_over_iteration_limit is True

    def test_is_over_iteration_limit_false(self):
        snap = BudgetSnapshot(tokens_used=0, token_limit=None, iterations=3, iteration_limit=5)
        assert snap.is_over_iteration_limit is False


# ---------------------------------------------------------------------------
# BudgetTracker — record_tokens
# ---------------------------------------------------------------------------


class TestBudgetTrackerTokens:
    def test_record_tokens_increments(self):
        bt = BudgetTracker(token_limit=10000, iteration_limit=5)
        total = bt.record_tokens(500)
        assert total == 500
        total = bt.record_tokens(300)
        assert total == 800

    def test_record_tokens_raises_when_over_budget(self):
        bt = BudgetTracker(token_limit=1000, iteration_limit=5)
        bt.record_tokens(800)
        with pytest.raises(ClawbossError) as exc_info:
            bt.record_tokens(300)
        assert exc_info.value.kind == "budget_exceeded"

    def test_record_tokens_exactly_at_limit_does_not_raise(self):
        bt = BudgetTracker(token_limit=1000, iteration_limit=5)
        total = bt.record_tokens(1000)
        assert total == 1000

    def test_record_tokens_no_limit_unlimited(self):
        bt = BudgetTracker(token_limit=None, iteration_limit=5)
        total = bt.record_tokens(999999)
        assert total == 999999
        # Should not raise
        total = bt.record_tokens(999999)
        assert total == 1999998


# ---------------------------------------------------------------------------
# BudgetTracker — record_iteration
# ---------------------------------------------------------------------------


class TestBudgetTrackerIterations:
    def test_record_iteration_increments(self):
        bt = BudgetTracker(token_limit=None, iteration_limit=5)
        assert bt.record_iteration() == 1
        assert bt.record_iteration() == 2
        assert bt.record_iteration() == 3

    def test_record_iteration_raises_when_over_limit(self):
        bt = BudgetTracker(token_limit=None, iteration_limit=2)
        bt.record_iteration()
        bt.record_iteration()
        with pytest.raises(ClawbossError) as exc_info:
            bt.record_iteration()
        assert exc_info.value.kind == "max_iterations"

    def test_record_iteration_exact_limit_succeeds(self):
        bt = BudgetTracker(token_limit=None, iteration_limit=3)
        bt.record_iteration()
        bt.record_iteration()
        count = bt.record_iteration()
        assert count == 3


# ---------------------------------------------------------------------------
# BudgetTracker — snapshot
# ---------------------------------------------------------------------------


class TestBudgetTrackerSnapshot:
    def test_snapshot_initial_values(self):
        bt = BudgetTracker(token_limit=10000, iteration_limit=5)
        snap = bt.snapshot()
        assert snap.tokens_used == 0
        assert snap.token_limit == 10000
        assert snap.iterations == 0
        assert snap.iteration_limit == 5

    def test_snapshot_reflects_usage(self):
        bt = BudgetTracker(token_limit=10000, iteration_limit=5)
        bt.record_tokens(500)
        bt.record_iteration()
        snap = bt.snapshot()
        assert snap.tokens_used == 500
        assert snap.iterations == 1


# ---------------------------------------------------------------------------
# BudgetTracker — from_policy
# ---------------------------------------------------------------------------


class TestBudgetTrackerFromPolicy:
    def test_from_policy_with_token_budget(self):
        policy = Policy(token_budget=5000, max_iterations=3)
        bt = BudgetTracker.from_policy(policy)
        snap = bt.snapshot()
        assert snap.token_limit == 5000
        assert snap.iteration_limit == 3

    def test_from_policy_no_token_budget(self):
        policy = Policy(token_budget=None, max_iterations=10)
        bt = BudgetTracker.from_policy(policy)
        snap = bt.snapshot()
        assert snap.token_limit is None
        assert snap.iteration_limit == 10


# ---------------------------------------------------------------------------
# BudgetTracker — reset
# ---------------------------------------------------------------------------


class TestBudgetTrackerReset:
    def test_reset_clears_counters(self):
        bt = BudgetTracker(token_limit=10000, iteration_limit=5)
        bt.record_tokens(500)
        bt.record_iteration()
        bt.reset()
        snap = bt.snapshot()
        assert snap.tokens_used == 0
        assert snap.iterations == 0

    def test_reset_allows_new_usage(self):
        bt = BudgetTracker(token_limit=100, iteration_limit=1)
        bt.record_tokens(100)
        bt.record_iteration()
        bt.reset()
        # Should not raise after reset
        bt.record_tokens(50)
        bt.record_iteration()
        snap = bt.snapshot()
        assert snap.tokens_used == 50
        assert snap.iterations == 1


# ---------------------------------------------------------------------------
# BudgetTracker — thread safety
# ---------------------------------------------------------------------------


class TestBudgetTrackerThreadSafety:
    def test_concurrent_record_tokens(self):
        bt = BudgetTracker(token_limit=None, iteration_limit=1000)
        num_threads = 10
        tokens_per_thread = 100
        calls_per_thread = 50

        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(calls_per_thread):
                bt.record_tokens(tokens_per_thread)

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        expected = num_threads * tokens_per_thread * calls_per_thread
        snap = bt.snapshot()
        assert snap.tokens_used == expected

    def test_concurrent_record_iterations(self):
        num_threads = 10
        iters_per_thread = 5
        bt = BudgetTracker(token_limit=None, iteration_limit=num_threads * iters_per_thread + 1)

        barrier = threading.Barrier(num_threads)

        def worker():
            barrier.wait()
            for _ in range(iters_per_thread):
                bt.record_iteration()

        threads = [threading.Thread(target=worker) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snap = bt.snapshot()
        assert snap.iterations == num_threads * iters_per_thread
