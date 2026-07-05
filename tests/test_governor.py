"""Unit tests for the governor core: race exposure, atomicity, lease invariants."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governor import AtomicLedger, NaiveLedger, OutputEstimator, QuotaError, QuotaNode


def test_naive_ledger_races_past_budget():
    """N concurrent calls all pass the naive check before any settles."""

    async def scenario() -> int:
        ledger = NaiveLedger(budget=1000)

        async def one_call() -> None:
            reservation = await ledger.try_reserve(400)
            assert reservation is not None  # everyone is admitted
            await asyncio.sleep(0.01)  # in flight
            await ledger.settle(reservation, 400)

        await asyncio.gather(*(one_call() for _ in range(5)))
        return ledger.overshoot

    assert asyncio.run(scenario()) == 1000  # 5 x 400 spent vs 1000 budget


def test_atomic_ledger_never_overshoots():
    async def scenario() -> tuple[int, int]:
        ledger = AtomicLedger(budget=1000, reserve_fraction=0.0)
        admitted = 0

        async def one_call() -> None:
            nonlocal admitted
            reservation = await ledger.try_reserve(400)
            if reservation is None:
                return
            admitted += 1
            await asyncio.sleep(0.01)
            await ledger.settle(reservation, 400)

        await asyncio.gather(*(one_call() for _ in range(5)))
        return ledger.overshoot, admitted

    overshoot, admitted = asyncio.run(scenario())
    assert overshoot == 0
    assert admitted == 2  # floor(1000 / 400)


def test_completion_reserve_gates_and_releases():
    async def scenario() -> tuple[bool, bool]:
        ledger = AtomicLedger(budget=1000, reserve_fraction=0.20)
        r1 = await ledger.try_reserve(800)  # usable is 800: fits exactly
        assert r1 is not None
        await ledger.settle(r1, 800)
        denied = await ledger.try_reserve(100) is None  # reserve still locked
        ledger.begin_finalization()
        released = await ledger.try_reserve(100) is not None
        return denied, released

    denied, released = asyncio.run(scenario())
    assert denied and released


def test_reconciliation_returns_overestimate():
    async def scenario() -> int:
        ledger = AtomicLedger(budget=1000, reserve_fraction=0.0)
        reservation = await ledger.try_reserve(900)
        await ledger.settle(reservation, 300)  # actual far below the reserve
        return ledger.available

    assert asyncio.run(scenario()) == 700


def test_quota_lease_cannot_exceed_parent():
    root = QuotaNode("root", 1000)
    root.spawn_child("a", 600)
    with pytest.raises(QuotaError):
        root.spawn_child("b", 600)  # only 400 remains


def test_quota_tree_bounds_total_spend():
    root = QuotaNode("root", 1000)
    a = root.spawn_child("a", 500)
    b = root.spawn_child("b", 500)
    aa = a.spawn_child("aa", 300)  # carved from a, not from the pool
    for node, amount in [(a, 200), (b, 500), (aa, 300)]:
        reservation = node.try_consume(amount)
        assert reservation is not None
        node.settle(reservation, amount)
    assert root.tree_spent() == 1000
    assert a.try_consume(1) is None  # a: 200 spent + 300 leased = 500


def test_quota_close_reverts_unspent_lease():
    root = QuotaNode("root", 1000)
    child = root.spawn_child("child", 800)
    reservation = child.try_consume(100)
    child.settle(reservation, 100)
    child.close()
    assert root.remaining == 900  # 800 lease returned, 100 actually spent


def test_estimator_converges_to_history_quantile():
    est = OutputEstimator(prior=2048, quantile=0.9, min_samples=5)
    assert est.predict("agent") == 2048  # prior until enough samples
    for value in [100, 110, 120, 130, 900]:
        est.update("agent", value)
    assert est.predict("agent") == 900  # p90 of the observed tail
    assert est.predict("other_agent") == 2048  # keys are independent


def test_spawn_cascade_self_extinguishes():
    """Recursive spawning under lease semantics runs out of divisible quota."""
    root = QuotaNode("root", 10_000)
    depth = 0
    node = root
    while True:
        child_alloc = int(node.remaining * 0.5)
        if child_alloc < 100:
            break
        node = node.spawn_child(f"d{depth}", child_alloc)
        depth += 1
    assert depth < 10  # geometric decay: 10k * 0.5^d < 100 well before d=10
