"""Unit tests for the governor core: race exposure, atomicity, lease invariants."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governor import (
    AppealsDesk,
    AtomicLedger,
    NaiveLedger,
    OutputEstimator,
    QuotaError,
    QuotaNode,
)


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


def test_quota_tree_spent_no_double_count_after_close():
    # A closed child's spend is rolled into the parent AND the child is dropped
    # from the parent's list, so tree_spent() counts it once, not twice. Before
    # the fix this scenario (the lease-inheritance sim) reported ~2x the spend.
    root = QuotaNode("root", 1000)
    child = root.spawn_child("child", 800)
    grandchild = child.spawn_child("grandchild", 400)
    for node, amount in [(root, 50), (child, 100), (grandchild, 200)]:
        node.settle(node.try_consume(amount), amount)
    grandchild.close()
    child.close()
    assert root.tree_spent() == 350  # 50 + 100 + 200, counted exactly once
    assert root.children == []  # closed children do not accumulate


def test_estimator_converges_to_history_quantile():
    est = OutputEstimator(prior=2048, quantile=0.9, min_samples=5)
    assert est.predict("agent") == 2048  # prior until enough samples
    for value in [100, 110, 120, 130, 900]:
        est.update("agent", value)
    assert est.predict("agent") == 900  # p90 of the observed tail
    assert est.predict("other_agent") == 2048  # keys are independent


def test_appeal_enters_tranche_but_never_the_reserve():
    """Ordinary stops at 80%; appeals reach 90%; the last 10% stays locked."""

    async def scenario():
        ledger = AtomicLedger(budget=1000, reserve_fraction=0.10, appeal_fraction=0.10)
        desk = AppealsDesk(ledger)
        r = await ledger.try_reserve(800)  # exactly the ordinary ceiling
        await ledger.settle(r, 800)
        assert await ledger.try_reserve(100) is None  # ordinary: denied
        granted = await desk.appeal("agent", 100, "critical synthesis step")
        assert granted is not None  # appeal tranche: admitted
        await ledger.settle(granted, 100)
        # even an appeal cannot touch the completion reserve
        assert await desk.appeal("agent", 50, "please") is None
        assert ledger.overshoot == 0

    asyncio.run(scenario())


def test_appeal_requires_justification_and_is_rationed():
    async def scenario():
        ledger = AtomicLedger(budget=10_000, reserve_fraction=0.10, appeal_fraction=0.10)
        desk = AppealsDesk(ledger, max_grants_per_agent=2)
        assert await desk.appeal("a", 100, "") is None  # no silent overrides
        assert await desk.appeal("a", 100, "reason 1") is not None
        assert await desk.appeal("a", 100, "reason 2") is not None
        assert await desk.appeal("a", 100, "reason 3") is None  # ration exhausted
        assert await desk.appeal("b", 100, "reason") is not None  # per-agent cap
        assert desk.log.granted == 3 and desk.log.refused == 2  # accountability

    asyncio.run(scenario())


def test_judge_rules_and_the_hearing_is_billed():
    """A GRANT admits the appeal; the hearing's own tokens hit the ledger."""
    from governor import MissionJudge

    async def scenario():
        ledger = AtomicLedger(budget=10_000, reserve_fraction=0.10, appeal_fraction=0.10)

        async def fake_model(prompt: str) -> tuple[str, int]:
            assert "the mission" in prompt.lower()
            return ("REFUSE" if "speculative" in prompt else "GRANT", 120)

        judge = MissionJudge("ship the report", ledger, caller=fake_model)
        desk = AppealsDesk(ledger, judge=judge.rule)

        assert await desk.appeal("a", 100, "final synthesis step") is not None
        assert await desk.appeal("b", 100, "speculative side quest") is None
        assert judge.hearings == 2  # both cases were heard...
        assert ledger.spent == 240  # ...and both hearings were billed

    asyncio.run(scenario())


def test_judge_abstains_offline_and_refuses_when_broke(monkeypatch):
    from governor import MissionJudge

    # caller=None means "auto-detect from the environment"; strip any real API
    # keys so the test exercises the offline path instead of calling Gemini.
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    async def scenario():
        ledger = AtomicLedger(budget=1000, reserve_fraction=0.0, appeal_fraction=0.0)
        # offline (no caller): abstains -> desk's mechanical policy decides
        judge = MissionJudge("mission", ledger, caller=None)
        assert await judge.rule("a", 100, "reason") is True

        # broke: cannot afford the hearing -> the appeal is refused outright
        async def fake_model(prompt: str) -> tuple[str, int]:
            return "GRANT", 10

        r = await ledger.try_reserve(1000)
        await ledger.settle(r, 1000)  # budget fully spent
        judge = MissionJudge("mission", ledger, caller=fake_model)
        assert await judge.rule("a", 100, "reason") is False
        assert judge.hearings == 0

    asyncio.run(scenario())


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
