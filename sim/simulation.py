"""Discrete-event simulations for the three capstone experiments.

Experiment 1 -- concurrent admission (leverage points #12 vs #8/#9):
    N agents fire calls concurrently against one budget under four regimes:
    naive check-then-act, atomic reservation with worst-case estimates,
    atomic reservation with empirical p90 estimates, and a decentralized
    quota tree with purely local decisions.

Experiment 2 -- subagent spawning (leverage point #7):
    a recursive spawn cascade under flat per-spawn grants vs lease
    inheritance, showing exponential blowup vs geometric convergence.

Experiment 3 -- the meter in the hallway (leverage point #6):
    identical agents under identical hard enforcement, with and without
    visibility of the remaining budget; sighted agents skip speculative
    calls proportionally to scarcity, replicating Meadows' Amsterdam
    electricity-meter observation in a multi-agent LLM setting.

Every run is seeded and finishes in a few seconds. Run as a script to save
figures/*.png and print the results tables.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from governor import AppealsDesk, AtomicLedger, NaiveLedger, OutputEstimator, QuotaNode

MAX_OUTPUT_TOKENS = 2048  # configured cap, used as the worst-case estimate
LATENCY_RANGE = (0.004, 0.016)  # simulated seconds per LLM call


@dataclass
class Workload:
    """Per-agent stream of calls with known input and hidden output tokens."""

    agent: str
    calls: list[tuple[int, int]]  # (input_tokens, actual_output_tokens)


def make_workload(
    n_agents: int, calls_per_agent: int, seed: int, uneven: bool = True
) -> list[Workload]:
    rng = np.random.default_rng(seed)
    workloads = []
    for i in range(n_agents):
        # uneven demand: odd agents are twice as chatty as even agents
        n_calls = calls_per_agent * (2 if (uneven and i % 2) else 1)
        inputs = rng.lognormal(mean=6.3, sigma=0.5, size=n_calls).astype(int) + 50
        outputs = rng.lognormal(mean=6.2, sigma=0.8, size=n_calls).astype(int) + 20
        outputs = np.minimum(outputs, MAX_OUTPUT_TOKENS)
        workloads.append(
            Workload(f"agent_{i}", list(zip(inputs.tolist(), outputs.tolist())))
        )
    return workloads


@dataclass
class RunResult:
    condition: str
    budget: int
    spent: int = 0
    completed: int = 0
    denied: int = 0
    timeline: list[tuple[float, int, int]] = field(default_factory=list)

    @property
    def overshoot_pct(self) -> float:
        return 100.0 * max(0, self.spent - self.budget) / self.budget

    @property
    def utilization_pct(self) -> float:
        return 100.0 * min(self.spent, self.budget) / self.budget


# --------------------------------------------------------------------------
# Experiment 1: concurrent admission against one budget
# --------------------------------------------------------------------------

async def _run_central(
    ledger, workloads: list[Workload], estimator: OutputEstimator | None, seed: int
) -> RunResult:
    rng = np.random.default_rng(seed)
    result = RunResult(condition="", budget=ledger.budget)

    async def agent_loop(wl: Workload) -> None:
        strikes = 0
        for input_tokens, actual_output in wl.calls:
            if estimator is None:
                estimate = input_tokens + MAX_OUTPUT_TOKENS
            else:
                estimate = input_tokens + estimator.predict(wl.agent)
            reservation = await ledger.try_reserve(estimate)
            if reservation is None:
                result.denied += 1
                strikes += 1
                if strikes >= 3:
                    return  # the agent knows to stop
                await asyncio.sleep(0.01)
                continue
            strikes = 0
            await asyncio.sleep(rng.uniform(*LATENCY_RANGE))  # call in flight
            await ledger.settle(reservation, input_tokens + actual_output)
            if estimator is not None:
                estimator.update(wl.agent, actual_output)
            result.completed += 1

    await asyncio.gather(*(agent_loop(wl) for wl in workloads))
    result.spent = ledger.spent
    result.timeline = list(ledger.stats.timeline)
    return result


async def _run_quota_tree(
    budget: int, workloads: list[Workload], seed: int
) -> RunResult:
    rng = np.random.default_rng(seed)
    root = QuotaNode("coordinator", allocation=budget)
    result = RunResult(condition="quota tree (local)", budget=budget)
    start = time.monotonic()

    async def agent_loop(wl: Workload, node: QuotaNode) -> None:
        for input_tokens, actual_output in wl.calls:
            estimate = input_tokens + MAX_OUTPUT_TOKENS
            reservation = node.try_consume(estimate)  # purely local, no lock
            if reservation is None:
                result.denied += 1
                return  # local quota exhausted: the agent stops by itself
            await asyncio.sleep(rng.uniform(*LATENCY_RANGE))
            node.settle(reservation, input_tokens + actual_output)
            result.completed += 1
            result.timeline.append(
                (time.monotonic() - start, root.tree_spent(), 0)
            )

    slice_ = budget // len(workloads)
    nodes = [root.spawn_child(wl.agent, slice_) for wl in workloads]
    await asyncio.gather(*(agent_loop(wl, n) for wl, n in zip(workloads, nodes)))
    result.spent = root.tree_spent()
    return result


async def run_experiment_concurrency_async(
    budget: int = 150_000, n_agents: int = 16, calls_per_agent: int = 12, seed: int = 7
) -> list[RunResult]:
    results = []

    workloads = make_workload(n_agents, calls_per_agent, seed)
    r = await _run_central(NaiveLedger(budget), workloads, None, seed)
    r.condition = "naive check-then-act"
    results.append(r)

    workloads = make_workload(n_agents, calls_per_agent, seed)
    r = await _run_central(
        AtomicLedger(budget, reserve_fraction=0.10), workloads, None, seed
    )
    r.condition = "atomic, worst-case reserve"
    results.append(r)

    workloads = make_workload(n_agents, calls_per_agent, seed)
    r = await _run_central(
        AtomicLedger(budget, reserve_fraction=0.10),
        workloads,
        OutputEstimator(prior=MAX_OUTPUT_TOKENS),
        seed,
    )
    r.condition = "atomic, p90 reserve + 10% buffer"
    results.append(r)

    workloads = make_workload(n_agents, calls_per_agent, seed)
    results.append(await _run_quota_tree(budget, workloads, seed))
    return results


def run_experiment_concurrency(**kwargs) -> list[RunResult]:
    return asyncio.run(run_experiment_concurrency_async(**kwargs))


# --------------------------------------------------------------------------
# Experiment 2: subagent spawning -- flat grants vs lease inheritance
# --------------------------------------------------------------------------

@dataclass
class SpawnResult:
    policy: str
    budget: int
    spent: int
    agents_spawned: int
    spent_by_depth: dict[int, int]

    @property
    def overshoot_pct(self) -> float:
        return 100.0 * max(0, self.spent - self.budget) / self.budget


def _spawn_flat(
    budget: int, per_agent_cap: int, branching: int, max_depth: int, rng
) -> SpawnResult:
    """Every agent gets the same standalone per-agent cap -- no inheritance.

    This is how limits are usually configured in practice ("each agent may
    use up to N tokens") and it does not compose under spawning: each spawned
    agent brings a fresh cap with it, so the global exposure is
    cap x (number of agents), and the number of agents grows exponentially
    with depth. The reinforcing loop has no gain limiter.
    """
    spent_by_depth: dict[int, int] = {}
    state = {"spent": 0, "agents": 0}

    def visit(depth: int) -> None:
        state["agents"] += 1
        work = int(rng.lognormal(6.0, 0.6)) + 100  # tokens this agent burns
        work = min(work, per_agent_cap)  # its own cap is the only brake
        state["spent"] += work
        spent_by_depth[depth] = spent_by_depth.get(depth, 0) + work
        if depth >= max_depth:
            return
        for _ in range(branching):
            visit(depth + 1)

    visit(0)
    return SpawnResult(
        "per-agent cap, no inheritance", budget, state["spent"], state["agents"], spent_by_depth
    )


def _spawn_lease(
    budget: int, lease_fraction: float, branching: int, max_depth: int, rng
) -> SpawnResult:
    """Children carve their allocation out of the parent's remaining lease."""
    root = QuotaNode("root", allocation=budget)
    spent_by_depth: dict[int, int] = {}
    agents = 0

    def visit(node: QuotaNode, depth: int) -> None:
        nonlocal agents
        agents += 1
        work = int(rng.lognormal(6.0, 0.6)) + 100
        reservation = node.try_consume(min(work, node.remaining))
        if reservation is not None:
            node.settle(reservation, reservation.amount)
            spent_by_depth[depth] = spent_by_depth.get(depth, 0) + reservation.amount
        if depth >= max_depth:
            return
        for i in range(branching):
            child_alloc = int(node.remaining * lease_fraction)
            if child_alloc < 200:  # not worth spawning: the cascade self-extinguishes
                break
            child = node.spawn_child(f"{node.name}.{i}", child_alloc)
            visit(child, depth + 1)
            child.close()  # unspent lease reverts to the parent

    visit(root, 0)
    return SpawnResult(
        "lease inheritance", budget, root.tree_spent(), agents, spent_by_depth
    )


def run_experiment_spawning(
    budget: int = 60_000, branching: int = 3, max_depth: int = 5, seed: int = 11
) -> list[SpawnResult]:
    rng = np.random.default_rng(seed)
    flat = _spawn_flat(
        budget, per_agent_cap=4000, branching=branching, max_depth=max_depth, rng=rng
    )
    rng = np.random.default_rng(seed)
    lease = _spawn_lease(
        budget, lease_fraction=0.30, branching=branching, max_depth=max_depth, rng=rng
    )
    return [flat, lease]


# --------------------------------------------------------------------------
# Experiment 3: the meter in the hallway (budget visibility)
# --------------------------------------------------------------------------

@dataclass
class MeterResult:
    condition: str
    budget: int
    spent: int
    tasks_completed: int
    speculative_calls: int


async def _run_meter(budget: int, sighted: bool, seed: int) -> MeterResult:
    """Tasks need 4 core calls each; between core calls agents are tempted to
    make speculative calls (prob 0.5). Sighted agents damp that probability by
    the remaining-budget fraction -- the only difference between conditions.
    """
    rng = np.random.default_rng(seed)
    ledger = AtomicLedger(budget, reserve_fraction=0.0)
    estimator = OutputEstimator(prior=MAX_OUTPUT_TOKENS)
    tasks_completed = 0
    speculative = 0

    async def one_call(agent: str, input_tokens: int, output_tokens: int) -> bool:
        nonlocal speculative
        estimate = input_tokens + estimator.predict(agent)
        reservation = await ledger.try_reserve(estimate)
        if reservation is None:
            return False
        await asyncio.sleep(rng.uniform(*LATENCY_RANGE))
        await ledger.settle(reservation, input_tokens + output_tokens)
        estimator.update(agent, output_tokens)
        return True

    async def agent_loop(agent: str) -> None:
        nonlocal tasks_completed, speculative
        while True:
            core_done = 0
            for _ in range(4):  # core calls: the task itself
                if not await one_call(
                    agent, int(rng.lognormal(6.3, 0.4)), int(rng.lognormal(6.0, 0.6))
                ):
                    return
                core_done += 1
                p_speculative = 0.5
                if sighted:  # the meter in the hallway
                    p_speculative *= max(0.0, ledger.available / ledger.budget)
                if rng.random() < p_speculative:
                    speculative += 1
                    if not await one_call(
                        agent, int(rng.lognormal(6.5, 0.4)), int(rng.lognormal(6.3, 0.6))
                    ):
                        return
            if core_done == 4:
                tasks_completed += 1

    await asyncio.gather(*(agent_loop(f"agent_{i}") for i in range(6)))
    return MeterResult(
        "sighted (meter visible)" if sighted else "blind (no meter)",
        budget,
        ledger.spent,
        tasks_completed,
        speculative,
    )


async def run_experiment_meter_async(
    budget: int = 120_000, seed: int = 23
) -> list[MeterResult]:
    return [
        await _run_meter(budget, sighted=False, seed=seed),
        await _run_meter(budget, sighted=True, seed=seed),
    ]


def run_experiment_meter(**kwargs) -> list[MeterResult]:
    return asyncio.run(run_experiment_meter_async(**kwargs))


# --------------------------------------------------------------------------
# Experiment 4: the right of appeal (voice)
# --------------------------------------------------------------------------

@dataclass
class AppealExpResult:
    condition: str
    budget: int
    spent: int
    tasks_completed: int
    tasks_stranded: int
    wasted_tokens: int  # spend sunk into tasks that never finished
    appeals_granted: int

    @property
    def overshoot_pct(self) -> float:
        return 100.0 * max(0, self.spent - self.budget) / self.budget


async def _run_appeal_exp(budget: int, enabled: bool, seed: int) -> AppealExpResult:
    """Tasks need 5 calls; abandon one mid-way and all its prior spend is
    wasted (stranded work). A call is *appealable* when its task already has
    sunk spend: finishing started work is what protects the mission.

    Same hard cap in both conditions. With appeals enabled, ordinary admission
    stops at 85% and the 85-95% tranche is reachable only by appealed calls
    that rescue in-progress tasks; disabled, ordinary admission uses the full
    95% but the wall is blind to what it strands.
    """
    ledger = AtomicLedger(
        budget,
        reserve_fraction=0.05,
        appeal_fraction=0.10 if enabled else 0.0,
    )
    desk = AppealsDesk(ledger, max_grants_per_agent=2)
    estimator = OutputEstimator(prior=MAX_OUTPUT_TOKENS)
    completed = stranded = wasted = 0

    async def agent_loop(agent: str, rng) -> None:
        # per-agent rng stream: token draws don't depend on asyncio
        # interleaving, so both conditions face the same workload
        nonlocal completed, stranded, wasted
        while True:
            task_spend = 0
            task_appealed = False  # a won appeal covers the whole task
            for _ in range(5):
                input_tokens = int(rng.lognormal(6.3, 0.4))
                output_tokens = int(rng.lognormal(6.0, 0.6))
                estimate = input_tokens + estimator.predict(agent)
                reservation = await ledger.try_reserve(estimate, priority=task_appealed)
                if reservation is None and enabled and task_spend > 0 and not task_appealed:
                    # the appeal is for the administrative act -- finishing
                    # this task -- not for a single call
                    reservation = await desk.appeal(
                        agent, estimate,
                        f"task in progress with {task_spend} tokens sunk: "
                        "completing it protects the mission's prior spend",
                    )
                    task_appealed = reservation is not None
                if reservation is None:
                    if task_spend:
                        stranded += 1
                        wasted += task_spend
                    return
                await asyncio.sleep(rng.uniform(*LATENCY_RANGE))
                actual = input_tokens + output_tokens
                await ledger.settle(reservation, actual)
                estimator.update(agent, output_tokens)
                task_spend += actual
            completed += 1

    await asyncio.gather(
        *(
            agent_loop(f"agent_{i}", np.random.default_rng((seed, i)))
            for i in range(8)
        )
    )
    return AppealExpResult(
        "with right of appeal" if enabled else "hard wall (no appeal)",
        budget, ledger.spent, completed, stranded, wasted, desk.log.granted,
    )


async def run_experiment_appeals_async(
    budget: int = 100_000, seed: int = 31
) -> list[AppealExpResult]:
    return [
        await _run_appeal_exp(budget, enabled=False, seed=seed),
        await _run_appeal_exp(budget, enabled=True, seed=seed),
    ]


def run_experiment_appeals(**kwargs) -> list[AppealExpResult]:
    return asyncio.run(run_experiment_appeals_async(**kwargs))


# --------------------------------------------------------------------------
# Figures and report
# --------------------------------------------------------------------------

def plot_concurrency(results1, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)

    # Figure 1: spent(+committed) timelines vs the budget line
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, r in zip(axes, [results1[0], results1[2]]):
        if r.timeline:
            t0 = r.timeline[0][0]
            ts = [p[0] - t0 for p in r.timeline]
            spent = [p[1] for p in r.timeline]
            committed = [p[1] + p[2] for p in r.timeline]
            ax.plot(ts, spent, label="spent", color="tab:blue")
            ax.plot(ts, committed, label="spent + committed", color="tab:orange", ls="--")
        ax.axhline(r.budget, color="tab:red", label="budget")
        ax.set_title(r.condition)
        ax.set_xlabel("time (s)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("tokens")
    fig.suptitle("Overshoot needs a delay: invisible in-flight calls vs reservation")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_overshoot_timeline.png", dpi=140)

    # Figure 2: overshoot vs throughput vs utilization per condition
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    names = [r.condition.replace(", ", ",\n") for r in results1]
    for ax, metric, title in zip(
        axes,
        [
            [r.overshoot_pct for r in results1],
            [r.completed for r in results1],
            [r.utilization_pct for r in results1],
        ],
        ["overshoot (% of budget)", "calls completed", "budget utilization (%)"],
    ):
        bars = ax.bar(names, metric, color=["tab:red", "tab:blue", "tab:green", "tab:purple"])
        ax.bar_label(bars, fmt="%.1f", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelsize=7, rotation=20)
    fig.suptitle("Exp. 1 — concurrent admission: correctness vs throughput trade-off")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_conditions.png", dpi=140)


def plot_spawning(results2, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.38
    depths = sorted({d for r in results2 for d in r.spent_by_depth})
    for i, r in enumerate(results2):
        vals = [r.spent_by_depth.get(d, 0) for d in depths]
        ax.bar(
            [d + (i - 0.5) * width for d in depths], vals, width,
            label=f"{r.policy} (total {r.spent:,}, overshoot {r.overshoot_pct:.0f}%)",
        )
    ax.axhline(results2[0].budget, color="tab:red", ls=":", label="budget")
    ax.set_xlabel("spawn depth")
    ax.set_ylabel("tokens spent at depth")
    ax.set_title("Exp. 2 — lease inheritance turns exponential spawning geometric")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_spawning.png", dpi=140)


def plot_meter(results3, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    names = [r.condition for r in results3]
    tasks = [r.tasks_completed for r in results3]
    spec = [r.speculative_calls for r in results3]
    x = np.arange(len(names))
    b1 = ax.bar(x - 0.18, tasks, 0.36, label="tasks completed on budget")
    b2 = ax.bar(x + 0.18, spec, 0.36, label="speculative calls made")
    ax.bar_label(b1)
    ax.bar_label(b2)
    ax.set_xticks(x, names)
    ax.set_title("Exp. 3 — the meter in the hallway (same budget, same enforcement)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_meter.png", dpi=140)


def plot_appeals(results4, outdir: Path) -> None:
    import matplotlib.pyplot as plt

    outdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    names = [r.condition for r in results4]
    x = np.arange(len(names))

    b1 = axes[0].bar(x - 0.18, [r.tasks_completed for r in results4], 0.36,
                     label="tasks completed", color="tab:green")
    b2 = axes[0].bar(x + 0.18, [r.tasks_stranded for r in results4], 0.36,
                     label="tasks stranded", color="tab:red")
    axes[0].bar_label(b1)
    axes[0].bar_label(b2)
    axes[0].set_xticks(x, names, fontsize=9)
    axes[0].set_title("delivered vs stranded work", fontsize=10)
    axes[0].legend(fontsize=8)

    b3 = axes[1].bar(names, [r.wasted_tokens for r in results4], color="tab:orange")
    axes[1].bar_label(b3, fmt="{:,.0f}")
    axes[1].set_title("tokens sunk into unfinished tasks", fontsize=10)
    axes[1].tick_params(axis="x", labelsize=9)

    fig.suptitle(
        "Exp. 4 — the right of appeal: same hard cap, less stranded work "
        f"(appeals granted: {results4[1].appeals_granted})"
    )
    fig.tight_layout()
    fig.savefig(outdir / "fig5_appeals.png", dpi=140)


def plot_all(results1, results2, results3, outdir: Path, results4=None) -> None:
    import matplotlib

    matplotlib.use("Agg")
    plot_concurrency(results1, outdir)
    plot_spawning(results2, outdir)
    plot_meter(results3, outdir)
    if results4 is not None:
        plot_appeals(results4, outdir)


def print_concurrency(results1) -> None:
    print(f"{'condition':38} {'overshoot%':>10} {'completed':>10} {'denied':>7} {'util%':>7}")
    for r in results1:
        print(
            f"{r.condition:38} {r.overshoot_pct:>10.2f} {r.completed:>10} "
            f"{r.denied:>7} {r.utilization_pct:>7.1f}"
        )


def print_spawning(results2) -> None:
    for r in results2:
        print(
            f"{r.policy:30} spent={r.spent:>8,}  agents={r.agents_spawned:>4}  "
            f"overshoot={r.overshoot_pct:6.1f}%"
        )


def print_meter(results3) -> None:
    for r in results3:
        print(
            f"{r.condition:26} spent={r.spent:>8,}  tasks={r.tasks_completed:>3}  "
            f"speculative={r.speculative_calls:>3}"
        )


def print_appeals(results4) -> None:
    for r in results4:
        print(
            f"{r.condition:24} spent={r.spent:>8,}  completed={r.tasks_completed:>3}  "
            f"stranded={r.tasks_stranded:>2}  wasted={r.wasted_tokens:>7,}  "
            f"appeals={r.appeals_granted}  overshoot={r.overshoot_pct:.1f}%"
        )


def main() -> None:
    results1 = run_experiment_concurrency()
    results2 = run_experiment_spawning()
    results3 = run_experiment_meter()
    results4 = run_experiment_appeals()

    print("\nExp. 1 — concurrent admission (budget 150,000 tokens)")
    print_concurrency(results1)

    print("\nExp. 2 — subagent spawning (budget 60,000 tokens)")
    print_spawning(results2)

    print("\nExp. 3 — budget visibility (budget 120,000 tokens)")
    print_meter(results3)

    print("\nExp. 4 — the right of appeal (budget 80,000 tokens)")
    print_appeals(results4)

    plot_all(
        results1, results2, results3,
        Path(__file__).resolve().parents[1] / "figures",
        results4=results4,
    )
    print("\nFigures saved to figures/")


if __name__ == "__main__":
    main()
