"""Builds the self-contained Kaggle capstone notebook from the repo sources.

The notebook inlines governor/{ledger,estimator,quota}.py and sim/simulation.py
so it runs on Kaggle with no repo attached; the ADK plugin cell is guarded so
"Run All" works whether or not google-adk is installed.

    python build_notebook.py   ->  overshoot-governor-capstone.ipynb
"""

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def strip_lines(source: str, needles: list[str]) -> str:
    return "\n".join(
        line for line in source.splitlines()
        if not any(n in line for n in needles)
    )


def cut_from(source: str, marker: str) -> str:
    idx = source.find(marker)
    return source[:idx].rstrip() + "\n" if idx != -1 else source


ledger_src = read("src/governor/ledger.py")
estimator_src = read("src/governor/estimator.py")
quota_src = read("src/governor/quota.py")
appeals_src = strip_lines(read("src/governor/appeals.py"), ["from .ledger import"])
judge_src = strip_lines(read("src/governor/judge.py"), ["from .ledger import"])

sim_src = read("sim/simulation.py")
sim_src = cut_from(sim_src, "def main() -> None:")
sim_src = strip_lines(
    sim_src,
    [
        "import sys",
        "sys.path.insert",
        "from governor import",
        "import matplotlib\n",
    ],
)
# plot_all's Agg backend switch would suppress inline rendering; drop it
sim_src = sim_src.replace(
    "    import matplotlib\n\n    matplotlib.use(\"Agg\")\n    plot_concurrency",
    "    plot_concurrency",
)

plugin_src = read("src/governor/adk_plugin.py")
plugin_src = strip_lines(
    plugin_src,
    [
        "from __future__ import annotations",
        "from .appeals import",
        "from .estimator import",
        "from .judge import",
        "from .ledger import",
    ],
)
plugin_guarded = (
    "import importlib.util\n\n"
    "ADK_AVAILABLE = importlib.util.find_spec(\"google.adk\") is not None\n"
    "# On Kaggle with internet enabled you can:  %pip install -q google-adk\n\n"
    "if ADK_AVAILABLE:\n"
    + "\n".join("    " + line if line.strip() else "" for line in plugin_src.splitlines())
    + "\n    print(\"google-adk found:\", BudgetGovernorPlugin(budget=10_000).report())\n"
    "else:\n"
    "    print(\"google-adk is not installed here; the plugin code above is\"\n"
    "          \" shown/defined only when it is. See demo/run_adk_demo.py in the repo.\")\n"
)

md = []
code = []

INTRO = """\
# Overshoot Governor
## Token-budget admission control for multi-agent systems, designed with Donella Meadows

**Kaggle capstone — Roberto García Patrón** · code: [github.com/RobertoGPAI](https://github.com/RobertoGPAI) · security skill: [SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE)

Multi-agent systems burn tokens the way growing economies burn resources: several
actors draw concurrently from a shared, finite stock, each learns the true cost of
its actions only *after* taking them, and any of them can spawn more actors.
In *The Limits to Growth* and *Thinking in Systems*, Donella Meadows showed that
this exact structure — **growth + a limit + a delay in the feedback about the
limit** — is the recipe for *overshoot and collapse*. A token budget has all
three ingredients: bursty concurrent calls (growth), the budget (limit), and
in-flight calls whose cost is unknown until they complete (delay).

This capstone builds a **budget governor** for an ADK 2.0 multi-agent system and
evaluates it with four seeded simulation experiments plus a live ADK demo.
The design is organized deliberately as a walk *up* Meadows' ladder of
[leverage points](https://donellameadows.org/archives/leverage-points-places-to-intervene-in-a-system/),
from the weakest intervention (tweak a parameter) to the strong ones
(information flows and rules):

| Mechanism in this project | Meadows leverage point |
|---|---|
| Raise the budget and hope | **#12** — constants and parameters (the failing baseline) |
| Completion reserve, 90% soft limit | **#11** — sizes of buffers |
| Reserve/reconcile accounting of in-flight calls | **#9** — lengths of delays |
| Atomic admission ledger nobody can race past | **#8** — strength of balancing feedback loops |
| Lease-inherited quotas that damp subagent spawning | **#7** — gain of reinforcing loops |
| Live budget state + the overall mission injected into each agent's context | **#6** — structure of information flows |
| Admission protocol, lease semantics, and a **right of appeal** against denials | **#5** — rules of the system |
| "Never endanger the mission to save tokens" — appeals argue *from* the mission | **#3** — the goal of the system |

**Related work.** LLM gateways and load balancers (e.g. Cordon, LiteLLM router,
provider-side rate limits) distribute traffic across endpoints for latency and
quota compliance. This project addresses a different layer: *admission control
against a finite budget inside one multi-agent system*, with in-flight
accounting, hierarchical quotas for spawned subagents, and the budget treated
as a dynamic system rather than a counter.
"""

GOVERNOR_MD = """\
## 1 · The governor core

Three small, dependency-free components (the full package, tests and ADK demo
live in the repo):

**Ledgers** — `NaiveLedger` is the failing baseline: it admits calls while the
*settled* spend is under the budget (the billing-dashboard pattern). In-flight
calls are invisible to it, so concurrent agents race past the limit — the
balancing loop is weak and delayed. `AtomicLedger` fixes both defects at once:
admission runs under a lock (decisions are never simultaneous) and it reserves
`input + estimated output` *before* the call, reconciling with the actual usage
after. A configurable **completion reserve** (leverage point #11) keeps a slice
of budget that only `begin_finalization()` unlocks, so the mission can always
afford to land.

**How do you know the cost before it happens?** You split it. *Input* tokens
are deterministic — countable before the call. *Output* tokens are not, so the
`OutputEstimator` keeps a rolling history of actual outputs per agent (fed from
usage metadata at settle time) and reserves a high quantile (p90). Any
estimation error is absorbed by the soft-limit buffer and corrected at
reconciliation. Setting `max_output_tokens` to the remaining allocation turns
the estimate into a hard guarantee.

**Quota tree** — subagents never get budget from the global pool: `spawn_child`
carves the child's allocation *out of the parent's remaining lease*, so
`Σ children ≤ parent` holds at every node and total exposure is bounded by the
root no matter how deep the spawn tree grows. Decisions are purely local (no
lock, no contention): the agent *knows when to stop by itself*. Closing a node
returns unspent quota to the parent.

**Appeals desk** — a denial is a contestable administrative act, not a wall.
A denied agent may appeal by stating why the call is critical to the overall
mission (which every agent carries in its context, next to its own task); a
granted appeal admits the call into a protected *appeal tranche* — never into
the completion reserve. Appeals are rationed per agent and logged with their
justifications. In Hirschman's terms, governed agents get *voice*, not just
compliance.
"""

EXP1_MD = """\
## 2 · Experiment 1 — concurrent admission (leverage points #12 vs #8/#9)

16 agents with uneven demand fire calls concurrently against one 150k-token
budget. Four regimes: the naive baseline, atomic reservation with worst-case
estimates, atomic reservation with empirical p90 estimates, and the
decentralized quota tree. The workload is seeded; call latencies are real
`asyncio` sleeps, so in-flight windows are real.
"""

EXP1_READ_MD = """\
**Reading the results.** The naive ledger overshoots by roughly **12–14%** — and
note *when*: the spent curve crosses the budget line while a whole wave of
admitted calls is still in flight. That is Meadows' delay ingredient, isolated.
Both atomic regimes eliminate overshoot entirely (the dashed *spent+committed*
curve is the one that respects the ceiling — the governor controls the
anticipated stock, not the measured one). The quota tree also never overshoots
and almost never has to say "no" (one local denial per agent, versus dozens of
central denials), but pays for its autonomy in utilization: a static even split
strands quota with the light agents. That trade — **exactness costs
coordination; autonomy costs utilization** — is the central finding.
"""

EXP2_MD = """\
## 3 · Experiment 2 — subagent spawning (leverage point #7)

What happens when agents can create agents? The common configuration
("each agent may spend up to N tokens") does not compose: every spawned agent
brings a fresh cap, so exposure is `cap × agents` and agents grow exponentially
with depth. Lease inheritance changes the loop's *gain*: a child can only
subdivide what its parent had left, so the cascade decays geometrically and
self-extinguishes — no recursion detector needed.
"""

EXP2_READ_MD = """\
**Reading the results.** Same spawn eagerness, same task appetite: per-agent
caps without inheritance produce **~360 agents and ~260% overshoot**; lease
inheritance produces ~40 agents, zero overshoot, and the per-depth spend decays
geometrically. A runaway spawn cascade — whether from a bug or from a prompt
injection — starves instead of exploding. This is also the security story:
the lease tree is the structural mitigation for the "agent fork bomb"
(see the STRIDE section below).
"""

EXP3_MD = """\
## 4 · Experiment 3 — the meter in the hallway (leverage point #6)

Meadows' canonical example for *information flows*: identical Amsterdam houses,
except some had the electricity meter in the hallway and some in the basement.
The hallway houses used ~30% less energy — no new rule, no new price, just the
signal reaching the decision-maker.

Here, identical agent teams run identical tasks under identical hard
enforcement. The *only* difference: **sighted** agents see the remaining budget
and damp their speculative (nice-to-have) calls proportionally to scarcity;
**blind** agents don't. In the live ADK system this is one flag — the governor
plugin appends the ledger state to each request's system instruction.
"""

EXP3_READ_MD = """\
**Reading the results.** Same budget, same wall: the sighted team completes
**more tasks** with **fewer speculative calls** — self-restraint funded by
information, not enforcement. The agent "compels itself not to act" precisely
when the forgone action doesn't endanger the overall goal, because the
constraint is part of what it perceives rather than a wall it discovers by
hitting it. The meter in the hallway works for agents too.
"""

EXP4_MD = """\
## 5 · Experiment 4 — the right of appeal (leverage point #5 serving #3)

A hard limit that cannot be contested strands work: a task denied at its
final synthesis call wastes everything already spent on it. But if agents can
override limits freely, the limit is fiction. The democratic answer is due
process: a denied agent may **appeal** — one line stating why finishing this
task protects the mission — and a granted appeal covers the rest of that task
from a protected tranche (85–95% of budget) that ordinary admission can never
touch. The completion reserve stays inviolable; appeals are rationed (2 per
agent) and logged with their justifications.

Both conditions below face the same hard cap. The difference is whether the
wall can hear reasons.
"""

EXP4_READ_MD = """\
**Reading the results.** Same ceiling, zero overshoot in both conditions —
but the hard wall strands ~6 tasks (~17k tokens of sunk work wasted), while
the appeals condition strands almost none, completes *more* tasks, and spends
*less* overall: ordinary admission stops earlier (85%), and the protected
tranche is used surgically, only where sunk cost justifies it. A handful of
granted appeals buys back nearly all the stranded value. The rule (#5) works
better because it answers to the goal (#3): the agent "compels itself not to
act" except when acting is precisely what protects the mission — and then it
gets to say so.
"""

ADK_MD = """\
## 6 · The governor as an ADK 2.0 Runner plugin

In ADK 2.x, a **plugin registered on the Runner** applies its callbacks to every
LLM call of *every agent and subagent* — exactly the semantics a governor needs
(a per-agent callback could be bypassed by a spawned subagent; a Runner plugin
cannot). The implementation below is the actual repo module:

- `before_model_callback` — counts the deterministic input side, adds the p90
  output estimate, and **atomically reserves** against the shared ledger.
  Before every ordinary admission it runs a **runway check**: landing the
  mission costs one more read of a context that grows every turn, so the
  governor keeps enough headroom to land *afterwards* — and at the point of
  no return it lands NOW: releases the completion reserve, strips the tool
  declarations (the only possible output is text — the deliverable), caps
  `max_output_tokens` to what fits, and admits one final call. A refusal
  `LlmResponse` (which short-circuits ADK's flow and, inside one invocation,
  becomes the agent's *final message*) fires only when not even the landing
  fits — but it states the right of appeal, and a retry carrying
  `APPEAL: <reason>` is routed through the AppealsDesk. If admitted normally
  and `visibility=True`, it appends the live budget state **and the overall
  mission** to the request's system instruction (the meter), so restraint
  and appeals alike can be weighed against the goal they serve.
- `after_model_callback` — reconciles the reservation with the actual
  `usage_metadata` and feeds the estimator.
- `on_model_error_callback` — cancels the reservation so failures don't leak
  committed budget.

Wiring it up (see `demo/run_adk_demo.py` for the runnable A/B demo —
coordinator + researcher + writer over Gemini, same task with the meter on and
off):

```python
from google.adk.apps.app import App
from google.adk.runners import InMemoryRunner

governor = BudgetGovernorPlugin(budget=20_000, reserve_fraction=0.10, visibility=True)
app = App(name="overshoot_demo", root_agent=coordinator, plugins=[governor])
runner = InMemoryRunner(app=app)
```
"""

SECURITY_MD = """\
## 7 · Security: the governor *is* the mitigation

The project treats unbounded token spend as a security problem, not just a cost
problem, and was analyzed with [SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE),
a STRIDE threat-modeling skill for agent workspaces (full `threat_model.md` in
the repo). The mapping is direct:

| STRIDE threat | Instance in a multi-agent system | Mitigation in this project |
|---|---|---|
| **D**enial of service | Prompt-injected agent loops tool calls until the budget is drained | Atomic ledger: hard admission ceiling |
| **D**enial of service | Spawn cascade ("agent fork bomb") multiplies exposure | Lease-inherited quotas: cascade self-extinguishes |
| **E**levation of privilege | Subagent escapes its parent's limits | Runner-level plugin + lease invariant: no agent exists outside the tree |
| **T**ampering | Agent inflates its own quota via crafted content | Ledger state lives outside the model context; visibility is read-only |
| **R**epudiation | "Which agent spent the budget?" | Per-agent reservation log with settle-time reconciliation |
| **D**oS on the *mission* | Hard limit strands the task unfinished at 99% | Completion reserve released by `begin_finalization()` |

Repo hygiene follows the course's security lessons: **pre-commit** hooks and a
**Semgrep** CI workflow (`p/python` + `p/security-audit` rulesets) run on every
push.
"""

CLOSING_MD = """\
## 8 · Limitations and future work

- **Single-process ledger.** `asyncio.Lock` serializes admission inside one
  Runner. Multiple runners need a shared atomic ledger (e.g. Redis `INCRBY`
  with a Lua check) — same protocol, different substrate.
- **Heuristic input counting** (~4 chars/token) in the offline plugin; swap in
  the Gemini `count_tokens` endpoint for exact pre-call input counts.
- **Static quota split.** A slow rebalancing loop (coordinator reallocates
  leases every N seconds) would recover the utilization the quota tree gives
  up; Meadows would call that hierarchy done right — fast local control, slow
  global reallocation.
- **Degradation ladder.** Before denying, step down: cheaper model → truncated
  context → answer with what you have. Only the last rung is a refusal.
- **The meter experiment with real models.** `demo/run_adk_demo.py` runs the
  A/B with live Gemini calls; scaling it to significance is the natural next
  study: *does telling an LLM agent its remaining budget change its behavior?*

## 9 · Conclusion

A token budget in a multi-agent system is a stock with delayed outflow
feedback, and it fails the way Meadows said such systems fail: overshoot from
in-flight delay, explosion from an ungoverned reinforcing loop. The fixes that
worked are her high-leverage interventions, implemented as ~200 lines of
Python: account for the delay (reserve/reconcile), cap the loop gain (lease
inheritance), buffer the landing (completion reserve) — and put the meter in
the hallway, because an agent that can *see* the limit compels itself not to
act long before it has to be stopped.

### References
- D. Meadows, *Thinking in Systems: A Primer* (2008)
- D. Meadows, *Leverage Points: Places to Intervene in a System* (1999)
- Meadows, Meadows, Randers, Behrens, *The Limits to Growth* (1972)
- [Google ADK docs — Plugins](https://google.github.io/adk-docs/plugins/) · [ADK Agent Skills](https://developers.googleblog.com/developers-guide-to-building-adk-agents-with-skills/)
- [SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE) — STRIDE threat modeling skill for agent workspaces
"""

nb = nbf.v4.new_notebook()
cells = [
    nbf.v4.new_markdown_cell(INTRO),
    nbf.v4.new_markdown_cell(GOVERNOR_MD),
    nbf.v4.new_code_cell(
        "%matplotlib inline\nimport asyncio\nfrom pathlib import Path\n\n"
        "import numpy as np\nimport matplotlib.pyplot as plt\n\n"
        "FIGDIR = Path(\"figures\")"
    ),
    nbf.v4.new_code_cell(ledger_src),
    nbf.v4.new_code_cell(estimator_src),
    nbf.v4.new_code_cell(quota_src),
    nbf.v4.new_code_cell(appeals_src),
    nbf.v4.new_code_cell(judge_src),
    nbf.v4.new_code_cell(sim_src),
    nbf.v4.new_markdown_cell(EXP1_MD),
    nbf.v4.new_code_cell(
        "results1 = await run_experiment_concurrency_async()\n"
        "print_concurrency(results1)\n"
        "plot_concurrency(results1, FIGDIR)"
    ),
    nbf.v4.new_markdown_cell(EXP1_READ_MD),
    nbf.v4.new_markdown_cell(EXP2_MD),
    nbf.v4.new_code_cell(
        "results2 = run_experiment_spawning()\n"
        "print_spawning(results2)\n"
        "plot_spawning(results2, FIGDIR)"
    ),
    nbf.v4.new_markdown_cell(EXP2_READ_MD),
    nbf.v4.new_markdown_cell(EXP3_MD),
    nbf.v4.new_code_cell(
        "results3 = await run_experiment_meter_async()\n"
        "print_meter(results3)\n"
        "plot_meter(results3, FIGDIR)"
    ),
    nbf.v4.new_markdown_cell(EXP3_READ_MD),
    nbf.v4.new_markdown_cell(EXP4_MD),
    nbf.v4.new_code_cell(
        "results4 = await run_experiment_appeals_async()\n"
        "print_appeals(results4)\n"
        "plot_appeals(results4, FIGDIR)"
    ),
    nbf.v4.new_markdown_cell(EXP4_READ_MD),
    nbf.v4.new_markdown_cell(ADK_MD),
    nbf.v4.new_code_cell(plugin_guarded),
    nbf.v4.new_markdown_cell(SECURITY_MD),
    nbf.v4.new_markdown_cell(CLOSING_MD),
]
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

out = ROOT / "overshoot-governor-capstone.ipynb"
nbf.write(nb, out)
print(f"wrote {out} ({len(cells)} cells)")
