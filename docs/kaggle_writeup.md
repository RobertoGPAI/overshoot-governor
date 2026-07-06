# Overshoot Governor
## Stopping denial-of-wallet in multi-agent systems with system dynamics — a token-budget governor built on ADK 2.0

**Track: Agents for Business**

---

### The problem: agent fleets overshoot their budgets, and the failure is structural

Every business deploying agentic AI meets the same uncomfortable moment: the bill. Multi-agent systems consume tokens the way growing economies consume resources — several autonomous actors draw concurrently from one finite budget, each learns the true cost of its actions only *after* taking them, and any of them can spawn more actors. The result is a familiar incident pattern: the dashboard said we were fine, and twenty minutes later we were 40% over budget.

Naive protections fail for reasons worth naming precisely:

1. **Check-then-act races.** Two agents read "5,000 tokens available" at the same time; both decide they fit; both call. Queues don't fix this — the *decision* is what must not be simultaneous.
2. **The feedback delay.** A call's real cost is unknown until it completes. Any controller watching only *settled* spend admits an entire wave of in-flight calls past the limit.
3. **Spawning doesn't compose.** The common configuration — "each agent may spend up to N tokens" — multiplies exposure with every spawned subagent: `N × agents`, and agents grow exponentially with spawn depth. A prompt-injected spawn loop is an **agent fork bomb** aimed at your wallet.
4. **Dumb hard limits endanger the mission.** A wall that cuts the system off at 99% completion converts a cost problem into a delivery failure.

Fifty years ago, Donella Meadows showed (*The Limits to Growth*, *Thinking in Systems*) that **growth + a limit + a delay in the feedback about the limit** is precisely the recipe for *overshoot and collapse*. A token budget has all three ingredients. So instead of treating this as a rate-limiting feature, this project treats it as what it is — a dynamic system — and fixes it with Meadows' own toolbox.

### The solution: a governor, designed as a walk up Meadows' leverage points

**Overshoot Governor** is admission control for token budgets in ADK 2.0 multi-agent systems. Each mechanism is an intervention at a named rung of Meadows' *Leverage Points: Places to Intervene in a System*, from weakest to strongest:

| Mechanism | Leverage point |
|---|---|
| Raise the budget and hope (failing baseline) | #12 — parameters |
| Completion reserve; 90% soft limit | #11 — buffer sizes |
| Reserve/reconcile accounting of in-flight calls | #9 — lengths of delays |
| Atomic admission ledger nobody can race past | #8 — balancing feedback loops |
| Lease-inherited quotas that damp spawning | #7 — reinforcing-loop gain |
| Budget state + overall mission injected into agent context | #6 — information flows |
| Admission protocol, leases, and a **right of appeal** | #5 — rules |
| "Never endanger the mission" — appeals argue *from* the mission | #3 — system goal |

**Architecture** (see cover image). The design is deliberately *one policy core, two adapters*: the governor core (ledger, estimator, quota tree) is a framework-agnostic policy engine — pure Python, no I/O, fully unit-testable — and both the ADK plugin and the MCP server are thin adapters over the same ledger, so enforcement semantics live in one place and cannot drift between surfaces. A team of ADK `LlmAgent`s (coordinator → researcher, writer) runs under a Runner. The governor is a **Runner plugin** — registered once, its callbacks cover every LLM call of every agent *and every spawned subagent*, which is exactly the semantics a governor needs: a per-agent callback could be bypassed by a spawned child; a Runner plugin cannot.

- `before_model_callback` counts the deterministic input side, adds an empirical **p90 estimate** of the output side, and **atomically reserves** `input + p90(output)` against the shared ledger. If denied, it returns a refusal `LlmResponse` and the model is never called.
- `after_model_callback` **reconciles** the reservation with the actual `usage_metadata` and feeds the estimator. In-flight cost is never invisible: the controlled quantity is `spent + committed`, not `spent`.
- A **completion reserve** (10%) is admissible only after `begin_finalization()` — the mission can always afford to land.
- **Quota lease tree:** a subagent never gets budget from the global pool; `spawn_child()` carves its allocation out of the parent's remainder, so `Σ children ≤ parent` holds at every node. Spawning — a reinforcing loop — has its gain capped: cascades decay geometrically and self-extinguish. Closing a node returns unspent quota to its parent.
- **The meter in the hallway:** optionally, the plugin appends the live budget state — *and the overall mission* — to each request's system instruction, next to the agent's own task, so agents can economize *before* hitting the wall and weigh every action against the goal their restraint must serve.
- **The right of appeal:** a denial is a contestable administrative act, not a wall. The refusal message states the right of appeal; a retry carrying `APPEAL: <reason tied to the mission>` is heard by an AppealsDesk that may admit the call into a protected *appeal tranche* (ordinary admission can never touch it; the completion reserve stays inviolable). Appeals are rationed per agent and logged with their justifications — governed agents get voice, not just compliance. Optionally, appeals are heard by a **judge agent** (`MissionJudge`): a separate, minimal arbiter — deliberately *not* the coordinator that allocates quotas, since no one should be judge in their own cause — that rules whether the justification actually serves the mission. And justice is not free: each hearing reserves and settles tokens on the same ledger it arbitrates, so an attacker flooding the appeal channel exhausts the hearing budget, never the mission's. Running this live exposed a structural subtlety: a denial *silences its addressee* — the refusal short-circuits the model call, so the denied agent never actually hears the verdict; to appeal, it would need the very call it was just denied. Voice must therefore come from the next turn, filed *ex officio* the way a public prosecutor speaks for a party that cannot speak for itself: the driver or a parent coordinator appeals on the mission's behalf, not the agent's. The live demo (`demo/run_adk_demo.py --appeal-demo`) stages the full arc — the budget bites mid-mission, the appeal is filed, the `MissionJudge` hears it, and the team lands the mission from the protected tranche.
- An **MCP server** exposes the same ledger (`reserve` / `settle` / `budget_status` / `begin_finalization`) over the standard protocol, so mixed fleets — ADK teams, Claude Code sessions, CI scripts — share one budget: cross-runtime governance.

**How do you know the cost before it happens?** You split it. Input tokens are deterministic — countable pre-call. Output tokens are not, so the estimator keeps a rolling per-agent history of actual outputs (fed at settle time) and reserves the p90; estimation error is absorbed by the soft-limit buffer and corrected at reconciliation. Setting `max_output_tokens` to the remaining allocation turns the estimate into a hard guarantee.

### The evidence: four seeded experiments

All four run in the submitted notebook/repo in seconds, with real `asyncio` concurrency (real in-flight windows).

**Experiment 1 — concurrent admission.** Sixteen agents with uneven demand fire concurrently at a 150k-token budget under four regimes. The naive check-then-act baseline overshoots **~13%** — and the timeline shows *why*: spend crosses the budget line while a whole wave of admitted calls is still in flight. Meadows' delay ingredient, isolated. Both atomic regimes (worst-case and p90 reservations) hit **0% overshoot** at ~89% utilization. The decentralized quota tree also never overshoots and almost never has to say "no" (one local denial per agent vs. ~48 central denials) but pays in utilization (78%): a static even split strands quota with light agents. The finding in one line: **exactness costs coordination; autonomy costs utilization** — so we ship both, ledger as hard guarantee, leases as the contention-free fast path.

**Experiment 2 — subagent spawning.** Same spawn eagerness, same appetite, two policies. Per-agent caps without inheritance: **364 agents, 259% overshoot**. Lease inheritance: **44 agents, 0% overshoot**, per-depth spend decaying geometrically until spawning is no longer affordable — the cascade extinguishes itself, no recursion detector needed. This is the fork-bomb mitigation, structurally.

**Experiment 3 — the meter in the hallway.** Meadows' canonical example for information flows: identical Amsterdam houses used ~30% less electricity when the meter was in the hallway instead of the basement. We replicate the structure with agents: identical teams, identical tasks, identical hard enforcement; the only difference is whether agents *see* the remaining budget (and damp speculative calls with scarcity). Blind: 15 tasks completed, 30 speculative calls. Sighted: **20 tasks, 16 speculative calls** — a third more delivered value from the same budget, funded by information rather than enforcement. The repo includes a live A/B (`demo/run_adk_demo.py`) running the same comparison over real Gemini calls.

**Experiment 4 — the right of appeal.** A limit that cannot be contested strands work: a task denied mid-way wastes everything already sunk into it. Both conditions here face the same hard cap; the difference is whether the wall can hear reasons. The hard wall completes 14 tasks, strands 7, and wastes ~17k tokens of sunk work. With appeals — a denied agent may state why finishing its in-progress task protects the mission, and a granted appeal covers that task from the protected tranche — the system completes **15 tasks, strands 1, wastes ~5k**, and spends *less* overall, because ordinary admission stops earlier and the protected tranche is used surgically, only where sunk cost justifies it. Zero overshoot in both. Five granted appeals buy back nearly all the stranded value: the rule works better because it answers to the goal.

The answer to the design question that started this project — *how does an agent compel itself not to act, as long as the overall goal is not endangered?* — turns out to be layered like a polity: the goal is protected by the completion reserve (#11), restraint is made rational by putting the constraint and the mission into the agent's information flow (#6), due process turns denials into contestable acts rather than walls (#5 serving #3), and the ledger (#8) stays as the guarantee nobody can talk their way past. Enforcement, information, and voice — law, transparency, and due process. The agent is governed the way citizens are, not the way resources are.

### Security: the governor *is* the mitigation

Unbounded token spend is a security problem, not just a cost problem. The repo was threat-modeled with **[SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE)**, our STRIDE-methodology agent skill (full `security/threat_model.md` in the repo): prompt-injection-driven budget drain and spawn cascades are **Denial-of-Service on the wallet**, mitigated by the atomic ledger and the lease tree; subagents escaping limits is **Elevation of Privilege**, mitigated by the Runner-level plugin plus the lease invariant; the enforcement channel never trusts model-generated text (the visibility string is advisory only — hard limits hold even if an attacker spoofs governor messages). The appeal channel is designed against abuse: appeals are rationed per agent, the tranche is bounded and never touches the completion reserve, and every appeal — granted or refused — is logged with its justification. Repo hygiene follows the course's security lessons: **pre-commit** hooks (ruff, private-key detection) and a **Semgrep CI workflow** (`p/python`, `p/security-audit`) on every push. No keys in code; the demo reads the API key (`GEMINI_API_KEY` / `GOOGLE_API_KEY`) from the environment.

### Course concepts demonstrated

| Concept | Where |
|---|---|
| Multi-agent system (ADK 2.0) | Code — coordinator/researcher/writer team; governor as Runner plugin (`src/governor/adk_plugin.py`, `demo/run_adk_demo.py`) |
| MCP server | Code — `src/governor/mcp_server.py`, the ledger over MCP for cross-runtime governance |
| Security features | Code + video — SKILLSTRIDE threat model, Semgrep CI, pre-commit, and the governor itself as DoS mitigation |
| Agent skills | Code — SKILLSTRIDE (our published skill) applied to this workspace |
| Deployability | Video — MCP server runs standalone; ADK app servable via `adk api_server` / Cloud Run |

### The build

The project was vibe-coded in a day with Claude Code against the installed ADK 2.3 package (callback signatures verified against source, not docs), test-first where it mattered: 13 unit tests pin the properties the whole argument rests on — the naive ledger *provably* races past the budget in a test, the atomic ledger provably cannot, lease invariants hold, cascades self-extinguish, reservations reconcile, appeals are rationed and can never touch the completion reserve. The simulation layer is dependency-free (numpy + asyncio), so every figure in this writeup reproduces from a clean clone in under a minute: `pytest && python sim/simulation.py`.

### Limitations and what's next

The `asyncio` ledger serializes admission within one process; multi-runner fleets need the same protocol on an external atomic store (Redis `INCRBY`) — the MCP server is the stepping stone. Input counting in the offline plugin is heuristic (~4 chars/token); swap in Gemini's `count_tokens` for exactness. The quota split is static; a slow rebalancing loop would recover the utilization gap (fast local control, slow global reallocation — hierarchy done Meadows' way). And the meter experiment begs to be run at statistical scale on real models: *does telling an agent its remaining budget change its behavior?* Our simulated and live-demo evidence says yes — someone should publish the full study.

**Repo:** github.com/RobertoGPAI/overshoot-governor (tests, simulations, executed notebook, threat model, live demo). **Video:** attached.

*Word count: ~2,100 (limit 2,500).*
