# Overshoot Governor

**Token-budget admission control for multi-agent systems, designed with Donella
Meadows.** Kaggle capstone — Roberto García Patrón.

A token budget in a multi-agent system has the three ingredients Meadows
identified for *overshoot and collapse*: growth (bursty concurrent calls), a
limit (the budget), and a delay in the feedback about the limit (in-flight
calls whose cost is unknown until they complete). This project builds the
governor that fixes it, as an **ADK 2.0 Runner plugin**, and evaluates it with
three seeded experiments. Full narrative and results:
[`overshoot-governor-capstone.ipynb`](overshoot-governor-capstone.ipynb).

## Findings (seeded simulations, details in the notebook)

1. **Concurrent admission** — a naive check-then-act limit overshoots ~13%
   (races + invisible in-flight calls); atomic reserve/reconcile eliminates
   overshoot; a decentralized lease tree also never overshoots and needs no
   coordination, at the price of utilization. *Exactness costs coordination;
   autonomy costs utilization.*
2. **Subagent spawning** — per-agent caps don't compose: a spawn cascade
   reaches ~360 agents and ~260% overshoot. Lease inheritance (children carve
   quota out of the parent's remainder) turns the exponential cascade
   geometric: ~40 agents, 0% overshoot, self-extinguishing.
3. **The meter in the hallway** — identical enforcement, but agents that can
   *see* the remaining budget complete more tasks with fewer speculative
   calls. Meadows' Amsterdam electricity-meter effect, replicated on agents.

## Layout

```
src/governor/          core: AtomicLedger, OutputEstimator, QuotaNode
src/governor/adk_plugin.py   BudgetGovernorPlugin (ADK 2.x Runner plugin)
sim/simulation.py      the three experiments (python sim/simulation.py)
demo/run_adk_demo.py   live Gemini A/B demo: meter on vs off (needs GOOGLE_API_KEY)
tests/                 9 unit tests (race exposure, atomicity, lease invariants)
security/threat_model.md     STRIDE analysis (SKILLSTRIDE methodology)
build_notebook.py      regenerates the Kaggle notebook from these sources
```

## Quickstart

```bash
pip install -r requirements.txt
python -m pytest tests -q      # 9 tests
python sim/simulation.py       # runs the 3 experiments, saves figures/
python demo/run_adk_demo.py    # live ADK demo (set GOOGLE_API_KEY first)
```

## Security

Unbounded token spend is treated as a security problem (denial-of-wallet,
agent fork bombs, prompt-injection-driven budget drain), analyzed with the
[SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE) STRIDE methodology —
see [security/threat_model.md](security/threat_model.md). Repo hygiene:
[pre-commit](.pre-commit-config.yaml) (ruff, private-key detection) and a
[Semgrep CI workflow](.github/workflows/security.yml) (`p/python`,
`p/security-audit`) on every push.

## References

Meadows, *Thinking in Systems* (2008) · Meadows, *Leverage Points* (1999) ·
Meadows et al., *The Limits to Growth* (1972) ·
[ADK Plugins](https://google.github.io/adk-docs/plugins/) ·
[ADK Agent Skills](https://developers.googleblog.com/developers-guide-to-building-adk-agents-with-skills/)
