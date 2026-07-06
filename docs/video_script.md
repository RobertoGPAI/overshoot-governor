# Guion del vídeo (≤ 5 min, YouTube)

> Graba pantalla + voz. Narración sugerida en inglés (los jueces evalúan en
> inglés); debajo de cada bloque tienes la idea en español por si prefieres
> locutar en español con subtítulos en inglés. Total apuntado: ~4:55.

---

## 0:00–0:35 — The problem (imagen: factura/dashboard + título)

**EN:** "Multi-agent systems have a money problem. Several autonomous agents
draw from one token budget, each learns the real cost of a call only after it
completes, and any agent can spawn more agents. Fifty years ago Donella
Meadows showed that growth plus a limit plus delayed feedback is the recipe
for overshoot and collapse. A token budget has all three ingredients. I built
the governor that fixes it."

**ES:** El problema: varios agentes, un presupuesto, coste conocido tarde,
spawning. Es la receta exacta del overshoot de Meadows.

*En pantalla: portada (figures/architecture.png título) → los 3 ingredientes.*

## 0:35–1:20 — Why agents + the failure, live (demo 1)

**EN:** "Why is this an agent problem? Because the failure modes are agentic:
concurrent admission races, and spawn cascades. Watch the baseline: a naive
limit that checks spent tokens — the industry-standard billing-dashboard
pattern — overshoots by 13%, because a whole wave of admitted calls is still
in flight when the budget line is crossed. And per-agent caps are worse: a
spawn cascade reaches 364 agents and 259% overshoot. That's a prompt-injected
agent fork bomb aimed at your wallet."

**ES:** Enseña `python sim/simulation.py` corriendo y las figuras 1 y 3.

*En pantalla: terminal con la tabla del Exp.1 → fig1 (curva cruzando el
presupuesto) → fig3 (spawning exponencial vs geométrico).*

## 1:20–2:20 — Architecture (diagrama)

**EN:** "The governor is an ADK 2.0 Runner plugin — registered once, it
intercepts every model call of every agent and every spawned subagent.
Before each call it reserves input tokens — which are countable — plus the
p90 of that agent's observed outputs, atomically, against a shared ledger.
After the call it reconciles with the real usage metadata. So the controlled
quantity is spent *plus committed* — the delay is gone from the loop.
Subagents never get fresh budget: they lease a slice of their parent's
remainder, so cascades decay geometrically and self-extinguish. A completion
reserve guarantees the mission can always afford to land. And the same ledger
is exposed as an MCP server, so Claude Code sessions or CI scripts can share
the same budget — cross-runtime governance."

**ES:** Recorre el diagrama de arquitectura señalando: plugin → ledger
atómico → estimador p90 → árbol de leases → reserva de finalización → MCP.

*En pantalla: figures/architecture.png, señalando cada caja al nombrarla.*

## 2:20–3:20 — The result that matters: the meter in the hallway (demo 2)

**EN:** "One more Meadows idea, and it's my favorite result. Her canonical
example of intervening through information flows: identical Amsterdam houses
used thirty percent less electricity when the meter was in the hallway
instead of the basement. I replicated that structure with agents. Identical
teams, identical enforcement — the only difference is a flag that injects the
live budget state into each agent's context. Blind agents completed 15 tasks
on the budget. Sighted agents completed 20, with half the speculative calls.
Self-restraint funded by information, not enforcement. The meter in the
hallway works for agents too — and in the live demo it's real Gemini calls
through an ADK coordinator, researcher and writer."

**ES:** Muestra fig4 y, si tienes API key, `python demo/run_adk_demo.py` con
los dos reports del ledger (blind vs sighted).

*En pantalla: fig4 → demo en vivo o captura del output del demo.*

## 3:20–4:00 — The right of appeal (el "wow")

**EN:** "One last mechanism, because a wall that can't be contested strands
work: a task denied at its final step wastes everything already spent on it.
So denials carry a right of appeal. Granted appeals draw from a protected
tranche, rationed and logged, never from the completion reserve. They're
heard by a judge agent — deliberately not the coordinator that allocates the
quotas, because nobody should be judge in their own cause — and the hearing
itself is billed to the same ledger: justice isn't free, so appeal spam
exhausts the hearing budget, never the mission's. And running this live
exposed a subtlety I love: a denial *silences* its addressee. The refusal
short-circuits the model call, so the denied agent never actually hears the
verdict — to appeal it would need the very call it was just denied. So the
appeal is filed *ex officio*, the way a public prosecutor speaks for a party
that cannot speak: the driver — or a parent agent — appeals on the mission's
behalf, not the agent's. Watch: the budget bites mid-mission, the appeal is
filed, the judge grants it, and the team lands the mission from the
protected tranche. Same hard cap, zero overshoot, seven times less stranded
work — while spending less. Enforcement, information, and voice: the agent
is governed the way citizens are, not the way resources are."

**ES:** Muestra fig5 y la tabla del Exp. 4; luego el demo en vivo
`python demo/run_adk_demo.py --appeal-demo` — señala la línea
"denial detected; filing an appeal", el `appeal [GRANTED]` y el coste de la
audiencia ("tokens spent on justice").

*En pantalla: fig5_appeals.png → terminal con --appeal-demo (denegación →
apelación → GRANTED → misión aterrizada) → el DENIAL_TEXT en el código.*

## 4:00–4:25 — Security (concepto del curso)

**EN:** "Unbounded token spend is a security problem: denial of service on
the wallet. The repo is threat-modeled with SKILLSTRIDE, our STRIDE skill for
agent workspaces — prompt-injection budget drain and fork bombs map to DoS,
subagents escaping limits map to elevation of privilege, and each has a
structural mitigation in the design. Pre-commit hooks and a Semgrep CI
workflow run on every push. No keys in code."

**ES:** Enseña `security/threat_model.md`, `.pre-commit-config.yaml` y el
workflow de Semgrep; menciona SKILLSTRIDE con tu repo en pantalla.

*En pantalla: threat_model.md scrolleando → workflow security.yml → (badge CI
si ya hiciste push).*

## 4:25–4:55 — The build + deployability + close

**EN:** "Thirteen unit tests pin the claims — the naive ledger provably races
in a test; the atomic one provably can't. Everything reproduces from a clean
clone in under a minute. The MCP server runs standalone today, and the ADK
app deploys with adk api_server or Cloud Run. It's about 200 lines of core
Python — because the hard part wasn't code. It was seeing the budget as what
it is: a stock, with flows, and a delay. Fix the system, not the symptom.
Thanks."

**ES:** Cierra con pytest en verde + el README, y la frase final sobre
Meadows.

*En pantalla: `pytest` (11 passed) → README → diapositiva final con el repo.*

---

## Checklist de grabación

- [ ] 1080p, audio limpio, sin API keys visibles en pantalla (¡ni en el env!)
- [ ] Subir a YouTube (público o unlisted), adjuntar a la Media Gallery
- [ ] Portada del writeup: `figures/architecture.png`
- [ ] Si usaste Antigravity durante el desarrollo, inserta 10–15 s de esa
      pantalla en el bloque 4:00 y di "built with Antigravity/Claude Code" —
      cuenta como concepto extra demostrado en vídeo
