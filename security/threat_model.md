# Threat Model — Overshoot Governor

> Produced following the [SKILLSTRIDE](https://github.com/RobertoGPAI/SKILLSTRIDE)
> methodology (public version: boundary mapping → infrastructure STRIDE →
> LLM-specific assessment → risk assessment). To be regenerated/validated with
> the enhanced SKILLSTRIDE Python tooling before final submission.

## Phase 1 — Boundary mapping

| Boundary | Description |
|---|---|
| B1: User ↔ Coordinator agent | Untrusted natural-language task input |
| B2: Agent ↔ LLM (Gemini API) | Every model call; interception point of the governor plugin |
| B3: Agent ↔ Agent (delegation/spawning) | Sub_agents transfer and quota leasing |
| B4: Agent ↔ Tools | Tool results re-enter model context (indirect injection surface) |
| B5: Runner ↔ Ledger state | In-process shared state (`AtomicLedger`, quota tree) |
| B6: Repo ↔ CI | pre-commit hooks, Semgrep workflow, secrets hygiene |

Assets: the token budget (financial), the mission outcome (availability of the
*result*), API credentials, ledger integrity.

## Phase 2 — Infrastructure STRIDE

| Threat | Scenario | Risk | Mitigation |
|---|---|---|---|
| **S**poofing | A component posing as the governor admits calls | Low | Plugin registered once on the Runner; no agent-supplied registration path |
| **T**ampering | Agent-generated content alters ledger state | Medium | Ledger lives outside model context; agents get read-only visibility text; only `usage_metadata` (API-provided) updates spend |
| **R**epudiation | No attribution of who drained the budget | Medium | Per-agent reservations keyed by invocation id; settle-time reconciliation log |
| **I**nformation disclosure | Budget telemetry leaks sensitive totals to agents | Low | Visibility string carries only aggregate counters; no per-agent breakdown, no credentials |
| **D**enial of service | Budget exhaustion halts the system mid-mission | High | Atomic admission ledger (hard ceiling) + completion reserve (mission can always land) |
| **E**levation of privilege | Subagent operates outside its parent's limits | High | Lease invariant `Σ children ≤ parent`; Runner-level plugin covers spawned agents automatically |

## Phase 3 — LLM-specific assessment

| Vector | Scenario | Risk | Mitigation |
|---|---|---|---|
| Direct prompt injection | User input instructs an agent to loop expensive calls until funds are gone (budget DoS) | High | Admission control is out-of-band: no instruction can raise the ceiling |
| Indirect prompt injection | Tool/web content triggers speculative call storms or spawn cascades | High | Lease-inherited quotas make cascades geometrically self-extinguishing ("agent fork bomb" mitigation) |
| Function/tool abuse | Repeated tool invocations amplify context size and cost per call | Medium | Input side counted per request; growth shows up in reservations immediately |
| Governor-text spoofing | Malicious content imitates `[BUDGET GOVERNOR]` messages to manipulate agents | Medium | Enforcement never depends on the text; visibility is advisory only. Residual: agent behavior may be nudged — accepted, since hard limits hold |
| Denial-of-wallet via estimator poisoning | Adversarial tasks inflate p90 history to starve admission | Low | Estimates are capped by `max_output_tokens`; worst case degrades to worst-case reservation, never past the budget |
| Appeal-channel abuse | Injected agent spams plausible-sounding appeals to drain the appeal tranche | Medium | Appeals rationed per agent; tranche bounded and never touches the completion reserve; every appeal logged with its justification (audit trail); optional `judge` hook for arbitration by a coordinator |

## Phase 4 — Risk assessment summary

- **High risks addressed by design:** budget DoS (atomic ledger), spawn
  cascade (lease tree), mission starvation (completion reserve), subagent
  privilege escape (Runner-level plugin + lease invariant).
- **Medium, monitored:** governor-text spoofing (advisory channel), tampering
  via crafted content (state isolation), repudiation (reservation log).
- **Accepted/residual:** single-process ledger assumes a trusted Runner host;
  distributed deployment requires an external atomic store (future work).

Repo hygiene: pre-commit (ruff, whitespace, merge-conflict, private-key
detection) and Semgrep CI (`p/python`, `p/security-audit`) on every push.
