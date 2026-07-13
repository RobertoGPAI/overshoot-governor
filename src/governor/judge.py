"""MissionJudge: an arbiter agent for appeals -- judgment as a budgeted call.

Two design decisions carry the weight here:

1. **The judge is not the coordinator.** The coordinator allocates quotas; if
   it also ruled on appeals against the scarcity it created, it would be judge
   in its own cause (nemo iudex in causa sua). The arbiter is a separate,
   minimal model call whose only mandate is: does this justification actually
   serve the mission?

2. **Justice is not free.** Every hearing reserves and settles tokens on the
   SAME ledger it arbitrates (priority tranche, like the appeals it hears).
   If the system cannot afford the hearing, the appeal is refused outright --
   which also caps the blast radius of appeal spam: an attacker flooding the
   channel exhausts the hearing budget, not the mission's.

Offline behavior: with no model caller available (no API key), the judge
abstains and defers to the desk's mechanical policy (justification present +
ration). That keeps tests and simulations deterministic; the LLM ruling is an
upgrade, not a dependency.

Provider resolution (any API, local included): set GOVERNOR_JUDGE_BASE_URL to
put justice on any OpenAI-compatible endpoint -- OpenAI itself, gateways
(LiteLLM proxy, OpenRouter), or local servers (Ollama, LM Studio, vLLM,
llama.cpp) -- with GOVERNOR_JUDGE_MODEL naming the model and
GOVERNOR_JUDGE_API_KEY if the endpoint wants one. Without a base URL the
google.genai path applies as before; without either, the judge abstains.
"""

from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable

from .ledger import AtomicLedger

# caller(prompt) -> (verdict_text, total_tokens_used)
Caller = Callable[[str], Awaitable[tuple[str, int]]]

RULING_PROMPT = """You are the arbiter of a token-budget governor for a \
multi-agent system. The overall mission is:

  {mission}

Agent '{agent}' was denied a model call (estimated {estimate} tokens) and \
appeals with this justification:

  "{justification}"

Grant the appeal ONLY if completing this specific call plausibly protects or \
advances the mission more than saving the tokens would. Spending already \
incurred is not a reason (sunk cost). Answer with exactly one word: \
GRANT or REFUSE."""


def genai_caller(model: str | None = None) -> Caller | None:
    """Default caller backed by google.genai; None when no API key is set.

    The hearing is a plain text prompt -- no tools, no system instruction --
    so even models without function-calling support (e.g. Gemma) can serve
    as arbiter. Override via GOVERNOR_JUDGE_MODEL to put justice on a small,
    frugal model while the governed agents run on a capable one.
    """
    # google-genai reads either name; AI Studio tutorials use GEMINI_API_KEY.
    if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
        return None
    model = model or os.environ.get("GOVERNOR_JUDGE_MODEL", "gemini-2.5-flash")
    from google import genai
    from google.genai import errors as genai_errors

    client = genai.Client()

    async def call(prompt: str) -> tuple[str, int]:
        # A transient 5xx must not decide a case: the hearing is one cheap
        # request, so retry briefly before failing closed as usual.
        for attempt in (1, 2, 3):
            try:
                response = await client.aio.models.generate_content(
                    model=model, contents=prompt
                )
                break
            except genai_errors.ServerError:
                if attempt == 3:
                    raise
                await asyncio.sleep(attempt)
        used = 0
        if response.usage_metadata and response.usage_metadata.total_token_count:
            used = response.usage_metadata.total_token_count
        return (response.text or ""), used

    return call


def openai_compatible_caller(
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> Caller | None:
    """Caller for any OpenAI-compatible /chat/completions endpoint; None when
    no base URL is configured.

    Covers OpenAI itself, gateways (LiteLLM proxy, OpenRouter) and local
    servers (Ollama, LM Studio, vLLM, llama.cpp) with one code path.
    Stdlib-only on purpose: the governor stays dependency-free.
    """
    base_url = base_url or os.environ.get("GOVERNOR_JUDGE_BASE_URL")
    if not base_url:
        return None
    from urllib.parse import urlparse

    scheme = urlparse(base_url).scheme
    if scheme not in ("http", "https"):
        # urllib would happily fetch file:// and friends; the judge endpoint
        # is operator configuration, but configuration can be wrong too.
        raise ValueError(
            f"GOVERNOR_JUDGE_BASE_URL must be http(s), got {scheme!r}."
        )
    model = model or os.environ.get("GOVERNOR_JUDGE_MODEL")
    if not model:
        raise ValueError(
            "GOVERNOR_JUDGE_BASE_URL is set but no judge model is named: "
            "set GOVERNOR_JUDGE_MODEL (there is no sane default across "
            "OpenAI-compatible servers)."
        )
    api_key = api_key if api_key is not None else os.environ.get("GOVERNOR_JUDGE_API_KEY", "")

    import json
    import urllib.error
    import urllib.request

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(prompt: str) -> tuple[str, int]:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers)
        # Scheme locked to http(s) at construction; the URL is operator
        # configuration (env var), never model- or user-controlled input.
        # nosemgrep: python.lang.security.audit.dynamic-urllib-use-detected.dynamic-urllib-use-detected
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = json.load(response)
        choices = payload.get("choices") or [{}]
        text = (choices[0].get("message") or {}).get("content") or ""
        usage = payload.get("usage") or {}
        return text, int(usage.get("total_tokens") or 0)

    async def call(prompt: str) -> tuple[str, int]:
        # Same policy as the genai path: a transient 5xx must not decide a
        # case; client errors (4xx) fail immediately and the hearing fails
        # closed.
        for attempt in (1, 2, 3):
            try:
                return await asyncio.to_thread(_post, prompt)
            except urllib.error.HTTPError as exc:
                if exc.code < 500 or attempt == 3:
                    raise
                await asyncio.sleep(attempt)
        raise RuntimeError("unreachable")

    return call


def default_caller(model: str | None = None) -> Caller | None:
    """Provider resolution for the judge: an OpenAI-compatible endpoint when
    GOVERNOR_JUDGE_BASE_URL is set, else google.genai when a Gemini key
    exists, else None (the judge abstains)."""
    return openai_compatible_caller(model=model) or genai_caller(model=model)


class MissionJudge:
    """Arbiter for AppealsDesk: plug ``.rule`` into the desk's judge hook."""

    def __init__(
        self,
        mission: str,
        ledger: AtomicLedger,
        caller: Caller | None = None,
        hearing_estimate: int = 400,
    ) -> None:
        self.mission = mission
        self.ledger = ledger
        self.caller = caller if caller is not None else default_caller()
        self.hearing_estimate = hearing_estimate
        self.hearings = 0
        self.hearing_tokens = 0
        self.hearing_failures = 0
        self.last_error: Exception | None = None

    async def rule(self, agent: str, estimate: int, justification: str) -> bool:
        if self.caller is None:
            return True  # abstain: the desk's mechanical policy decides

        # The hearing itself must be admitted -- from the priority tranche,
        # like the appeal it examines. No budget for justice, no appeal.
        reservation = await self.ledger.try_reserve(
            self.hearing_estimate, priority=True
        )
        if reservation is None:
            return False

        try:
            verdict, used = await self.caller(
                RULING_PROMPT.format(
                    mission=self.mission,
                    agent=agent,
                    estimate=estimate,
                    justification=justification,
                )
            )
        except Exception as exc:
            self.hearing_failures += 1
            self.last_error = exc
            await self.ledger.cancel(reservation)
            return False  # a hearing that fails grants nothing

        await self.ledger.settle(reservation, used)
        self.hearings += 1
        self.hearing_tokens += used
        return verdict.strip().upper().startswith("GRANT")
