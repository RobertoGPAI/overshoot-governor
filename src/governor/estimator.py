"""Pre-call cost estimation.

Input tokens are deterministic (countable before the call); output tokens are
not. The estimator closes that gap empirically: it keeps a rolling history of
actual output token counts per (agent, task-type) key, fed from settle-time
usage metadata, and predicts a high quantile of that history. Reserving the
p90 instead of the configured worst case (max_output_tokens) trades a small,
quantifiable risk of underestimation for much higher admission throughput.
"""

from __future__ import annotations

from collections import defaultdict, deque


class OutputEstimator:
    """Rolling per-key quantile estimator for output tokens."""

    def __init__(
        self,
        prior: int = 1024,
        quantile: float = 0.90,
        window: int = 50,
        min_samples: int = 5,
    ) -> None:
        self.prior = prior
        self.quantile = quantile
        self.min_samples = min_samples
        self._history: dict[str, deque[int]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def predict(self, key: str) -> int:
        """Estimated output tokens for the next call under `key`."""
        samples = self._history.get(key)
        if not samples or len(samples) < self.min_samples:
            return self.prior
        ordered = sorted(samples)
        idx = min(len(ordered) - 1, int(self.quantile * len(ordered)))
        return ordered[idx]

    def update(self, key: str, actual: int) -> None:
        """Feed the actual output token count observed at settle time."""
        self._history[key].append(actual)


class InputCalibrator:
    """Rolling correction factor for the chars-per-token input heuristic.

    A constant like chars//4 embeds a language and a tokenizer (English,
    ~4 chars/token); real tokenizers differ in both directions -- Spanish
    prose runs ~3.5, Llama-family tokenizers on structured text ~6. Both
    errors are harmful: undercounting risks overshoot, overcounting closes
    the landing window early (observed live: a mission force-landed on
    turn 2 with 93% of its budget unspent). At settle time the provider
    reports the true prompt token count; the calibrator learns
    actual/estimated per key, and the estimate stops assuming.
    """

    def __init__(
        self, window: int = 50, min_samples: int = 2, min_input: int = 512
    ) -> None:
        self.min_samples = min_samples
        # Small requests are dominated by fixed overhead the heuristic never
        # sees (chat template, tool declarations, instructions appended after
        # estimation): their ratios measure the takeoff, not the cruise.
        # Observed live: a 186-token estimate against a 585-token prompt
        # taught the calibrator 3.14x and the very next admission was denied
        # at triple its true cost. Only slope-dominated samples train.
        self.min_input = min_input
        self._ratios: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=window)
        )

    def factor(self, key: str) -> float:
        samples = self._ratios.get(key)
        if not samples or len(samples) < self.min_samples:
            return 1.0  # trust the heuristic until evidence arrives
        ordered = sorted(samples)
        mid = len(ordered) // 2
        if len(ordered) % 2:
            return ordered[mid]
        # An even count averages the middle pair -- taking ordered[mid]
        # crowns the larger of two, and with two samples that IS the max.
        return (ordered[mid - 1] + ordered[mid]) / 2

    def update(self, key: str, estimated: int, actual: int) -> None:
        if estimated >= self.min_input and actual > 0:
            self._ratios[key].append(actual / estimated)
