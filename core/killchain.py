"""Kill-chain transition prior for attack forecasting.

Attacks progress through MITRE-style stages in a largely ordered way
(Reconnaissance -> Initial Access -> Execution -> ... -> Exfiltration/Impact).
This module learns P(next_stage | current_stage) from chronological sessions
and combines it with the LSTM world model's prediction:

    P_hybrid(s) ∝ P_lstm(s)^alpha · P_killchain(s | current)^(1 - alpha)

(a product-of-experts / geometric blend). The LSTM sees the detailed traffic
dynamics; the prior contributes structural knowledge of how intrusions unfold.
That structure carries over to sessions and attack progressions the network
model has never seen.

Smoothing has two parts, so unseen transitions are never impossible:
  * additive (Laplace) smoothing over all stages
  * a kill-chain ordering bonus: moving forward by 1-2 stages, or staying in
    the current stage, gets extra pseudo-counts
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

KILL_CHAIN = [
    "Normal", "Suspicious", "Reconnaissance", "Initial Access", "Execution",
    "Persistence", "Lateral Movement", "Command & Control", "Exfiltration", "Impact",
]
_POS = {s: i for i, s in enumerate(KILL_CHAIN)}


class KillChainPrior:
    def __init__(self, laplace: float = 0.5, order_bonus: float = 1.5):
        self.laplace = laplace
        self.order_bonus = order_bonus
        self.counts: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    def fit(self, transitions: Iterable[Tuple[str, str]]) -> "KillChainPrior":
        self.counts = defaultdict(lambda: defaultdict(float))
        for cur, nxt in transitions:
            self.counts[str(cur)][str(nxt)] += 1.0
        return self

    def _structural(self, cur: str, nxt: str) -> float:
        a, b = _POS.get(cur), _POS.get(nxt)
        if a is None or b is None:
            return 0.0
        step = b - a
        if step == 0:
            return self.order_bonus           # attackers often stay in a stage
        if 1 <= step <= 2:
            return self.order_bonus           # advance along the chain
        if step > 2:
            return self.order_bonus * 0.4     # skip ahead (less common)
        return self.order_bonus * 0.1         # regress (rare)

    def proba(self, cur: str, classes: List[str]) -> np.ndarray:
        row = self.counts.get(cur, {})
        raw = np.array([row.get(c, 0.0) + self.laplace + self._structural(cur, c) for c in classes])
        return raw / raw.sum()

    def to_dict(self) -> dict:
        return {"laplace": self.laplace, "order_bonus": self.order_bonus,
                "counts": {k: dict(v) for k, v in self.counts.items()}}

    @classmethod
    def from_dict(cls, d: dict) -> "KillChainPrior":
        kp = cls(d.get("laplace", 0.5), d.get("order_bonus", 1.5))
        for cur, row in (d.get("counts") or {}).items():
            for nxt, c in row.items():
                kp.counts[cur][nxt] = float(c)
        return kp


def blend(p_model: np.ndarray, p_prior: np.ndarray, alpha: float) -> np.ndarray:
    """Geometric (product-of-experts) blend; alpha=1 -> model only, 0 -> prior only."""
    eps = 1e-6
    logp = alpha * np.log(p_model + eps) + (1 - alpha) * np.log(p_prior + eps)
    p = np.exp(logp - logp.max())
    return p / p.sum()


def gated_blend(p_model: np.ndarray, p_prior: np.ndarray, alpha_confident: float,
                alpha_uncertain: float, tau: float) -> np.ndarray:
    """Confidence-gated hybrid.

    When the network model is confident (max prob >= tau) the situation resembles
    its training data, so trust it more (alpha_confident). When it is uncertain,
    which is typical of a never-seen campaign, lean on kill-chain structure
    (alpha_uncertain)."""
    alpha = alpha_confident if float(np.max(p_model)) >= tau else alpha_uncertain
    return blend(p_model, p_prior, alpha)


PRIOR_PATH = Path(__file__).resolve().parents[1] / "models" / "temporal" / "killchain_prior.json"
DEFAULT_GATE = (0.8, 0.2, 0.6)
_cache: Optional[tuple] = None


def save_prior(prior: KillChainPrior, gate: Tuple[float, float, float], extra: dict | None = None,
               path: Path = PRIOR_PATH) -> None:
    global _cache
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prior": prior.to_dict(), "gate": list(gate),
                                "gate_fields": ["alpha_confident", "alpha_uncertain", "tau"],
                                **(extra or {})}, indent=2), encoding="utf-8")
    _cache = None


def load_prior(path: Path = PRIOR_PATH) -> Optional[tuple]:
    """Return (KillChainPrior, gate) or None if the hybrid has not been trained."""
    global _cache
    if _cache is not None:
        return _cache
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        _cache = (KillChainPrior.from_dict(d["prior"]), tuple(d.get("gate") or DEFAULT_GATE))
    except Exception:
        return None
    return _cache
