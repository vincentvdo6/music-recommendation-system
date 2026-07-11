"""
The deployed scoring policy — shared VERBATIM by serving and evaluation.

Everything between raw model scores and the final candidate order lives here,
so scripts/evaluate.py measures exactly what engine.recommend() ships:
query-local standardization, the seed-affinity and discovery dials, and the
eligible-first seed-cosine floor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Defaults from the validation sweep in standardized units; env-overridable
# via SEED_AFFINITY / DISCOVERY (see factory.py).
DEFAULT_SEED_AFFINITY = 2.2
DEFAULT_DISCOVERY = 1.5
SEED_COS_FLOOR = 0.25

POLICY_FILE = "models/policy.json"


def load_policy(path: str = POLICY_FILE, env: dict | None = None) -> "ScoringPolicy":
    """Validation-frozen policy from the installed artifact, then env overrides.

    Precedence: env var > models/policy.json > module defaults — the training
    notebook freezes (lam, mu, floor) on validation; SEED_AFFINITY/DISCOVERY/
    SEED_COS_FLOOR env vars remain the user-facing dials.
    """
    import json
    import os
    from pathlib import Path

    env = os.environ if env is None else env
    lam, mu, floor = DEFAULT_SEED_AFFINITY, DEFAULT_DISCOVERY, SEED_COS_FLOOR
    p = Path(path)
    if p.exists():
        try:
            frozen = json.loads(p.read_text())
            lam = float(frozen.get("lam", lam))
            mu = float(frozen.get("mu", mu))
            floor = float(frozen.get("floor", floor))
        except (ValueError, OSError):
            pass

    def env_float(name: str, default: float) -> float:
        try:
            return float(env.get(name, default))
        except (TypeError, ValueError):
            return default

    return ScoringPolicy(
        seed_affinity=env_float("SEED_AFFINITY", lam),
        discovery=env_float("DISCOVERY", mu),
        floor=env_float("SEED_COS_FLOOR", floor),
    )


@dataclass(frozen=True)
class ScoringPolicy:
    seed_affinity: float = DEFAULT_SEED_AFFINITY
    discovery: float = DEFAULT_DISCOVERY
    floor: float = SEED_COS_FLOOR

    def blend(
        self,
        scores: np.ndarray,
        seed_cos: np.ndarray,
        log_pop: np.ndarray,
        has_seed: bool,
    ) -> np.ndarray:
        """zscore(model) + λ·seed_cos − μ·log_pop (standardization makes the
        dials mean the same thing for LightGBM and the 0-1 linear fallback)."""
        blended = np.asarray(scores, dtype=np.float64)
        if self.seed_affinity or self.discovery:
            blended = (blended - blended.mean()) / (blended.std() or 1.0)
        if has_seed and self.seed_affinity:
            blended = blended + self.seed_affinity * seed_cos
        if self.discovery:
            blended = blended - self.discovery * log_pop
        return blended

    def select(
        self,
        blended: np.ndarray,
        seed_cos: np.ndarray,
        limit: int,
        has_seed: bool,
    ) -> np.ndarray:
        """Score order with an eligible-first floor: candidates at least
        `floor`-similar to the seed rank first; ineligible candidates fill any
        remaining slots (never the old all-or-nothing fallback)."""
        order = np.argsort(-blended)
        if not has_seed or not self.floor:
            return order[:limit]
        eligible = order[seed_cos[order] >= self.floor]
        if len(eligible) >= limit:
            return eligible[:limit]
        ineligible = order[seed_cos[order] < self.floor]
        return np.concatenate([eligible, ineligible])[:limit]
