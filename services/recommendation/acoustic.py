"""Independent acoustic-discovery lane configuration.

MPD playlist-membership labels cannot make extension-catalog tracks positive.
This lane is therefore mixed after collaborative ranking and can be calibrated
from explicit feedback without corrupting the collaborative ranker objective.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

ACOUSTIC_POLICY_FILE = "models/acoustic_policy.json"


@dataclass(frozen=True)
class AcousticPolicy:
    slots: int = 3
    min_cos: float = 0.35
    reciprocal_rank_weight: float = 0.10
    extension_only: bool = True

    @classmethod
    def from_dict(cls, payload: dict) -> "AcousticPolicy":
        policy = cls(**payload)
        if policy.slots < 0:
            raise ValueError("acoustic slots must be >= 0")
        if not -1.0 <= policy.min_cos <= 1.0:
            raise ValueError("acoustic min_cos must be in [-1, 1]")
        if policy.reciprocal_rank_weight < 0:
            raise ValueError("reciprocal_rank_weight must be >= 0")
        return policy


def load_acoustic_policy(path: str = ACOUSTIC_POLICY_FILE) -> AcousticPolicy:
    payload = {}
    artifact = Path(path)
    if artifact.exists():
        payload = json.loads(artifact.read_text())
    policy = AcousticPolicy.from_dict(payload)

    def env_int(name: str, default: int) -> int:
        try:
            return int(os.getenv(name, default))
        except (TypeError, ValueError):
            return default

    def env_float(name: str, default: float) -> float:
        try:
            return float(os.getenv(name, default))
        except (TypeError, ValueError):
            return default

    extension_raw = os.getenv("ONEREC_ACOUSTIC_EXTENSION_ONLY")
    extension_only = policy.extension_only
    if extension_raw is not None:
        extension_only = extension_raw.strip().lower() not in {"0", "false", "no"}
    return AcousticPolicy.from_dict({
        "slots": env_int("ONEREC_DISCOVERY_SLOTS", policy.slots),
        "min_cos": env_float("ONEREC_DISCOVERY_MIN_COS", policy.min_cos),
        "reciprocal_rank_weight": env_float(
            "ONEREC_ACOUSTIC_RR_WEIGHT", policy.reciprocal_rank_weight
        ),
        "extension_only": extension_only,
    })
