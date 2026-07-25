"""Request-level recommendation styles layered over a trained policy.

Profiles are deliberately small post-ranking adjustments. They do not alter
the retrieval or feature contracts that the installed ranker was trained on.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from typing import Dict

from services.recommendation.acoustic import AcousticPolicy
from services.recommendation.policy import ScoringPolicy


@dataclass(frozen=True)
class PreferenceProfile:
    name: str
    seed_affinity_delta: float
    discovery_delta: float
    floor: float | None
    acoustic_slots: int
    session_weight: float
    max_per_artist: int
    era_diversity: bool

    def scoring_policy(self, base: ScoringPolicy) -> ScoringPolicy:
        floor = base.floor if self.floor is None else self.floor
        return replace(
            base,
            seed_affinity=max(0.0, base.seed_affinity + self.seed_affinity_delta),
            discovery=max(0.0, base.discovery + self.discovery_delta),
            floor=max(0.0, min(1.0, floor)),
        )

    def acoustic_policy(self, base: AcousticPolicy) -> AcousticPolicy:
        return replace(base, slots=max(0, self.acoustic_slots))


PROFILES: Dict[str, PreferenceProfile] = {
    "familiar": PreferenceProfile(
        name="familiar",
        seed_affinity_delta=0.35,
        discovery_delta=-0.25,
        floor=0.25,
        acoustic_slots=1,
        session_weight=0.30,
        max_per_artist=2,
        era_diversity=False,
    ),
    "balanced": PreferenceProfile(
        name="balanced",
        seed_affinity_delta=0.0,
        discovery_delta=0.0,
        floor=None,
        acoustic_slots=2,
        session_weight=0.25,
        max_per_artist=1,
        era_diversity=True,
    ),
    "explorer": PreferenceProfile(
        name="explorer",
        seed_affinity_delta=-0.25,
        discovery_delta=0.50,
        floor=0.10,
        acoustic_slots=3,
        session_weight=0.20,
        max_per_artist=1,
        era_diversity=True,
    ),
}

DEFAULT_PROFILE = "familiar"
PROFILE_CONTRACT = hashlib.sha256(
    json.dumps(
        {name: asdict(profile) for name, profile in PROFILES.items()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()[:16]


def get_profile(name: str | None) -> PreferenceProfile:
    """Return a validated profile; unknown names fail closed to closest-match."""
    return PROFILES.get((name or "").strip().lower(), PROFILES[DEFAULT_PROFILE])
