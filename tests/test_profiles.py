"""Preference profiles and session-facing policy adjustments."""

from services.recommendation.acoustic import AcousticPolicy
from services.recommendation.policy import ScoringPolicy
from services.recommendation.profiles import get_profile


def test_profiles_move_policy_in_documented_directions():
    base = ScoringPolicy(seed_affinity=1.0, discovery=0.5, floor=0.2)
    familiar = get_profile("familiar").scoring_policy(base)
    explorer = get_profile("explorer").scoring_policy(base)
    assert familiar.seed_affinity > base.seed_affinity > explorer.seed_affinity
    assert familiar.discovery < base.discovery < explorer.discovery
    assert familiar.floor > explorer.floor


def test_balanced_profile_preserves_trained_policy_and_limits_discovery():
    base = ScoringPolicy(seed_affinity=1.5, discovery=1.0, floor=0.0, scale="fixed")
    profile = get_profile("balanced")
    assert profile.scoring_policy(base) == base
    assert profile.acoustic_policy(AcousticPolicy(slots=3)).slots == 2


def test_closest_profile_relaxes_presentation_diversity():
    profile = get_profile("familiar")
    assert profile.max_per_artist == 2
    assert profile.era_diversity is False


def test_unknown_profile_fails_closed_to_closest_match():
    assert get_profile("not-a-profile").name == "familiar"
