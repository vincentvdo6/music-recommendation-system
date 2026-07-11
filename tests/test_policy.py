"""ScoringPolicy: the single source of truth for serving/evaluation parity."""

import numpy as np

from services.recommendation.policy import ScoringPolicy


def test_zscore_alone_is_rank_stable():
    policy = ScoringPolicy(seed_affinity=0.0, discovery=0.0, floor=0.0)
    scores = np.array([3.0, -1.0, 0.5, 2.0])
    zeros = np.zeros(4)
    blended = policy.blend(scores, zeros, zeros, has_seed=True)
    assert (np.argsort(-blended) == np.argsort(-scores)).all()


def test_dials_move_scores_in_documented_directions():
    policy = ScoringPolicy(seed_affinity=1.0, discovery=1.0, floor=0.0)
    scores = np.array([1.0, 1.0])
    seed_cos = np.array([0.9, 0.1])
    log_pop = np.array([0.1, 0.9])
    blended = policy.blend(scores, seed_cos, log_pop, has_seed=True)
    assert blended[0] > blended[1]  # seed-similar unpopular beats dissimilar popular


def test_floor_is_eligible_first_then_fills():
    policy = ScoringPolicy(seed_affinity=0.0, discovery=0.0, floor=0.5)
    blended = np.array([4.0, 3.0, 2.0, 1.0])
    seed_cos = np.array([0.1, 0.9, 0.1, 0.9])
    # Only two candidates clear the floor but limit is 3: eligible first
    # (by score), then the best ineligible fills the last slot.
    order = policy.select(blended, seed_cos, limit=3, has_seed=True)
    assert order.tolist() == [1, 3, 0]


def test_floor_ignored_without_seed():
    policy = ScoringPolicy(seed_affinity=0.0, discovery=0.0, floor=0.5)
    blended = np.array([1.0, 2.0, 3.0])
    seed_cos = np.zeros(3)
    order = policy.select(blended, seed_cos, limit=2, has_seed=False)
    assert order.tolist() == [2, 1]


def test_engine_and_evaluator_share_the_policy():
    """Parity guard: both must construct orders through ScoringPolicy."""
    import inspect

    from scripts import evaluate
    from services.recommendation import engine

    assert "self.policy.blend" in inspect.getsource(engine.RecommendationEngine.recommend)
    assert "self.policy.select" in inspect.getsource(engine.RecommendationEngine.recommend)
    assert "POLICY.blend" in inspect.getsource(evaluate.served_order)
    assert "POLICY.select" in inspect.getsource(evaluate.served_order)
