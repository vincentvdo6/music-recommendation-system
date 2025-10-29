"""
A/B testing framework for comparing engines.

Routes traffic between baseline and Phase 1 engines.
"""

import hashlib
import logging
from typing import Literal

logger = logging.getLogger(__name__)


class ABTest:
    """Simple A/B test router."""

    def __init__(
        self,
        baseline_engine,
        phase1_engine,
        phase1_traffic: float = 0.5,
    ):
        """
        Initialize A/B test.

        Args:
            baseline_engine: Original contextual engine
            phase1_engine: New hybrid engine
            phase1_traffic: Fraction of traffic to route to Phase 1 (0.0-1.0)
        """
        self.baseline = baseline_engine
        self.phase1 = phase1_engine
        self.phase1_traffic = phase1_traffic

    def get_variant(self, user_id: str) -> Literal["baseline", "phase1"]:
        """
        Determine which variant to show user.

        Uses consistent hashing so same user always sees same variant.
        """
        # Hash user ID to get deterministic assignment
        hash_value = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
        threshold = hash_value / (2 ** 128)  # Normalize to 0-1

        if threshold < self.phase1_traffic:
            return "phase1"
        else:
            return "baseline"

    async def get_recommendations(
        self,
        user_id: str,
        **kwargs
    ):
        """
        Get recommendations from assigned variant.

        Logs variant assignment for analysis.
        """
        variant = self.get_variant(user_id)

        logger.info(f"User {user_id[:8]} assigned to variant: {variant}")

        if variant == "phase1":
            engine = self.phase1
        else:
            engine = self.baseline

        # Get recommendations
        recs = await engine.get_recommendations(**kwargs)

        # Add variant metadata
        for rec in recs:
            if isinstance(rec, dict):
                rec["_variant"] = variant

        return recs, variant


# Metrics tracking
class ABMetrics:
    """Track A/B test metrics."""

    def __init__(self):
        self.metrics = {
            "baseline": {
                "requests": 0,
                "clicks": 0,
                "avg_rank_clicked": 0,
                "latency_ms": [],
            },
            "phase1": {
                "requests": 0,
                "clicks": 0,
                "avg_rank_clicked": 0,
                "latency_ms": [],
            }
        }

    def record_request(self, variant: str, latency_ms: float):
        """Record a recommendation request."""
        self.metrics[variant]["requests"] += 1
        self.metrics[variant]["latency_ms"].append(latency_ms)

    def record_click(self, variant: str, rank: int):
        """Record a user click on a recommendation."""
        m = self.metrics[variant]
        m["clicks"] += 1

        # Update average rank clicked
        n = m["clicks"]
        m["avg_rank_clicked"] = (m["avg_rank_clicked"] * (n - 1) + rank) / n

    def get_summary(self):
        """Get summary statistics."""
        import numpy as np

        summary = {}
        for variant in ["baseline", "phase1"]:
            m = self.metrics[variant]

            ctr = m["clicks"] / m["requests"] if m["requests"] > 0 else 0
            avg_latency = np.mean(m["latency_ms"]) if m["latency_ms"] else 0
            p50_latency = np.percentile(m["latency_ms"], 50) if m["latency_ms"] else 0
            p95_latency = np.percentile(m["latency_ms"], 95) if m["latency_ms"] else 0

            summary[variant] = {
                "requests": m["requests"],
                "clicks": m["clicks"],
                "ctr": round(ctr, 4),
                "avg_rank_clicked": round(m["avg_rank_clicked"], 2),
                "latency": {
                    "avg_ms": round(avg_latency, 1),
                    "p50_ms": round(p50_latency, 1),
                    "p95_ms": round(p95_latency, 1),
                }
            }

        # Statistical significance (simple z-test)
        n_baseline = summary["baseline"]["requests"]
        n_phase1 = summary["phase1"]["requests"]

        if n_baseline > 30 and n_phase1 > 30:
            ctr_baseline = summary["baseline"]["ctr"]
            ctr_phase1 = summary["phase1"]["ctr"]

            # Pooled proportion
            p_pool = (
                (summary["baseline"]["clicks"] + summary["phase1"]["clicks"]) /
                (n_baseline + n_phase1)
            )

            # Standard error
            se = (p_pool * (1 - p_pool) * (1/n_baseline + 1/n_phase1)) ** 0.5

            # Z-score
            z = (ctr_phase1 - ctr_baseline) / se if se > 0 else 0

            # P-value (two-tailed)
            import scipy.stats as stats
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))

            summary["significance"] = {
                "z_score": round(z, 2),
                "p_value": round(p_value, 4),
                "significant": p_value < 0.05,
                "lift": round((ctr_phase1 - ctr_baseline) / ctr_baseline * 100, 1) if ctr_baseline > 0 else 0,
            }

        return summary
