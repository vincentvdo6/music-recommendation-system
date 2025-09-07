"""Request timing middleware for performance monitoring."""

import time
from typing import Dict

import structlog
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger()


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request timing and component performance."""
    
    def __init__(self, app):
        super().__init__(app)
        self.latency_budgets = self._get_latency_budgets()
    
    def _get_latency_budgets(self) -> Dict[str, float]:
        """Define latency budgets for different components (in seconds)."""
        return {
            "consent": 0.002,      # 2ms
            "identity": 0.003,     # 3ms
            "embedding_lookup": 0.005,  # 5ms (cache)
            "embedding_compute": 0.035, # 35ms (when needed)
            "ann_search": 0.008,   # 8ms
            "features": 0.010,     # 10ms
            "ranking": 0.010,      # 10ms
            "mmr_flow": 0.005,     # 5ms
            "api_glue": 0.005,     # 5ms
            "total": 0.100,        # 100ms total (warm)
        }
    
    async def dispatch(self, request: Request, call_next):
        """Track request timing and check against budgets."""
        
        start_time = time.time()
        
        # Initialize timing state
        request.state.component_timings = {}
        request.state.start_time = start_time
        
        response = await call_next(request)
        
        total_duration = time.time() - start_time
        
        # Log timing information
        timing_data = {
            "request_id": getattr(request.state, "request_id", "unknown"),
            "total_duration": total_duration,
            "method": request.method,
            "endpoint": request.url.path,
            "component_timings": getattr(request.state, "component_timings", {})
        }
        
        # Check if we exceeded budgets
        budget_violations = self._check_budget_violations(
            total_duration, 
            timing_data["component_timings"]
        )
        
        if budget_violations:
            logger.warning(
                "Latency budget exceeded",
                **timing_data,
                budget_violations=budget_violations
            )
        else:
            logger.info(
                "Request timing",
                **timing_data
            )
        
        # Add timing headers
        response.headers["X-Response-Time"] = str(int(total_duration * 1000))
        
        return response
    
    def _check_budget_violations(
        self, 
        total_duration: float, 
        component_timings: Dict[str, float]
    ) -> Dict[str, Dict[str, float]]:
        """Check which components exceeded their latency budgets."""
        violations = {}
        
        # Check total budget
        total_budget = self.latency_budgets["total"]
        if total_duration > total_budget:
            violations["total"] = {
                "actual": total_duration,
                "budget": total_budget,
                "excess": total_duration - total_budget
            }
        
        # Check component budgets
        for component, actual_time in component_timings.items():
            if component in self.latency_budgets:
                budget = self.latency_budgets[component]
                if actual_time > budget:
                    violations[component] = {
                        "actual": actual_time,
                        "budget": budget,
                        "excess": actual_time - budget
                    }
        
        return violations


def track_component_time(request: Request, component: str, duration: float):
    """Helper function to track component timing."""
    if hasattr(request.state, "component_timings"):
        request.state.component_timings[component] = duration