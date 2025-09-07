"""FastAPI application for music recommendation system."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, Any

import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest

from api.middleware.auth import JWTAuthMiddleware
from api.middleware.timing import TimingMiddleware
from api.routers import analyze, recommend
from services.normalize.config import get_config

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status_code'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ANALYZE_DURATION = Histogram('analyze_request_duration_seconds', 'Audio analysis duration')
RECOMMEND_DURATION = Histogram('recommend_request_duration_seconds', 'Recommendation duration')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    config = get_config()
    
    logger.info(
        "Starting music recommendation API",
        version="0.1.0",
        config_loaded=True
    )
    
    # Initialize services here if needed
    # await init_services()
    
    yield
    
    logger.info("Shutting down music recommendation API")


app = FastAPI(
    title="Music Recommendation API",
    description="Neural music recommendation system with embeddings and learned ranking",
    version="0.1.0",
    lifespan=lifespan
)

config = get_config()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(JWTAuthMiddleware)
app.add_middleware(TimingMiddleware)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all requests with structured logging."""
    request_id = str(uuid.uuid4())
    
    # Add request ID to request state
    request.state.request_id = request_id
    
    start_time = time.time()
    
    logger.info(
        "Request started",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        user_agent=request.headers.get("user-agent"),
        ip=request.client.host if request.client else None
    )
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    
    # Update Prometheus metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code
    ).inc()
    REQUEST_DURATION.observe(duration)
    
    logger.info(
        "Request completed",
        request_id=request_id,
        method=request.method,
        url=str(request.url),
        status_code=response.status_code,
        duration_seconds=duration
    )
    
    # Add request ID to response headers
    response.headers["X-Request-ID"] = request_id
    
    return response


# Include routers
app.include_router(analyze.router, prefix="/api/v1")
app.include_router(recommend.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": time.time()
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Music Recommendation API",
        "version": "0.1.0",
        "description": "Neural music recommendation system with embeddings and learned ranking",
        "endpoints": {
            "analyze": "/api/v1/analyze",
            "recommend": "/api/v1/recommend",
            "health": "/health",
            "metrics": "/metrics"
        }
    }