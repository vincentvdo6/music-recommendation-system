"""FastAPI application for music recommendation system."""

import os
import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, ORJSONResponse, PlainTextResponse
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded

from api.routers import search
from services.spotify.client import SpotifyClient
from api.models import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("music-rec")

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Music Recommendation API v0.1.0")

    # Initialize Spotify client
    app.state.spotify = SpotifyClient()
    await app.state.spotify.start()

    try:
        yield
    finally:
        # Clean shutdown
        logger.info("Shutting down Music Recommendation API")
        await app.state.spotify.close()


# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Music Recommendation API",
    description="Music recommendation system powered by Spotify Web API",
    version="0.1.0",
    default_response_class=ORJSONResponse,  # Faster JSON serialization
    lifespan=lifespan
)

# Add rate limiting
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)

# Add GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

# Environment and security setup
ENV = os.getenv("ENV", "development")
allowed_origin = os.getenv("ALLOWED_ORIGIN")
if ENV == "production" and not allowed_origin:
    raise RuntimeError("ALLOWED_ORIGIN must be set in production")

# Add CORS middleware with proper security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[allowed_origin] if allowed_origin else ["*"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers and request logging."""
    rid = str(uuid.uuid4())[:8]
    request.state.request_id = rid

    start = time.perf_counter()
    response = await call_next(request)
    dur = (time.perf_counter() - start) * 1000

    logger.info("%s %s %s %d %.2fms", rid, request.method, request.url.path, response.status_code, dur)

    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
    response.headers["X-Request-ID"] = rid

    # Content Security Policy (temporary inline allowed for existing script)
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' https: data: *.scdn.co; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self';"
    )

    return response


# Cached static files serving
class CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp: Response = await super().get_response(path, scope)
        if resp.status_code == 200:
            resp.headers.setdefault("Cache-Control", "public, max-age=86400")
        return resp

# Mount static files with caching
app.mount("/static", CachedStatic(directory="static"), name="static")

# Rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    response = PlainTextResponse("Rate limit exceeded. Please try again later.", status_code=429)
    response.headers["Retry-After"] = str(exc.retry_after or 60)
    return response

# Include routers
app.include_router(search.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=time.time()
    )


@app.get("/")
async def root():
    """Serve the main web interface."""
    return FileResponse('static/index.html')


@app.get("/api")
async def api_info():
    """API information endpoint."""
    return {
        "name": "Music Recommendation API",
        "version": "0.1.0",
        "description": "Music recommendation system powered by Spotify Web API",
        "endpoints": {
            "search": "/api/v1/search",
            "recommendations": "/api/v1/recommendations",
            "health": "/health"
        }
    }
