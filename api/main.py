"""FastAPI application for music recommendation system."""

import logging
import logging.config
import os
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, ORJSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from api.models import HealthResponse
from api.routers import search
from services.music.service import MusicService
from services.spotify.client import SpotifyClient
from services.storage import LocalStore

# Load .env before reading any configuration values in this module.
load_dotenv()

# Configure logging with a clean, compact format
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
API_VERSION = "0.1.0"

LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            "datefmt": "%H:%M:%S",
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
        }
    },
    "root": {
        "level": LOG_LEVEL,
        "handlers": ["console"],
    },
}

logging.config.dictConfig(LOG_CONFIG)

NOISY_LOGGERS = {
    "uvicorn": logging.WARNING,
    "uvicorn.error": logging.WARNING,
    "uvicorn.access": logging.WARNING,
    "uvicorn.asgi": logging.WARNING,
    "watchfiles": logging.WARNING,
    "httpx": logging.WARNING,
}

for name, level in NOISY_LOGGERS.items():
    logging.getLogger(name).setLevel(level)

logger = logging.getLogger("music-rec")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    logger.info("Starting Music Recommendation API v%s", API_VERSION)

    # Initialize Spotify client
    spotify_client = SpotifyClient()
    store_path = os.getenv("ONEREC_STORE_PATH", "data/one_rec.sqlite3").strip()
    feedback_store = None
    if store_path:
        try:
            feedback_store = LocalStore(store_path)
        except Exception as exc:
            # Feedback is useful but must never make recommendation serving
            # unavailable (read-only filesystems are common in deployments).
            logger.warning("Local feedback store unavailable (%s) — continuing without it", exc)
    music_service = MusicService(spotify=spotify_client, store=feedback_store)
    await music_service.start()

    app.state.music = music_service
    app.state.feedback_store = feedback_store
    logger.info("Spotify demo_mode=%s", spotify_client.demo_mode)

    try:
        yield
    finally:
        # Clean shutdown
        logger.info("Shutting down Music Recommendation API")
        await app.state.music.close()
        if app.state.feedback_store is not None:
            app.state.feedback_store.close()


# Use the same limiter instance that owns the route decorators.
limiter = search.limiter

app = FastAPI(
    title="Music Recommendation API",
    description="Hybrid recommendation engine with Spotify and Apple metadata enrichment",
    version=API_VERSION,
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
    allow_credentials=bool(allowed_origin and allowed_origin != "*"),
    allow_methods=["GET", "POST"],
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

    # Strict CSP: all scripts/styles/fonts are self-hosted static files.
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' https: data:; "
        "media-src 'self' https:; "
        "script-src 'self'; "
        "style-src 'self'; "
        "font-src 'self'; "
        "connect-src 'self'; "
        "frame-src https://open.spotify.com;"
    )

    return response


# Cached static files serving
class CachedStatic(StaticFiles):
    async def get_response(self, path, scope):
        resp: Response = await super().get_response(path, scope)
        if resp.status_code == 200:
            # Disable caching in development so changes show immediately
            if ENV == "development":
                resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
            else:
                resp.headers.setdefault("Cache-Control", "public, max-age=86400")
        return resp

# Mount static files with caching
app.mount("/static", CachedStatic(directory="static"), name="static")

# Rate limit exception handler
@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    response = PlainTextResponse("Rate limit exceeded. Please try again later.", status_code=429)
    response.headers["Retry-After"] = str(exc.limit.limit.get_expiry())
    return response

# Include routers
app.include_router(search.router, prefix="/api/v1")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        timestamp=time.time()
    )


@app.get("/")
async def root():
    """Serve the main web interface."""
    response = FileResponse("static/index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


