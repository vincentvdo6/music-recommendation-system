"""FastAPI application for music recommendation system."""

import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, ORJSONResponse
from dotenv import load_dotenv

from api.routers import search
from services.spotify.client import get_spotify_client

# Load environment variables
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    print("Starting Music Recommendation API v0.1.0")
    # Note: Lazy loading client on first request to avoid blocking startup
    yield
    # Clean shutdown
    try:
        spotify_client = get_spotify_client()
        await spotify_client.close()
    except:
        pass
    print("Shutting down Music Recommendation API")


app = FastAPI(
    title="Music Recommendation API",
    description="Music recommendation system powered by Spotify Web API",
    version="0.1.0",
    default_response_class=ORJSONResponse,  # Faster JSON serialization
    lifespan=lifespan
)

# Add GZip compression for responses > 500 bytes
app.add_middleware(GZipMiddleware, minimum_size=500)

# Add CORS middleware with proper security
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGIN", "http://localhost:8000")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    """Log all requests and add security headers."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    print(f"{request.method} {request.url.path} - {response.status_code} - {duration:.3f}s")
    
    # Add security headers
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' fonts.googleapis.com; font-src fonts.gstatic.com; img-src 'self' data: https:; connect-src 'self'"
    
    return response


# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(search.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "timestamp": time.time()
    }


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
