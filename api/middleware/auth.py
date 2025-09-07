"""JWT authentication middleware."""

import logging
from typing import Optional

import jwt
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from services.normalize.config import get_config

logger = logging.getLogger(__name__)


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """JWT authentication middleware for protected endpoints."""
    
    # Endpoints that don't require authentication
    EXEMPT_PATHS = {
        "/health",
        "/metrics", 
        "/",
        "/docs",
        "/openapi.json"
    }
    
    def __init__(self, app):
        super().__init__(app)
        self.config = get_config()
    
    async def dispatch(self, request: Request, call_next):
        """Process request with JWT validation."""
        
        # Skip auth for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)
        
        # Extract JWT token
        token = self._extract_token(request)
        
        if not token:
            raise HTTPException(status_code=401, detail="Missing authentication token")
        
        # Verify token
        try:
            payload = jwt.decode(
                token,
                self.config.api.jwt_secret,
                algorithms=[self.config.api.jwt_algorithm]
            )
            
            # Add user info to request state
            request.state.user_id = payload.get("sub")
            request.state.user_scopes = payload.get("scopes", [])
            
            logger.debug(
                "JWT validated",
                user_id=request.state.user_id,
                scopes=request.state.user_scopes
            )
            
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning("Invalid JWT token", error=str(e))
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        
        return await call_next(request)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """Extract JWT token from Authorization header."""
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return None
        
        try:
            scheme, token = auth_header.split()
            if scheme.lower() != "bearer":
                return None
            return token
        except ValueError:
            return None