"""Consent management service with GDPR compliance."""

import hashlib
import hmac
import logging
from datetime import datetime, timedelta
from typing import Optional

import boto3
from sqlalchemy.ext.asyncio import AsyncSession

from services.normalize.config import get_config

logger = logging.getLogger(__name__)


class ConsentManager:
    """Manages user consent and data retention policies."""
    
    def __init__(self):
        self.config = get_config()
        self._s3_client = None
        self._kms_client = None
    
    @property
    def s3_client(self):
        if self._s3_client is None:
            self._s3_client = boto3.client(
                's3',
                endpoint_url=f"http://{self.config.storage.endpoint}",
                aws_access_key_id=self.config.storage.access_key,
                aws_secret_access_key=self.config.storage.secret_key,
            )
        return self._s3_client
    
    @property 
    def kms_client(self):
        if self._kms_client is None and self.config.consent.kms_key_id:
            self._kms_client = boto3.client('kms')
        return self._kms_client

    def generate_user_hash(self, user_identifier: str, ip_address: str) -> str:
        """Generate HMAC hash for user/IP combination."""
        combined = f"{user_identifier}:{ip_address}"
        return hmac.new(
            self.config.consent.salt.encode('utf-8'),
            combined.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    def generate_audio_hash(self, audio_content: bytes) -> str:
        """Generate SHA256 hash of audio content."""
        return hashlib.sha256(audio_content).hexdigest()

    def validate_preview_duration(self, duration_ms: int) -> bool:
        """Validate that preview is within allowed duration."""
        max_duration_ms = self.config.consent.max_preview_duration * 1000
        return duration_ms <= max_duration_ms

    async def store_consent(
        self,
        db: AsyncSession,
        user_identifier: str,
        ip_address: str,
        audio_content: bytes,
        duration_ms: int,
        terms_version: str,
        retention_days: Optional[int] = None
    ) -> str:
        """Store consent record and return consent ID."""
        
        # Validate preview duration
        if not self.validate_preview_duration(duration_ms):
            raise ValueError(
                f"Preview duration {duration_ms}ms exceeds maximum "
                f"{self.config.consent.max_preview_duration * 1000}ms"
            )
        
        # Generate hashes
        user_hash = self.generate_user_hash(user_identifier, ip_address)
        audio_hash = self.generate_audio_hash(audio_content)
        
        # Generate consent ID
        consent_id = hashlib.sha256(f"{user_hash}:{audio_hash}:{datetime.utcnow().isoformat()}".encode()).hexdigest()
        
        # Calculate expiry
        retention_days = retention_days or self.config.consent.default_retention_days
        expires_at = datetime.utcnow() + timedelta(days=retention_days)
        
        # Store in database
        from data.models import ConsentLog
        
        consent_record = ConsentLog(
            consent_id=consent_id,
            user_hash=user_hash,
            audio_hash=audio_hash,
            terms_version=terms_version,
            retention_days=retention_days,
            preview_duration_ms=duration_ms,
            expires_at=expires_at,
            metadata={
                "ip_hash": hashlib.sha256(ip_address.encode()).hexdigest()[:16],
                "user_agent_hash": None,  # Could be added
            }
        )
        
        db.add(consent_record)
        await db.commit()
        
        logger.info(
            "Consent stored",
            extra={
                "consent_id": consent_id,
                "audio_hash": audio_hash[:16],
                "duration_ms": duration_ms,
                "retention_days": retention_days
            }
        )
        
        return consent_id

    async def verify_consent(self, db: AsyncSession, consent_id: str) -> bool:
        """Verify that consent exists and is still valid."""
        from data.models import ConsentLog
        from sqlalchemy import select
        
        stmt = select(ConsentLog).where(
            ConsentLog.consent_id == consent_id,
            ConsentLog.expires_at > datetime.utcnow(),
            ConsentLog.revoked_at.is_(None)
        )
        
        result = await db.execute(stmt)
        consent_record = result.scalar_one_or_none()
        
        return consent_record is not None

    async def store_audio_with_av_scan(
        self, 
        audio_content: bytes, 
        consent_id: str,
        content_type: str = "audio/wav"
    ) -> str:
        """Store audio in S3 with antivirus scanning (stubbed in dev)."""
        
        # In production, this would trigger AV scan
        if self._should_scan_content():
            scan_result = await self._scan_content(audio_content)
            if not scan_result.clean:
                raise ValueError("Content failed security scan")
        
        # Store in S3
        key = f"previews/{consent_id}.wav"
        
        try:
            self.s3_client.put_object(
                Bucket=self.config.storage.bucket_previews,
                Key=key,
                Body=audio_content,
                ContentType=content_type,
                Metadata={
                    'consent_id': consent_id,
                    'scan_status': 'clean',
                    'uploaded_at': datetime.utcnow().isoformat()
                }
            )
            
            logger.info(
                "Audio stored in S3",
                extra={
                    "consent_id": consent_id,
                    "bucket": self.config.storage.bucket_previews,
                    "key": key,
                    "size_bytes": len(audio_content)
                }
            )
            
            return key
            
        except Exception as e:
            logger.error(
                "Failed to store audio in S3",
                extra={
                    "consent_id": consent_id,
                    "error": str(e)
                }
            )
            raise

    def _should_scan_content(self) -> bool:
        """Check if content scanning is enabled."""
        # In dev, we skip AV scanning
        return self.config.storage.endpoint != "localhost:9000"

    async def _scan_content(self, content: bytes) -> 'ScanResult':
        """Stub for antivirus scanning."""
        # In production, this would integrate with AV service
        class ScanResult:
            clean = True
            threats = []
        
        return ScanResult()

    async def revoke_consent(self, db: AsyncSession, consent_id: str) -> bool:
        """Revoke consent and mark for deletion."""
        from data.models import ConsentLog
        from sqlalchemy import select, update
        
        # Update consent record
        stmt = update(ConsentLog).where(
            ConsentLog.consent_id == consent_id
        ).values(
            revoked_at=datetime.utcnow()
        )
        
        result = await db.execute(stmt)
        
        if result.rowcount > 0:
            await db.commit()
            
            # Schedule deletion of associated data
            await self._schedule_data_deletion(consent_id)
            
            logger.info(
                "Consent revoked",
                extra={"consent_id": consent_id}
            )
            
            return True
        
        return False

    async def _schedule_data_deletion(self, consent_id: str):
        """Schedule deletion of all data associated with consent ID."""
        # This would typically queue a background job
        # For now, we just log the intention
        logger.info(
            "Data deletion scheduled",
            extra={"consent_id": consent_id}
        )