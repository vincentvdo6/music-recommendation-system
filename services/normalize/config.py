"""Configuration management with Pydantic validation."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseSettings, Field, validator


class DatabaseConfig(BaseSettings):
    url: str = Field(...)
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30
    echo: bool = False


class RedisConfig(BaseSettings):  
    url: str = Field(...)
    max_connections: int = 100
    socket_timeout: int = 5
    socket_connect_timeout: int = 5


class StorageConfig(BaseSettings):
    endpoint: str = Field(...)
    access_key: str = Field(...)
    secret_key: str = Field(...)
    secure: bool = False
    bucket_embeddings: str = "embeddings"
    bucket_raw_audio: str = "raw-audio"
    bucket_previews: str = "previews" 
    bucket_models: str = "models"


class ApiConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    log_level: str = "info"
    cors_origins: List[str] = []
    jwt_secret: str = Field(...)
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440


class AnnConfig(BaseSettings):
    default_k: int = 200
    rerank_k: int = 200
    index_path: str = "data/indexes"
    recall_threshold: float = 0.92


class ModelConfig(BaseSettings):
    version: str
    batch_size: int
    device: str = "cpu"


class EmbeddingsConfig(BaseSettings):
    models: Dict[str, ModelConfig] = {}
    preview_adapter: Dict = {}
    fallback: Dict = {}
    
    @validator('models', pre=True)
    def parse_models(cls, v):
        if isinstance(v, dict):
            return {k: ModelConfig(**config) for k, config in v.items()}
        return v


class RankerConfig(BaseSettings):
    model_path: str = "models/xgb_ranker.bin"
    feature_cache_ttl: int = 60
    batch_size: int = 200


class DiversityConfig(BaseSettings):
    mmr_lambda: float = 0.7
    flow: Dict = {}


class ConsentConfig(BaseSettings):
    max_preview_duration: int = 30
    default_retention_days: int = 90
    salt: str = Field(...)
    kms_key_id: Optional[str] = None


class MetricsConfig(BaseSettings):
    prometheus_port: int = 9090
    log_format: str = "json"


class Config(BaseSettings):
    database: DatabaseConfig
    redis: RedisConfig
    storage: StorageConfig
    api: ApiConfig
    ann: AnnConfig
    embeddings: EmbeddingsConfig
    ranker: RankerConfig
    diversity: DiversityConfig
    consent: ConsentConfig
    metrics: MetricsConfig

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from YAML file with environment variable substitution."""
    if config_path is None:
        config_path = Path("conf/default.yaml")
    
    with open(config_path) as f:
        yaml_content = f.read()
    
    # Simple environment variable substitution
    for key, value in os.environ.items():
        yaml_content = yaml_content.replace(f"${{{key}:.*?}}", value)
        yaml_content = yaml_content.replace(f"${{{key}}}", value)
    
    config_dict = yaml.safe_load(yaml_content)
    return Config(**config_dict)


# Global config instance
config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = load_config()
    return config