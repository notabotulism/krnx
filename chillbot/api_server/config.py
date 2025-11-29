"""
KRNX API Server Configuration

Settings for FastAPI server, Redis, and kernel initialization.
"""

from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache


class Settings(BaseSettings):
    """API Server Settings"""
    
    # Server
    host: str = "0.0.0.0"
    port: int = 6380
    debug: bool = False
    
    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_max_connections: int = 200
    
    # Kernel
    data_path: str = "./krnx-data"
    enable_backpressure: bool = True
    enable_hash_chain: bool = True
    max_queue_depth: int = 50000
    max_lag_seconds: float = 30.0
    
    # LTM
    warm_retention_days: int = 30
    
    # Fabric
    auto_embed: bool = False  # Disabled by default until embeddings configured
    auto_enrich: bool = True
    default_workspace: str = "default"
    
    # API
    api_prefix: str = "/api/v1"
    cors_origins: list = ["*"]
    
    class Config:
        env_prefix = "KRNX_"
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
