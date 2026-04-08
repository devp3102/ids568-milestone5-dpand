from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── Model Settings ────────────────────────────────────────────────────────
    model_name: str = Field(
        default="sshleifer/tiny-gpt2",
        description="HuggingFace model ID to load for inference",
    )
    max_sequence_length: int = Field(
        default=512,
        description="Maximum prompt + output token length (safety limit)",
    )
    max_output_tokens: int = Field(
        default=200,
        description="Default cap on generated tokens per request",
    )

    # ── Batching Settings ─────────────────────────────────────────────────────
    max_batch_size: int = Field(
        default=8,
        description="Maximum number of requests to group into a single batch",
    )
    batch_timeout_ms: float = Field(
        default=50.0,
        description="Maximum milliseconds to wait before processing a partial batch",
    )

    # ── Caching Settings ──────────────────────────────────────────────────────
    cache_backend: str = Field(
        default="memory",
        description="Cache backend to use: 'memory' (in-process) or 'redis'",
    )
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL (only used when cache_backend='redis')",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live in seconds for cached responses",
    )
    cache_max_entries: int = Field(
        default=1000,
        description="Maximum number of entries to keep in the in-process cache",
    )

    # ── Server Settings ───────────────────────────────────────────────────────
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    request_timeout_seconds: float = Field(
        default=30.0,
        description="Per-request timeout; requests exceeding this are cancelled",
    )

    class Config:
        env_prefix = "LLM_"       # e.g. LLM_MAX_BATCH_SIZE=16
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton settings instance consumed by all modules
settings = Settings()