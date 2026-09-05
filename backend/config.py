"""NovaScientist Production Configuration.

Manages environment-driven settings for storage, API server, LLM providers,
sandboxing, security, and execution resource limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class NovaScientistConfig:
    """Production configuration for NovaScientist."""

    # Environment & Version
    app_version: str = "2.3.0"
    environment: str = os.getenv("NOVASCIENTIST_ENV", "production")
    debug: bool = os.getenv("NOVASCIENTIST_DEBUG", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    # Storage & Workspace
    data_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("NOVASCIENTIST_DATA_DIR", ".novascientist_data")
        ).resolve()
    )

    # API Server
    api_host: str = os.getenv("NOVASCIENTIST_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("NOVASCIENTIST_PORT", "8000"))
    api_rate_limit: int = int(os.getenv("NOVASCIENTIST_RATE_LIMIT", "60"))  # req / min
    cors_origins: list[str] = field(
        default_factory=lambda: [
            origin.strip()
            for origin in os.getenv("NOVASCIENTIST_CORS_ORIGINS", "*").split(",")
        ]
    )

    # LLM Provider Configuration
    llm_provider: str = os.getenv("NOVASCIENTIST_LLM_PROVIDER", "deterministic_mock")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    llm_timeout_seconds: float = float(os.getenv("NOVASCIENTIST_LLM_TIMEOUT", "45.0"))
    llm_max_retries: int = int(os.getenv("NOVASCIENTIST_LLM_MAX_RETRIES", "3"))
    llm_cost_limit_usd: float = float(
        os.getenv("NOVASCIENTIST_LLM_COST_LIMIT_USD", "10.0")
    )

    # Execution & Resource Controls
    max_concurrent_jobs: int = int(os.getenv("NOVASCIENTIST_MAX_CONCURRENT_JOBS", "4"))
    job_timeout_seconds: float = float(os.getenv("NOVASCIENTIST_JOB_TIMEOUT", "600.0"))
    sandbox_max_memory_mb: int = int(
        os.getenv("NOVASCIENTIST_SANDBOX_MAX_MEMORY_MB", "1024")
    )
    demo_mode: bool = os.getenv("NOVASCIENTIST_DEMO_MODE", "false").lower() in (
        "true",
        "1",
        "yes",
    )

    def __post_init__(self) -> None:
        """Ensure critical directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "workspaces").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(parents=True, exist_ok=True)


# Global default configuration instance
config = NovaScientistConfig()
