from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"
    ollama_timeout_seconds: float = 120.0

    # Retry policy for transient model-call failures (timeouts, connection errors, 429/5xx).
    # max_attempts=3 means up to 2 retries after the first try; keep this conservative so a
    # flaky run doesn't silently take multiples of the expected time.
    ollama_retry_max_attempts: int = 3
    ollama_retry_base_delay_seconds: float = 0.5
    ollama_retry_max_delay_seconds: float = 5.0
    ollama_retry_jitter_seconds: float = 0.25

    # Comma-separated list of origins allowed to call the API (the dashboard).
    cors_origins: str = "http://localhost:5173"

    # Root of the benchmark data (documents/, test_cases/, results/, experiments/).
    data_dir: Path = REPO_ROOT / "data"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def test_cases_path(self) -> Path:
        return self.data_dir / "test_cases" / "test_cases.jsonl"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def experiments_dir(self) -> Path:
        return self.data_dir / "experiments"


settings = Settings()
