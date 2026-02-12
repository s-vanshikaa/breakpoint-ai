from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:1b"

    class Config:
        env_file = ".env"


settings = Settings()
