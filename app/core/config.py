from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "chan-knn-stock-backend"
    app_version: str = "0.1.0"
    debug: bool = False
    storage_dir: Path = Path(__file__).resolve().parents[2] / "storage"
    data_dir_name: str = "data_cache"
    model_dir_name: str = "models"
    default_start_date: str = "20180101"
    default_end_date: str = "20261231"
    model_config = SettingsConfigDict(env_prefix="STOCK_AI_", env_file=".env", extra="ignore")

    @property
    def data_dir(self) -> Path:
        return self.storage_dir / self.data_dir_name

    @property
    def model_dir(self) -> Path:
        return self.storage_dir / self.model_dir_name


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.model_dir.mkdir(parents=True, exist_ok=True)
