# evoverse/config/settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class LLMSettings(BaseSettings):
    base_url: str = "http://localhost:8000/v1"  # vLLM / DeepSeek / Qwen 兼容接口
    api_key: str = "dummy-key"
    model: str = "qwen3-32b-instruct"
    max_tokens: int = 2048
    temperature: float = 0.2
    request_timeout: float = Field(
        default=30.0,
        description="LLM 请求超时时间（秒），避免长时间无响应"
    )

    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


class TownSettings(BaseSettings):
    # 小镇讨论与演化参数
    num_rounds: int = 3
    max_opinion_length: int = 800
    evolution_threshold: float = 0.4  # 低于这个声望就触发“进化”

    model_config = SettingsConfigDict(
        env_prefix="TOWN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


llm_settings = LLMSettings()
town_settings = TownSettings()