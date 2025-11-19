# evoverse/config/settings.py
from typing import Optional

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

class LiteratureConfig(BaseSettings):
    """文献检索配置"""
    semantic_scholar_api_key: Optional[str] = Field(default=None, description="Semantic Scholar API Key")
    semantic_scholar_api_url: Optional[str] = Field(default=None, description="Semantic Scholar API URL")
    pubmed_api_key: Optional[str] = Field(default=None, description="PubMed API Key")
    pubmed_email: Optional[str] = Field(default=None, description="PubMed 联系邮箱")
    cache_dir: str = Field(default=".literature_cache", description="缓存目录")
    cache_ttl_hours: int = Field(default=48, description="缓存 TTL (小时)")
    max_cache_size_mb: int = Field(default=1000, description="缓存最大大小 (MB)")
    max_results_per_query: int = Field(default=100, description="单次检索最大结果数")
    pdf_download_timeout: int = Field(default=30, description="PDF 下载超时 (秒)")

    model_config = {
        "env_prefix": "LITERATURE_",
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore"
    }

class EvoVerseConfig(BaseSettings):
    """EvoVerse 配置"""
    llm: LLMSettings = Field(default=LLMSettings())
    town: TownSettings = Field(default=TownSettings())
    literature: LiteratureConfig = Field(default=LiteratureConfig())


def get_config() -> EvoVerseConfig:
    """获取 EvoVerse 配置"""
    return EvoVerseConfig()

