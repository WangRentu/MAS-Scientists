"""
Configuration module for EvoVerse.
"""

from .settings import LLMSettings, TownSettings, llm_settings, town_settings
from typing import Any, Dict
from pydantic import BaseModel


class Config(BaseModel):
    """Unified configuration."""
    llm: LLMSettings
    town: TownSettings


def get_config() -> Config:
    """Get unified configuration."""
    return Config(llm=llm_settings, town=town_settings)


__all__ = [
    "LLMSettings",
    "TownSettings", 
    "llm_settings",
    "town_settings",
    "get_config",
    "Config"
]
