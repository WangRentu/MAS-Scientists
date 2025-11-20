from __future__ import annotations

"""
High-level EvoVerse system wrapper.

封装常用组件（LLMClient、LiteratureAgent、DomainRouter、TownOrchestrator），
对外提供一个统一的 `EvoVerseSystem.run(question, min_experts=3)` 接口。
"""

from typing import Optional, Dict, Any
import logging

from evoverse.core.llm_client import LLMClient
from evoverse.core.domain_router import DomainRouter
from evoverse.agents.literature_agent import LiteratureAgent
from evoverse.agents.town_orchestrator import TownOrchestrator

logger = logging.getLogger(__name__)


class EvoVerseSystem:
    """
    High-level orchestrator of MAS Scientists.

    Usage:
        system = EvoVerseSystem()
        result = system.run("如何用深度学习预测蛋白质结构和功能？")
    """

    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        literature_agent: Optional[LiteratureAgent] = None,
        domain_router: Optional[DomainRouter] = None,
    ):
        self.llm = llm or LLMClient()
        self.literature_agent = literature_agent or LiteratureAgent(llm_client=self.llm)
        self.domain_router = domain_router or DomainRouter(llm_client=self.llm)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        min_experts: int = 3,
    ) -> Dict[str, Any]:
        """
        Run the full MAS workflow for a single research question.
        Returns TownOrchestrator's output with domain classification/routing info.
        """
        orchestrator = TownOrchestrator(
            llm=self.llm,
            literature_agent=self.literature_agent,
            domain_router=self.domain_router,
            min_experts=min_experts,
        )

        result = orchestrator.run_town_meeting(question, min_experts=min_experts)

        return result


__all__ = ["EvoVerseSystem"]
