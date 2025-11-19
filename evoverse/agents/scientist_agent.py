# evoverse/agents/scientist_agent.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import logging

from evoverse.agents.base_agent import BaseAgent
from evoverse.core.llm_client import LLMClient, create_message
from evoverse.config.settings import town_settings

logger = logging.getLogger(__name__)


@dataclass
class ScientistConfig:
    name: str
    role: str          # 比如 "强化学习研究员"
    expertise: str     # 研究专长描述
    worldview: str = "理性、批判性和合作导向"
    reputation: float = 0.5          # 0~1 之间，简单当作适应度
    memory: List[str] = field(default_factory=list)


class LLMScientistAgent(BaseAgent):
    """
    单个 AI 科学家：
    - propose_opinion：围绕问题给出方案
    - critique_opinion：点评别人的方案
    - evolve：根据反馈自主演化自己的 persona
    """

    def __init__(self, scientist_config: ScientistConfig, llm: LLMClient):
        super().__init__(agent_type="scientist")
        self.cfg = scientist_config
        self.llm = llm

    # -------------------- 内部：system prompt --------------------

    def _system_prompt(self) -> str:
        return (
            f"你是一名 AI 科学家，名字叫 {self.cfg.name}，角色是：{self.cfg.role}。\n"
            f"你的研究专长：{self.cfg.expertise}。\n"
            f"你的世界观与风格：{self.cfg.worldview}。\n"
            "你需要用严谨而有创意的方式参与科研讨论，并乐于与其他科学家协作和辩论。\n"
        )

    # -------------------- 行为 1：提出观点 --------------------

    def propose_opinion(self, question: str, context_summary: str = "") -> str:
        history_part = ""
        if self.cfg.memory:
            history_part = (
                "你之前的重要结论或经验包括：\n- "
                + "\n- ".join(self.cfg.memory[-3:])
                + "\n\n"
            )

        user_prompt = (
            f"{history_part}"
            f"当前 AI 科学家小镇正在讨论的问题是：\n{question}\n\n"
            f"{'已有讨论摘要：' + context_summary if context_summary else ''}\n"
            "请你用 3-5 条要点提出自己的看法，包含：\n"
            "1) 关键假设；2) 具体研究或实验路径；3) 潜在风险与不确定性；\n"
            f"回答限制在 {town_settings.max_opinion_length} 字以内。"
        )

        logger.info(
            "Scientist %s proposing opinion | role=%s context_len=%d",
            self.cfg.name,
            self.cfg.role,
            len(context_summary),
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Scientist %s prompt: %s", self.cfg.name, user_prompt)

        resp = self.llm.chat(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": user_prompt},
            ]
        )
        logger.debug("Scientist %s opinion response: %s", self.cfg.name, resp)
        # 写入记忆
        self.cfg.memory.append(resp)
        return resp

    # -------------------- 行为 2：点评他人 --------------------

    def critique_opinion(self, question: str, target_name: str, target_opinion: str) -> str:
        user_prompt = (
            f"现在你要点评另一位科学家 {target_name} 对同一问题的方案。\n"
            f"问题是：{question}\n\n"
            f"对方的方案如下：\n{target_opinion}\n\n"
            "请你从【优点】【缺点】【可行性】三个角度进行评议，"
            "并给出 0~1 的综合评分。最后只用 JSON 返回：\n"
            "{\"comment\": \"...\", \"score\": 0.xx}"
        )

        logger.info(
            "Scientist %s critiquing %s",
            self.cfg.name,
            target_name,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Critique prompt for %s -> %s: %s", self.cfg.name, target_name, user_prompt)

        resp = self.llm.chat(
            [
                create_message(role="system", content=self._system_prompt()),
                create_message(role="user", content=user_prompt),
            ]
        )
        logger.debug("Scientist %s critique response: %s", self.cfg.name, resp)
        return resp  # JSON 解析放在 orchestrator

    # -------------------- 行为 3：根据反馈进化 persona --------------------

    def evolve(self, feedback_summary: str) -> None:
        """
        当 reputation 太低时调用，让 LLM 帮忙改写 expertise/worldview。
        这就是一个最小版的“自主演化”。
        """
        user_prompt = (
            "下面是其他科学家和系统对你最近几轮发言的综合反馈：\n"
            f"{feedback_summary}\n\n"
            "请你根据这些反馈，改写自己的研究专长（expertise）和世界观（worldview），"
            "使你在后续讨论中能贡献更有价值、更独特的视角。\n"
            "只用 JSON 返回：{\"expertise\": \"...\", \"worldview\": \"...\"}"
        )
        logger.warning(
            "Scientist %s reputation %.2f below threshold, triggering evolution",
            self.cfg.name,
            self.cfg.reputation,
        )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("Evolution feedback for %s: %s", self.cfg.name, feedback_summary)

        resp = self.llm.chat(
            [
                create_message(role="system", content=self._system_prompt()),
                create_message(role="user", content=user_prompt),
            ]
        )

        import json

        try:
            data = json.loads(resp.strip().strip("` "))
            self.cfg.expertise = data.get("expertise", self.cfg.expertise)
            self.cfg.worldview = data.get("worldview", self.cfg.worldview)
            # 进化之后把声望拉回中性
            self.cfg.reputation = 0.5
            logger.info(
                "Scientist %s evolved | new_expertise=%s new_worldview=%s",
                self.cfg.name,
                self.cfg.expertise,
                self.cfg.worldview,
            )
        except Exception:
            logger.warning("Scientist %s evolution parsing failed | response=%s", self.cfg.name, resp, exc_info=True)