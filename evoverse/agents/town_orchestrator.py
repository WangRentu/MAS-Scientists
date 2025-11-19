# evoverse/agents/town_orchestrator.py
from __future__ import annotations

from typing import List, Dict, Any
import json
import logging

from evoverse.agents.scientist_agent import LLMScientistAgent
from evoverse.core.llm_client import LLMClient, create_message
from evoverse.config.settings import town_settings

logger = logging.getLogger(__name__)


class TownOrchestrator:
    def __init__(self, scientists: List[LLMScientistAgent], llm: LLMClient):
        self.scientists = scientists
        self.llm = llm
        logger.info("Town orchestrator initialized with %d scientists", len(scientists))

    # -------------------- 单轮讨论 --------------------

    def _run_single_round(
        self,
        question: str,
        round_idx: int,
        global_summary: str,
    ) -> Dict[str, Any]:
        # 1. 每个 scientist 给出意见
        logger.info("Running round %d for question: %s", round_idx, question)
        opinions = []
        for sci in self.scientists:
            opinion = sci.propose_opinion(question, context_summary=global_summary)
            opinions.append(
                {
                    "name": sci.cfg.name,
                    "role": sci.cfg.role,
                    "opinion": opinion,
                    "reputation": sci.cfg.reputation,
                }
            )
            logger.debug("Opinion collected | sci=%s rep=%.2f", sci.cfg.name, sci.cfg.reputation)

        # 2. 互评（简化版：每个人只点评“下一个人”）
        critiques: List[Dict[str, Any]] = []
        n = len(self.scientists)
        for i, sci in enumerate(self.scientists):
            target = self.scientists[(i + 1) % n]
            target_opinion = opinions[(i + 1) % n]["opinion"]

            critique_raw = sci.critique_opinion(question, target.cfg.name, target_opinion)

            try:
                data = json.loads(critique_raw.strip().strip("` "))
                score = float(data.get("score", 0.5))
                comment = data.get("comment", critique_raw)
            except Exception:
                score = 0.5
                comment = critique_raw

            critiques.append(
                {
                    "from": sci.cfg.name,
                    "to": target.cfg.name,
                    "score": score,
                    "comment": comment,
                }
            )
            # 被点评者 reputation 更新：简单滑动平均
            target.cfg.reputation = 0.7 * target.cfg.reputation + 0.3 * score
            logger.info(
                "Critique | from=%s to=%s score=%.2f updated_rep=%.2f",
                sci.cfg.name,
                target.cfg.name,
                score,
                target.cfg.reputation,
            )

        # 3. 用 LLM 做本轮总结
        summary = self._summarize_round(question, round_idx, opinions, critiques)
        logger.info("Round %d summary generated", round_idx)

        return {
            "round": round_idx,
            "opinions": opinions,
            "critiques": critiques,
            "summary": summary,
        }

    def _summarize_round(
        self,
        question: str,
        round_idx: int,
        opinions: List[Dict[str, Any]],
        critiques: List[Dict[str, Any]],
    ) -> str:
        text = f"第 {round_idx} 轮讨论的问题是：{question}\n\n"
        text += "各位科学家的意见：\n"
        for op in opinions:
            text += f"- {op['name']} ({op['role']}):\n{op['opinion']}\n\n"

        text += "互评情况：\n"
        for c in critiques:
            text += f"- {c['from']} 给 {c['to']} 评分 {c['score']:.2f}，评论：{c['comment']}\n"

        user_prompt = (
            "请你作为会议主持人，总结这一轮讨论的："
            "1) 共识点；2) 主要分歧；3) 下一步可以执行的实验或行动建议。"
        )

        logger.info("Summarizing round %d via LLM", round_idx)
        resp = self.llm.chat(
            [
                create_message(role="system", content="你是一个严谨的科研讨论主持人。"),
                create_message(role="user", content=text + "\n\n" + user_prompt),
            ]
        )
        return resp

    # -------------------- 多轮讨论 + 自主演化 --------------------

    def run_town_meeting(self, question: str) -> Dict[str, Any]:
        global_summary = ""
        history_rounds: List[Dict[str, Any]] = []
        logger.info("Town meeting started | question=%s rounds=%d", question, town_settings.num_rounds)

        for r in range(1, town_settings.num_rounds + 1):
            result = self._run_single_round(question, r, global_summary)
            history_rounds.append(result)
            global_summary = result["summary"]
            logger.info("Round %d completed", r)

            # 每轮结束后，看哪些 agent 声望太低要进化
            feedback_text = self._build_feedback_for_evolution(history_rounds)
            for sci in self.scientists:
                if sci.cfg.reputation < town_settings.evolution_threshold:
                    logger.warning(
                        "Scientist %s reputation %.2f below threshold %.2f, triggering evolution",
                        sci.cfg.name,
                        sci.cfg.reputation,
                        town_settings.evolution_threshold,
                    )
                    sci.evolve(feedback_text)

        final_summary = self._final_summary(question, history_rounds)

        return {
            "question": question,
            "rounds": history_rounds,
            "final_summary": final_summary,
            "agents": [
                {
                    "name": s.cfg.name,
                    "role": s.cfg.role,
                    "expertise": s.cfg.expertise,
                    "worldview": s.cfg.worldview,
                    "reputation": s.cfg.reputation,
                }
                for s in self.scientists
            ],
        }

    def _build_feedback_for_evolution(self, history_rounds: List[Dict[str, Any]]) -> str:
        # 只用最近两轮总结 + 评分做反馈
        text = ""
        for r in history_rounds[-2:]:
            text += f"[第 {r['round']} 轮总结]\n{r['summary']}\n\n"
            for c in r["critiques"]:
                text += f"评分记录：{c['from']} 给 {c['to']} {c['score']:.2f} 分。\n"
        return text

    def _final_summary(self, question: str, history_rounds: List[Dict[str, Any]]) -> str:
        text = f"科研问题：{question}\n\n"
        for r in history_rounds:
            text += f"第 {r['round']} 轮总结：\n{r['summary']}\n\n"

        user_prompt = (
            "基于以上多轮讨论，请你给出一个最终研究计划，包含：\n"
            "1) 最核心的研究假设；\n"
            "2) 建议的实验或仿真设计；\n"
            "3) 需要跨学科合作的关键点；\n"
            "4) 主要风险和备选方案。"
        )

        logger.info("Generating final summary via LLM")
        resp = self.llm.chat(
            [
                create_message(role="system", content="你是一名 AI 科学项目总负责人。"),
                create_message(role="user", content=text + "\n\n" + user_prompt),
            ]
        )
        return resp