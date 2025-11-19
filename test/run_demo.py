# evoverse/test/run_demo.py
from __future__ import annotations


import logging

from evoverse.agents.scientist_agent import ScientistConfig, LLMScientistAgent
from evoverse.agents.town_orchestrator import TownOrchestrator
from evoverse.core.llm_client import LLMClient


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)

    llm = LLMClient()

    scientists_cfg = [
        ScientistConfig(
            name="Dr. RL",
            role="强化学习理论研究员",
            expertise="元强化学习、POMDP、探索-利用权衡"
        ),
        ScientistConfig(
            name="Dr. Neuro",
            role="类脑认知科学家",
            expertise="神经表征、可解释性、多模态认知"
        ),
        ScientistConfig(
            name="Dr. Systems",
            role="系统工程师",
            expertise="多智能体系统、大规模分布式训练、系统优化"
        ),
    ]

    scientists = [LLMScientistAgent(cfg, llm) for cfg in scientists_cfg]
    orchestrator = TownOrchestrator(scientists, llm)

    question = "如何设计一个能够自主演化的多智能体 AI 科学家系统，用于跨学科科研？"

    result = orchestrator.run_town_meeting(question)

    print("\n=== 最终总结 ===\n")
    print(result["final_summary"])

    print("\n=== 最终 agent 状态（演化之后） ===\n")
    for a in result["agents"]:
        print(a)


if __name__ == "__main__":
    main()