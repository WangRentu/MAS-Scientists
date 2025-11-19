# evoverse/test/run_demo.py
from __future__ import annotations


import logging

from evoverse.agents.scientist_agent import (
    ScientistConfig,
    ScientistGenome,
    LLMScientistAgent,
)
from evoverse.agents.literature_agent import LiteratureAgent
from evoverse.agents.town_orchestrator import TownOrchestrator
from evoverse.core.llm_client import LLMClient


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)

    llm = LLMClient()
    literature_agent = LiteratureAgent(llm_client=llm)

    scientists_cfg = [
        ScientistConfig(
            name="Dr. RL",
            role="强化学习理论研究员",
            expertise="元强化学习、POMDP、探索-利用权衡",
            genome=ScientistGenome(
                discipline="Reinforcement Learning",
                risk_taking=0.7,
                criticism_level=0.4,
                collab_preference=0.6,
                rag_depth=2,
                summary_granularity="fine",
            ),
        ),
        ScientistConfig(
            name="Dr. Neuro",
            role="类脑认知科学家",
            expertise="神经表征、可解释性、多模态认知",
            genome=ScientistGenome(
                discipline="Neuroscience",
                risk_taking=0.5,
                criticism_level=0.6,
                collab_preference=0.7,
                rag_depth=3,
                summary_granularity="coarse",
            ),
        ),
        ScientistConfig(
            name="Dr. Systems",
            role="系统工程师",
            expertise="多智能体系统、大规模分布式训练、系统优化",
            genome=ScientistGenome(
                discipline="Distributed Systems",
                risk_taking=0.4,
                criticism_level=0.7,
                collab_preference=0.5,
                rag_depth=1,
                summary_granularity="coarse",
            ),
        ),
    ]

    scientists = [
        LLMScientistAgent(cfg, llm, literature_agent=literature_agent)
        for cfg in scientists_cfg
    ]
    orchestrator = TownOrchestrator(scientists, llm)

    question = "如何用深度学习预测蛋白质结构和功能？"

    result = orchestrator.run_town_meeting(question)

    print("\n=== 最终总结 ===\n")
    print(result["final_summary"])

    print("\n=== 最终 agent 状态（演化之后） ===\n")
    for a in result["agents"]:
        print(a)


if __name__ == "__main__":
    main()
