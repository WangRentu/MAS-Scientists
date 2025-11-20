# evoverse/test/run_demo.py
from __future__ import annotations

import logging

from evoverse.core.system import EvoVerseSystem


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)

    system = EvoVerseSystem()

    question = "如何用深度学习预测蛋白质结构和功能？"

    result = system.run(question)

    print("\n=== 最终总结 ===\n")
    print(result["final_summary"])

    print("\n=== 域分类结果 ===\n")
    print(result.get("classification"))

    print("\n=== 最终 agent 状态（演化之后） ===\n")
    for agent_info in result["agents"]:
        print(agent_info)


if __name__ == "__main__":
    main()
