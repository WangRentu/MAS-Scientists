from __future__ import annotations

"""
Expert factory utilities.

提供三类能力：
1) 原始生成：根据自然语言科研需求 + 领域，生成一个全新的 ScientistConfig。
2) 变异（mutation）：在现有专家基因的基础上做小幅扰动，产生变体专家。
3) 杂交（crossover）：将两个专家的基因组合，产生新专家。
"""

from typing import Dict, Any, Optional
import logging
import random

from evoverse.core.llm_client import LLMClient
from evoverse.models.domain import ScientificDomain
from evoverse.agents.scientist_agent import (
    ScientistConfig,
    ScientistGenome,
    LLMScientistAgent,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON schema for LLM-based generation
# ---------------------------------------------------------------------------

def _schema_for_scientist() -> Dict[str, Any]:
    """JSON schema describing the structure of a generated ScientistConfig."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "role": {"type": "string"},
            "expertise": {"type": "string"},
            "worldview": {"type": "string"},
            "genome": {
                "type": "object",
                "properties": {
                    "domain": {
                        "type": "string",
                        "description": (
                            "ScientificDomain 枚举值，如 biology / neuroscience / "
                            "materials / physics / chemistry / astronomy / "
                            "social_science / general"
                        ),
                    },
                    "discipline": {
                        "type": "string",
                        "description": (
                            "更细学科标签，例如 Computational Epidemiology, "
                            "Reinforcement Learning"
                        ),
                    },
                    "risk_taking": {
                        "type": "number",
                        "description": "0~1 之间的浮点数，越大越敢于探索高风险方案",
                    },
                    "criticism_level": {
                        "type": "number",
                        "description": "0~1 之间的浮点数，越大越尖锐批判",
                    },
                    "collab_preference": {
                        "type": "number",
                        "description": "0~1 之间的浮点数，越大越偏向合作和引用他人观点",
                    },
                    "rag_depth": {
                        "type": "integer",
                        "description": "检索深度，通常为 1, 2 或 3",
                    },
                    "summary_granularity": {
                        "type": "string",
                        "description": "记忆写入粒度：coarse / fine",
                    },
                },
            },
        },
        "required": ["name", "role", "expertise", "worldview", "genome"],
    }


# ---------------------------------------------------------------------------
# 1) 原始生成：question + domain -> ScientistConfig
# ---------------------------------------------------------------------------

def generate_scientist_for_domain(
    question: str,
    domain: ScientificDomain,
    llm: LLMClient,
) -> ScientistConfig:
    """
    Generate a new scientist expert tailored to a given research question
    and scientific domain.

    典型用法：根据用户输入的“计算机 × 流行病学交叉需求”和分类出的领域，
    自动合成一个新的专家配置。
    """
    system_prompt = (
        "你是一个科研组织中的人事负责人，负责为特定交叉科研需求设计 AI 科学家专家。\n"
        "你需要根据用户给出的研究问题和目标学科领域，设计一位最合适的专家，包括：姓名、角色、专长、世界观，"
        "以及一组可演化的基因参数（domain / discipline / risk_taking / criticism_level / "
        "collab_preference / rag_depth / summary_granularity）。\n"
        "请确保返回的 JSON 严格符合给定的 schema。"
    )

    prompt = (
        "【研究问题描述】\n"
        f"{question}\n\n"
        "【目标学科领域（ScientificDomain 枚举）】\n"
        f"{domain.value}\n\n"
        "请为这个问题和领域设计一位 AI 科学家专家，要求：\n"
        "1. 他的 role 和 expertise 要清晰描述在这个交叉方向上擅长什么；\n"
        "2. worldview 反映他的科研风格（例如：谨慎严谨 / 激进探索 / 强调因果 / 偏向实证等）；\n"
        "3. genome.domain 字段尽量与上面的目标领域一致；\n"
        "4. risk_taking / criticism_level / collab_preference 在 0~1 之间，数值含义清晰；\n"
        "5. rag_depth 为 1~3 的整数；summary_granularity 为 \"coarse\" 或 \"fine\"。\n\n"
        "只需要返回 JSON，不要附加解释。"
    )

    schema = _schema_for_scientist()

    logger.info("Generating new scientist for domain=%s", domain.value)
    data = llm.generate_structured(
        prompt=prompt,
        output_schema=schema,
        system=system_prompt,
        max_tokens=1024,
    )

    genome_raw = (data or {}).get("genome") or {}

    domain_str = str(genome_raw.get("domain") or "").strip().lower()
    try:
        domain_enum = ScientificDomain(domain_str) if domain_str else domain
    except ValueError:
        domain_enum = domain

    discipline = genome_raw.get("discipline") or ""

    def _float(name: str, default: float) -> float:
        try:
            return float(genome_raw.get(name, default))
        except (TypeError, ValueError):
            return default

    def _int(name: str, default: int) -> int:
        try:
            return int(genome_raw.get(name, default))
        except (TypeError, ValueError):
            return default

    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    risk_taking = _clip01(_float("risk_taking", 0.5))
    criticism_level = _clip01(_float("criticism_level", 0.5))
    collab_preference = _clip01(_float("collab_preference", 0.5))
    rag_depth = max(1, _int("rag_depth", 1))
    summary_granularity = genome_raw.get("summary_granularity") or "coarse"
    if summary_granularity not in ("coarse", "fine"):
        summary_granularity = "coarse"

    genome = ScientistGenome(
        domain=domain_enum,
        discipline=discipline,
        risk_taking=risk_taking,
        criticism_level=criticism_level,
        collab_preference=collab_preference,
        rag_depth=rag_depth,
        summary_granularity=summary_granularity,
    )

    cfg = ScientistConfig(
        name=data.get("name", "AutoScientist"),
        role=data.get("role", f"{domain_enum.value} 方向科研专家"),
        expertise=data.get("expertise", ""),
        worldview=data.get("worldview", "理性、批判性和合作导向"),
        genome=genome,
    )

    logger.info(
        "Generated scientist | name=%s role=%s domain=%s discipline=%s",
        cfg.name,
        cfg.role,
        genome.domain.value,
        genome.discipline,
    )
    return cfg


# ---------------------------------------------------------------------------
# 2) 变异：在单个专家基因上做小幅扰动
# ---------------------------------------------------------------------------

def mutate_genome(parent: ScientistGenome, noise: float = 0.15) -> ScientistGenome:
    """
    数值层面的基因变异：在不改变领域的大前提下，对若干浮点 trait 做轻微扰动。

    适合在父代表现不错时，复制出风格略有差异的变体。
    """

    def clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    return ScientistGenome(
        domain=parent.domain,
        discipline=parent.discipline,
        risk_taking=clip01(parent.risk_taking + random.uniform(-noise, noise)),
        criticism_level=clip01(parent.criticism_level + random.uniform(-noise, noise)),
        collab_preference=clip01(
            parent.collab_preference + random.uniform(-noise, noise)
        ),
        rag_depth=max(1, parent.rag_depth + random.choice([-1, 0, 0, 1])),
        summary_granularity=parent.summary_granularity,
    )


def spawn_mutant(
    parent: LLMScientistAgent,
    llm: LLMClient,
    name_suffix: str = "_mut",
) -> LLMScientistAgent:
    """
    基于单个父代专家生成一个“变体专家”。

    - 继承 name/role/expertise/worldview（名字附加后缀以示区分）
    - genome 通过 mutate_genome 做轻微扰动
    - 初始 reputation 设为中性略低，后续由系统根据表现更新
    """
    new_genome = mutate_genome(parent.cfg.genome)

    cfg = ScientistConfig(
        name=f"{parent.cfg.name}{name_suffix}",
        role=parent.cfg.role,
        expertise=parent.cfg.expertise,
        worldview=parent.cfg.worldview,
        reputation=0.45,
        genome=new_genome,
        use_literature=parent.cfg.use_literature,
    )

    logger.info(
        "Spawned mutant scientist | parent=%s new=%s domain=%s",
        parent.cfg.name,
        cfg.name,
        new_genome.domain.value,
    )
    return LLMScientistAgent(cfg, llm, literature_agent=parent.literature_agent)


# ---------------------------------------------------------------------------
# 3) 杂交：将两个专家基因组合
# ---------------------------------------------------------------------------

def crossover_genome(a: ScientistGenome, b: ScientistGenome) -> ScientistGenome:
    """
    将两个父代的基因组合成一个新的 genome。

    这里采用一个非常简单的策略：
    - domain：优先选择相同领域，如果不同则随机二选一；
    - discipline：二选一；
    - 浮点 trait：取平均值并加入小噪声；
    - rag_depth / summary_granularity：随机从父代中选择。
    """

    def clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    if a.domain == b.domain:
        domain = a.domain
    else:
        domain = random.choice([a.domain, b.domain])

    discipline = random.choice([a.discipline, b.discipline])

    def avg_with_noise(x: float, y: float, noise: float = 0.1) -> float:
        base = (x + y) / 2.0
        return clip01(base + random.uniform(-noise, noise))

    return ScientistGenome(
        domain=domain,
        discipline=discipline,
        risk_taking=avg_with_noise(a.risk_taking, b.risk_taking),
        criticism_level=avg_with_noise(a.criticism_level, b.criticism_level),
        collab_preference=avg_with_noise(
            a.collab_preference, b.collab_preference
        ),
        rag_depth=random.choice([a.rag_depth, b.rag_depth]),
        summary_granularity=random.choice(
            [a.summary_granularity, b.summary_granularity]
        ),
    )


def spawn_offspring(
    parent_a: LLMScientistAgent,
    parent_b: LLMScientistAgent,
    llm: LLMClient,
    name: Optional[str] = None,
) -> LLMScientistAgent:
    """
    基于两个父代专家生成一个“杂交后代”专家。

    - genome：通过 crossover_genome 组合
    - role/expertise/worldview：简单地从父代中二选一（可以后续用 LLM 做更精致的融合）
    """
    new_genome = crossover_genome(parent_a.cfg.genome, parent_b.cfg.genome)

    base_parent = random.choice([parent_a, parent_b])
    cfg = ScientistConfig(
        name=name or f"{parent_a.cfg.name}_{parent_b.cfg.name}_child",
        role=base_parent.cfg.role,
        expertise=base_parent.cfg.expertise,
        worldview=base_parent.cfg.worldview,
        reputation=0.45,
        genome=new_genome,
        use_literature=base_parent.cfg.use_literature,
    )

    logger.info(
        "Spawned offspring scientist | parents=(%s,%s) new=%s domain=%s",
        parent_a.cfg.name,
        parent_b.cfg.name,
        cfg.name,
        new_genome.domain.value,
    )
    # 这里简单沿用 base_parent 的 literature_agent；如果你有全局共享的也可以直接传入
    return LLMScientistAgent(cfg, llm, literature_agent=base_parent.literature_agent)

