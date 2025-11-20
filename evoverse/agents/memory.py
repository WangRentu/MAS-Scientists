from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional
from pathlib import Path
import json


@dataclass
class TaskEpisodeMemory:
    """
    单次任务/项目的记忆片段。
    用于记录这个专家在某个任务中的角色、关键决策、结果和自我反思。
    """

    task_id: str
    task_title: str
    task_domains: List[str]
    task_embedding: List[float]

    role: str                      # 比如 "pi", "modeling_expert", "reviewer"
    key_decisions: List[str]       # 这次做过的关键决策摘要
    key_contributions: List[str]   # 自己提出的关键观点/方法

    outcome_score: float           # 0~1 项目整体质量
    outcome_feedback: str          # 来自系统/LLM 的评语
    self_reflection: str           # 专家自己的短反思

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SemanticSkillMemory:
    """
    语义层技能记忆：这个专家在哪些领域/方法上积累了什么经验性规则。
    """

    domain: str                    # "epidemiology"
    methods: List[str]             # ["rl", "seir_modeling"]
    heuristic_rules: List[str]     # 经验性规则的自然语言描述
    tool_preferences: List[str]    # 喜欢用哪些工具和为什么
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class SocialMemory:
    """
    社会记忆：与其他专家合作的历史与体验。
    """

    collaborator_id: str
    num_projects: int
    avg_outcome_score: float
    notes: str                     # 对合作体验的简短文字描述
    last_collab_time: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MetaSelfMemory:
    """
    自我认知记忆：总结自身强项、弱点、常见失败模式与策略演化。
    """

    strengths: List[str]           # "strong in causal inference for epidemiology"
    weaknesses: List[str]          # "struggles with very sparse time series"
    typical_failures: List[str]    # 容易犯的错误模式
    strategy_updates: List[str]    # 自己策略的进化记录
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class AgentMemory:
    """
    专家整体记忆容器。
    当前放在 ScientistConfig 上，由 LLMScientistAgent 读写。
    """

    episodes: List[TaskEpisodeMemory] = field(default_factory=list)
    skills: List[SemanticSkillMemory] = field(default_factory=list)
    social: List[SocialMemory] = field(default_factory=list)
    meta: Optional[MetaSelfMemory] = None


def agent_memory_to_dict(memory: AgentMemory) -> dict:
    """
    将 AgentMemory 序列化为纯 dict，便于放入 AgentState.data 持久化。
    使用 dataclasses.asdict 递归展开。
    """
    return asdict(memory)


def agent_memory_from_dict(data: dict) -> AgentMemory:
    """
    从 dict 反序列化为 AgentMemory。
    假设字段结构与 agent_memory_to_dict 的输出一致。
    """
    episodes_data = data.get("episodes", []) or []
    skills_data = data.get("skills", []) or []
    social_data = data.get("social", []) or []
    meta_data = data.get("meta")

    episodes: List[TaskEpisodeMemory] = []
    for e in episodes_data:
        episodes.append(
            TaskEpisodeMemory(
                task_id=e.get("task_id", ""),
                task_title=e.get("task_title", ""),
                task_domains=e.get("task_domains", []) or [],
                task_embedding=e.get("task_embedding", []) or [],
                role=e.get("role", ""),
                key_decisions=e.get("key_decisions", []) or [],
                key_contributions=e.get("key_contributions", []) or [],
                outcome_score=e.get("outcome_score", 0.0),
                outcome_feedback=e.get("outcome_feedback", ""),
                self_reflection=e.get("self_reflection", ""),
                timestamp=e.get("timestamp", datetime.utcnow()),
            )
        )

    skills: List[SemanticSkillMemory] = []
    for s in skills_data:
        skills.append(
            SemanticSkillMemory(
                domain=s.get("domain", ""),
                methods=s.get("methods", []) or [],
                heuristic_rules=s.get("heuristic_rules", []) or [],
                tool_preferences=s.get("tool_preferences", []) or [],
                last_updated=s.get("last_updated", datetime.utcnow()),
            )
        )

    socials: List[SocialMemory] = []
    for sm in social_data:
        socials.append(
            SocialMemory(
                collaborator_id=sm.get("collaborator_id", ""),
                num_projects=sm.get("num_projects", 0),
                avg_outcome_score=sm.get("avg_outcome_score", 0.0),
                notes=sm.get("notes", ""),
                last_collab_time=sm.get("last_collab_time", datetime.utcnow()),
            )
        )

    meta: Optional[MetaSelfMemory] = None
    if meta_data:
        meta = MetaSelfMemory(
            strengths=meta_data.get("strengths", []) or [],
            weaknesses=meta_data.get("weaknesses", []) or [],
            typical_failures=meta_data.get("typical_failures", []) or [],
            strategy_updates=meta_data.get("strategy_updates", []) or [],
            last_updated=meta_data.get("last_updated", datetime.utcnow()),
        )

    return AgentMemory(
        episodes=episodes,
        skills=skills,
        social=socials,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# 磁盘持久化：默认存储在项目根目录下的 `.agent_memory/` 目录
# ---------------------------------------------------------------------------

DEFAULT_MEMORY_DIR = Path(".agent_memory")


def save_agent_memory(
    agent_id: str,
    memory: AgentMemory,
    base_dir: Path | None = None,
) -> Path:
    """
    将 AgentMemory 序列化为 JSON 并写入磁盘。

    - 默认存储路径：`.agent_memory/{agent_id}.json`
    - base_dir 可选，便于测试或自定义目录。
    """
    dir_path = base_dir or DEFAULT_MEMORY_DIR
    dir_path.mkdir(parents=True, exist_ok=True)

    path = dir_path / f"{agent_id}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(agent_memory_to_dict(memory), f, ensure_ascii=False, indent=2)

    return path


def load_agent_memory(
    agent_id: str,
    base_dir: Path | None = None,
) -> Optional[AgentMemory]:
    """
    从 `.agent_memory/{agent_id}.json` 读取并反序列化为 AgentMemory。

    如果文件不存在，返回 None。
    """
    dir_path = base_dir or DEFAULT_MEMORY_DIR
    path = dir_path / f"{agent_id}.json"

    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return None

    return agent_memory_from_dict(data)

