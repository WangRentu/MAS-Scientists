# evoverse/agents/scientist_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import logging

from evoverse.agents.base_agent import BaseAgent
from evoverse.core.llm_client import LLMClient, create_message
from evoverse.config import get_config
from pydantic import BaseModel, Field
from evoverse.agents.literature_agent import LiteratureAgent
from evoverse.models.domain import ScientificDomain
from evoverse.agents.memory import (
    AgentMemory,
    TaskEpisodeMemory,
    SocialMemory,
    MetaSelfMemory,
    agent_memory_to_dict,
    agent_memory_from_dict,
)


logger = logging.getLogger(__name__)

class ScientistGenome(BaseModel):
    """可演化的基因参数——对应我们之前讨论的 trait。"""
    domain: ScientificDomain = ScientificDomain.GENERAL # 科学家所在领域枚举
    discipline: str                    # 领域/学科标签，比如 "RL", "Neuro", "GraphTheory"
    risk_taking: float = 0.5           # 冒险程度，影响是否提出激进假设（0~1）
    criticism_level: float = 0.5       # 批判强度，影响在讨论里吐槽/审稿的力度（0~1）
    collab_preference: float = 0.5     # 合作偏好，影响是否引用/接住别人的观点（0~1）
    rag_depth: int = 1                 # RAG 检索深度（hop 数）
    summary_granularity: str = "coarse"  # 记忆写入的粗细程度："coarse" / "fine"

class ScientistConfig(BaseModel):
    """科研智能体的静态配置 + 当前可演化状态。"""
    name: str                          # 智能体名字（唯一标识更友好）
    role: str                          # 角色，例如 "强化学习研究员"
    expertise: str                     # 专长描述（会写进 system prompt）
    worldview: str = "理性、批判性和合作导向"

    # 适应度 / 声誉（演化算法直接用这个）
    reputation: float = 0.5            # 0~1 初始值

    # 简单字符串记忆：沿用原有实现，用于 prompt 中快速回顾最近几条经验
    memory: List[str] = Field(default_factory=list)

    # 结构化长期记忆
    agent_memory: AgentMemory = Field(default_factory=AgentMemory)

    # 可演化基因：用前面的 ScientistGenome 来承载
    genome: ScientistGenome

    # 是否为该科学家启用针对性的文献检索
    use_literature: bool = True

    # 你也可以预留一些自由字段给后续扩展
    # extra_traits: Dict[str, Any] = Field(default_factory=dict)


class LLMScientistAgent(BaseAgent):
    """
    单个 AI 科学家：
    - propose_opinion：围绕问题给出方案
    - critique_opinion：点评别人的方案
    - evolve：根据反馈自主演化自己的 persona
    """

    def __init__(
        self,
        scientist_config: ScientistConfig,
        llm: LLMClient,
        literature_agent: Optional[LiteratureAgent] = None,
    ):
        super().__init__(agent_type="scientist")
        self.cfg = scientist_config
        self.llm = llm
        # 文献 Agent：用于该科学家在会前自行规划检索和做笔记
        self.literature_agent = literature_agent
        # 简单缓存：按问题缓存一次文献笔记，避免每轮都重复检索
        self._literature_cache = {}

    def domain(self) -> ScientificDomain:
        return self.cfg.genome.domain

    # -------------------- 结构化记忆写入助手 --------------------

    def add_task_episode_memory(
        self,
        task_id: str,
        task_title: str,
        task_domains: List[str],
        outcome_score: float,
        outcome_feedback: str = "",
        key_decisions: Optional[List[str]] = None,
        key_contributions: Optional[List[str]] = None,
        self_reflection: str = "",
    ) -> None:
        """
        记录一次任务级别的记忆。

        目前在镇会议结束时由 orchestrator 调用：
        - task_id: 可以是问题的 hash 或外部任务 ID
        - task_title: 问题本身
        - task_domains: 该任务涉及的领域列表（字符串形式）
        - outcome_score: 通常可使用当前专家的 reputation 或任务总体评分
        """
        episode = TaskEpisodeMemory(
            task_id=task_id,
            task_title=task_title,
            task_domains=task_domains,
            task_embedding=[],  # 目前暂不计算 embedding，留空列表占位
            role=self.cfg.role,
            key_decisions=key_decisions or [],
            key_contributions=key_contributions or [],
            outcome_score=outcome_score,
            outcome_feedback=outcome_feedback,
            self_reflection=self_reflection,
        )
        self.cfg.agent_memory.episodes.append(episode)

    def update_social_memory(self, collaborator_id: str, score: float) -> None:
        """
        根据一次合作/互评更新社会记忆。

        在 TownOrchestrator 中，每次互评时可以调用：
            critic.update_social_memory(target_name, score)
        """
        social_list = self.cfg.agent_memory.social
        for sm in social_list:
            if sm.collaborator_id == collaborator_id:
                # 更新聚合统计
                total = sm.avg_outcome_score * sm.num_projects + score
                sm.num_projects += 1
                sm.avg_outcome_score = total / sm.num_projects
                sm.last_collab_time = sm.last_collab_time.__class__.utcnow()
                return

        social_list.append(
            SocialMemory(
                collaborator_id=collaborator_id,
                num_projects=1,
                avg_outcome_score=score,
                notes="",
            )
        )

    def append_meta_strategy_update(self, note: str) -> None:
        """
        在专家发生明显演化（例如触发 evolve）时，追加一条策略演化记录。
        """
        if self.cfg.agent_memory.meta is None:
            self.cfg.agent_memory.meta = MetaSelfMemory(
                strengths=[],
                weaknesses=[],
                typical_failures=[],
                strategy_updates=[],
            )
        meta = self.cfg.agent_memory.meta
        meta.strategy_updates.append(note)
        meta.last_updated = meta.last_updated.__class__.utcnow()

    # ========================================================================
    # STATE PERSISTENCE (deep integration with BaseAgent)
    # ========================================================================

    def get_state(self):  # type: ignore[override]
        """
        将结构化记忆持久化到 BaseAgent.state_data 中，然后调用基类的 get_state。

        这样，上层只要拿到 AgentState 并做磁盘序列化，就自动包含了专家的记忆。
        """
        # 先把当前 AgentMemory 序列化进 state_data
        self.save_state_data("agent_memory", agent_memory_to_dict(self.cfg.agent_memory))
        return super().get_state()

    def restore_state(self, state):  # type: ignore[override]
        """
        从 AgentState 中恢复 BaseAgent 状态，并尝试反序列化结构化记忆。
        """
        super().restore_state(state)
        mem_dict = self.get_state_data("agent_memory")
        if isinstance(mem_dict, dict):
            try:
                self.cfg.agent_memory = agent_memory_from_dict(mem_dict)
            except Exception:
                logger.warning(
                    "Failed to restore agent_memory for %s from state_data",
                    self.cfg.name,
                    exc_info=True,
                )

    # -------------------- 内部：system prompt --------------------

    def _system_prompt(self) -> str:
        genome = self.cfg.genome
        collab_style = (
            "积极衔接并引用其他科学家的观点"
            if genome.collab_preference >= 0.6
            else "保持独立判断，仅在必要时引用他人观点"
        )
        risk_style = (
            "敢于提出激进假设并拥抱高风险方案"
            if genome.risk_taking >= 0.6
            else "优先稳健、可验证的方案"
        )
        return (
            f"你是一名 AI 科学家，名字叫 {self.cfg.name}，角色是：{self.cfg.role}。\n"
            f"你的研究专长：{self.cfg.expertise}。\n"
            f"你的世界观与风格：{self.cfg.worldview}。\n"
            f"所属学科：{genome.domain.value}；风险偏好 {genome.risk_taking:.2f}，"
            f"批判强度 {genome.criticism_level:.2f}，合作偏好 {genome.collab_preference:.2f}。\n"
            f"请在讨论中{risk_style}，并{collab_style}。\n"
        )

    # -------------------- 行为 1：提出观点 --------------------

    def _prepare_literature_context(self, question: str) -> str:
        """
        针对当前科学家的身份与专长，对研究问题做一次“自助式”文献检索：
        1) 先让 LLM 帮该 agent 拆分子问题并产出检索关键词；
        2) 再调用 LiteratureAgent 逐个检索并生成简短笔记；
        3) 返回一段简明的“个人文献笔记”，在最终发言前拼接进 prompt。
        """
        if not self.cfg.use_literature or not self.literature_agent:
            return ""

        # 缓存：同一个问题只检索一次
        if question in self._literature_cache:
            return self._literature_cache[question]

        import json

        genome = self.cfg.genome
        planning_prompt = (
            "你将以自己的科研身份在会前阅读并理解下面的科研问题，"
            "请先从你的视角拆分出 1-3 个你最关注的子问题，并为每个子问题设计检索关键词。\n\n"
            f"【你的角色】{self.cfg.role}；专长：{self.cfg.expertise}；学科：{genome.discipline}\n\n"
            f"【待研究的问题】\n{question}\n\n"
            "请只返回 JSON，格式如下：\n"
            '{"sub_questions":[{"text":"子问题描述","keywords":["keyword1","keyword2"]}, ...]}\n'
            "其中 keywords 尽量使用英文技术术语，方便检索 arXiv / PubMed 等学术数据库。"
        )

        logger.info("Scientist %s planning literature search", self.cfg.name)
        resp = self.llm.chat(
            [
                {"role": "system", "content": self._system_prompt()},
                {"role": "user", "content": planning_prompt},
            ]
        )

        try:
            data = json.loads(resp.strip().strip("` "))
            sub_questions = data.get("sub_questions") or []
        except Exception:
            logger.warning(
                "Scientist %s literature planning parsing failed | response=%s",
                self.cfg.name,
                resp,
                exc_info=True,
            )
            return ""

        if not sub_questions:
            return ""

        blocks: List[str] = []
        # 检索结果数量可以粗略受 rag_depth 影响
        max_results_per_query = max(2, min(5, genome.rag_depth * 2))

        for idx, sq in enumerate(sub_questions, start=1):
            text = (sq.get("text") or "").strip()
            keywords = [k for k in (sq.get("keywords") or []) if isinstance(k, str)]
            if not keywords:
                continue
            query = " ".join(keywords)

            try:
                papers = self.literature_agent.search_and_summarize(
                    query=query,
                    max_results=max_results_per_query,
                )
            except Exception:
                logger.warning(
                    "Scientist %s literature search failed | query=%s",
                    self.cfg.name,
                    query,
                    exc_info=True,
                )
                continue

            if not papers:
                continue

            lines: List[str] = []
            if text:
                lines.append(f"[子问题 {idx}] {text}")
            else:
                lines.append(f"[子问题 {idx}]（未命名子问题）")
            lines.append(f"检索关键词: {', '.join(keywords)}")
            lines.append("代表性文献：")

            for j, p in enumerate(papers[:2], start=1):
                title = p.get("title", "") or ""
                year = p.get("year") or ""
                summary = p.get("summary") or p.get("abstract") or ""
                summary = summary.strip()
                if len(summary) > 160:
                    summary = summary[:160] + "..."
                lines.append(f"- ({year}) {title}：{summary}")

            blocks.append("\n".join(lines))

        if not blocks:
            return ""

        note = (
            "在正式给出方案前，你已经根据自己的视角快速检索并阅读了一些相关文献。"
            "以下是你为自己准备的简要文献笔记，请在回答时结合这些内容思考，"
            "但不要逐条复述。\n\n"
            + "\n\n".join(blocks)
            + "\n\n"
        )

        self._literature_cache[question] = note
        return note

    def propose_opinion(self, question: str, context_summary: str = "") -> str:
        genome = self.cfg.genome
        history_part = ""
        if self.cfg.memory:
            history_part = (
                "你之前的重要结论或经验包括：\n- "
                + "\n- ".join(self.cfg.memory[-3:])
                + "\n\n"
            )
        rag_hint = (
            f"如需引用外部知识，可以进行 {genome.rag_depth} 跳检索并在行末标注来源。"
            if genome.rag_depth > 1
            else "优先基于当前讨论与记忆作答，必要时再引用外部知识。"
        )
        tone_hint = (
            "允许提出高风险高收益的设想，但要清楚标注验证路径。"
            if genome.risk_taking >= 0.6
            else "请保持稳健、可落地的方案，并给出保守备选。"
        )
        collab_hint = (
            "请引用其他科学家的观点并说明如何协同推进。"
            if genome.collab_preference >= 0.6
            else "请独立阐述观点，并在结尾点出可能的合作接口。"
        )
        town_cfg = get_config().town
        literature_part = self._prepare_literature_context(question)
        user_prompt = (
            f"{history_part}"
            f"{literature_part}"
            f"当前 AI 科学家小镇正在讨论的问题是：\n{question}\n\n"
            f"{'已有讨论摘要：' + context_summary if context_summary else ''}\n"
            "请你用 3-5 条要点提出自己的看法，包含：\n"
            "1) 关键假设；2) 具体研究或实验路径；3) 潜在风险与不确定性；\n"
            f"回答限制在 {town_cfg.max_opinion_length} 字以内。\n"
            f"{rag_hint}\n{tone_hint}\n{collab_hint}"
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
        memory_entry = resp.strip()
        if genome.summary_granularity != "fine":
            memory_entry = memory_entry.split("\n")[0]
        if memory_entry:
            self.cfg.memory.append(memory_entry)
        return resp

    # -------------------- 行为 2：点评他人 --------------------

    def critique_opinion(self, question: str, target_name: str, target_opinion: str) -> str:
        genome = self.cfg.genome
        critique_tone = (
            "保持犀利直接，抓住关键漏洞或假设问题"
            if genome.criticism_level >= 0.6
            else "保持建设性与礼貌，强调改进建议"
        )
        user_prompt = (
            f"现在你要点评另一位科学家 {target_name} 对同一问题的方案。\n"
            f"问题是：{question}\n\n"
            f"对方的方案如下：\n{target_opinion}\n\n"
            "请你从【优点】【缺点】【可行性】三个角度进行评议，"
            "并给出 0~1 的综合评分。\n"
            f"{critique_tone}，并结合你的合作偏好（{genome.collab_preference:.2f}）提出可行的联合作业建议。\n"
            "最后只用 JSON 返回：{\"comment\": \"...\", \"score\": 0.xx}"
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
            "如有必要，也可以调整你的基因参数（discipline、risk_taking、criticism_level、"
            "collab_preference、rag_depth、summary_granularity）。\n"
            '只用 JSON 返回：{"expertise": "...", "worldview": "...", "genome": { ... }}，其中 genome 字段可选。'
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
            genome_updates = data.get("genome") or {}
            for field_name, value in genome_updates.items():
                if hasattr(self.cfg.genome, field_name) and value is not None:
                    setattr(self.cfg.genome, field_name, value)
            # 进化之后把声望拉回中性
            self.cfg.reputation = 0.5
            # 记录一次策略演化
            self.append_meta_strategy_update(
                f"Evolution triggered due to low reputation; feedback summary length={len(feedback_summary)}."
            )
            logger.info(
                "Scientist %s evolved | new_expertise=%s new_worldview=%s",
                self.cfg.name,
                self.cfg.expertise,
                self.cfg.worldview,
            )
        except Exception:
            logger.warning("Scientist %s evolution parsing failed | response=%s", self.cfg.name, resp, exc_info=True)
