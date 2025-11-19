"""
LiteratureAgent - MVP 文献检索 Agent

封装 UnifiedLiteratureSearch，提供：
- search_and_summarize：按查询检索文献并用 LLM 做简要摘要
"""

from typing import Any, Dict, List, Optional
import logging

from evoverse.agents.base_agent import BaseAgent
from evoverse.literature.unified_search import UnifiedLiteratureSearch
from evoverse.literature.base_client import PaperMetadata, PaperAnalysis
from evoverse.core.llm_client import LLMClient
import hashlib
import time


logger = logging.getLogger(__name__)


class LiteratureAgent(BaseAgent):
    """MVP 版本文献 Agent。"""

    def __init__(
        self,
        agent_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        llm_client: Optional[LLMClient] = None,
        searcher: Optional[UnifiedLiteratureSearch] = None,
    ):
        super().__init__(agent_id=agent_id, agent_type="LiteratureAgent", config=config)
        self.llm = llm_client or LLMClient(max_history=16)
        self.searcher = searcher or UnifiedLiteratureSearch(
            semantic_scholar_enabled=False
        )

    def search_and_summarize(
        self,
        query: str,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        检索文献并为每篇生成简短摘要。

        返回的每个元素是经过精简的 dict，方便后续序列化和 prompt 使用：
        {
            "title": ...,
            "authors": [...],
            "year": ...,
            "abstract": ...,
            "summary": ...,
            "source": ...,
            "primary_id": ...
        }
        """
        logger.info("LiteratureAgent searching papers: %s", query)
        papers = self.searcher.search(
            query=query,
            max_results_per_source=max_results,
            total_max_results=max_results,
            deduplicate=True,
            extract_full_text=True,
        )
        for i, p in enumerate(papers, start=1):
            logger.info(
                "Paper %d: [%s] %s",
                i,
                getattr(p, "primary_identifier", getattr(p, "id", "")),
                getattr(p, "title", ""),
            )

        simplified: List[Dict[str, Any]] = []
        for p in papers:
            simplified.append(self._simplify_paper(p))

        # 用 LLM 生成摘要和深度分析
        for item in simplified:
            item["summary"] = self._summarize_paper(item)
            # 添加深度分析选项（可选）
            if self.config.get("enable_deep_analysis", False):
                deep_analysis = self.analyze_paper_deep(item)
                item["deep_analysis"] = deep_analysis.to_dict()

        logger.info("LiteratureAgent retrieved %d papers", len(simplified))
        return simplified

    def _simplify_paper(self, paper: Any) -> Dict[str, Any]:
        """将 PaperMetadata 压缩成易于传输和序列化的字典，并补充统一 ID。"""
        # 1. 取基础字段
        title = getattr(paper, "title", "") or ""
        authors = getattr(paper, "authors", []) or []
        year = getattr(paper, "year", None)
        abstract = getattr(paper, "abstract", None) or getattr(paper, "summary", "")
        full_text = getattr(paper, "full_text", None) or ""
        source = getattr(paper, "source", None)
        primary_id = getattr(paper, "primary_identifier", None)

        # 2. 规范作者列表为 List[str]
        norm_authors: List[str] = []
        for a in authors:
            if isinstance(a, str):
                norm_authors.append(a)
            else:
                name = getattr(a, "name", None) or str(a)
                norm_authors.append(name)

        # 3. 规范 source 为字符串
        if source is None:
            source_str = "unknown"
        else:
            source_str = getattr(source, "value", str(source)).lower()

        # 4. 构造统一逻辑 ID（和 Kosmos 一样的思路）
        global_id = primary_id or f"unknown:{hashlib.sha1(title.encode()).hexdigest()[:16]}"

        # 5. 返回带 id 的简化结构
        return {
            "id": global_id,          # 🔴 统一的内部 ID（后面所有地方都用它）
            "source": source_str,     # 文献来源：arxiv / pubmed / semanticscholar / unknown
            "primary_id": primary_id, # 原始 source 的主键，方便 debug 或外部跳转
            "title": title,
            "authors": norm_authors,
            "year": year,
            # 对齐 Kosmos：保留全文，摘要阶段优先使用
            "full_text": full_text,
            "abstract": abstract,
            "summary": "",            # 这里先留空，后面 _summarize_paper 再填
        }
    

    def _summarize_paper(self, paper: Dict[str, Any]) -> str:
        """使用 LLM 为单篇文献生成结构化分析摘要。"""
        title = paper.get("title", "")
        full_text = paper.get("full_text") or ""
        abstract = paper.get("abstract", "")

        # 对齐 Kosmos：优先使用全文，其次摘要，如果都没有就直接返回空
        if full_text:
            base_text = f"全文（截断）：\n{full_text[:5000]}"
        elif abstract:
            base_text = f"摘要：\n{abstract}"
        else:
            return ""

        # 构建专业化的总结 prompt
        text = f"标题：{title}\n\n{base_text}"

        prompt = f"""分析这篇科学论文并提供全面的总结。

{text}

请提供结构化分析，包括：
1. 执行摘要（2-3句话）
2. 关键发现（3-5个主要结果）
3. 研究方法（研究方法概述）
4. 重要性（科学意义）
5. 局限性（弱点或注意事项）

请用中文回复，直接给出分析内容。"""

        try:
            messages = [
                {
                    "role": "system",
                    "content": "你是资深的科学文献分析师。请提供全面、准确的分析。",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            summary = self.llm.chat(messages)
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM summarize paper failed: %s", exc)
            summary = f"论文《{title}》的分析暂时不可用。"

        return summary

    def analyze_paper_deep(self, paper: Dict[str, Any]) -> PaperAnalysis:
        """
        深度分析单篇论文，返回结构化的 PaperAnalysis。

        Args:
            paper: 简化后的论文数据字典

        Returns:
            PaperAnalysis: 完整的论文分析结果
        """
        start_time = time.time()

        # 构建论文文本
        title = paper.get("title", "")
        full_text = paper.get("full_text", "")
        abstract = paper.get("abstract", "")

        text = f"标题：{title}\n\n"
        if full_text:
            text += f"全文（截断）：\n{full_text[:5000]}"
        elif abstract:
            text += f"摘要：\n{abstract}"
        else:
            text += "摘要：不可用"

        # 构建分析prompt
        prompt = f"""分析这篇科学论文并提供全面的总结。

{text}

请提供结构化分析，包括：
1. 执行摘要（2-3句话）
2. 关键发现（3-5个主要结果，用列表形式）
3. 方法论（研究方法概述）
4. 重要性（科学意义）
5. 局限性（弱点或注意事项，用列表形式）
6. 置信度（0-1之间的数值，表示分析的信心水平）

请返回JSON格式的响应。
"""

        # 定义输出模式
        output_schema = {
            "type": "object",
            "properties": {
                "executive_summary": {"type": "string"},
                "key_findings": {
                    "type": "array",
                    "items": {"type": "object", "properties": {"finding": {"type": "string"}}}
                },
                "methodology": {"type": "object"},
                "significance": {"type": "string"},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "confidence_score": {"type": "number"}
            }
        }

        try:
            # 使用结构化生成
            analysis = self.llm.generate_structured(
                prompt=prompt,
                output_schema=output_schema,
                system="你是资深的科学文献分析师。请提供全面、准确的分析。",
                max_tokens=2048
            )

            result = PaperAnalysis(
                paper_id=paper.get("id", ""),
                executive_summary=analysis.get("executive_summary", ""),
                key_findings=analysis.get("key_findings", []),
                methodology=analysis.get("methodology", {}),
                significance=analysis.get("significance", ""),
                limitations=analysis.get("limitations", []),
                confidence_score=analysis.get("confidence_score", 0.5),
                analysis_time=time.time() - start_time
            )

            logger.info(f"完成论文深度分析: {title}")
            return result

        except Exception as e:
            logger.error(f"论文深度分析失败: {e}")
            # 返回默认分析结果
            return PaperAnalysis(
                paper_id=paper.get("id", ""),
                executive_summary=f"论文《{title}》的分析暂时不可用。",
                key_findings=[],
                methodology={},
                significance="",
                limitations=[],
                confidence_score=0.0,
                analysis_time=time.time() - start_time
            )

    def analyze_corpus(self, papers: List[Dict[str, Any]], generate_insights: bool = True) -> Dict[str, Any]:
        """
        分析论文语料库，提取共同主题和趋势。

        Args:
            papers: 论文列表
            generate_insights: 是否生成高层洞见

        Returns:
            语料库分析结果
        """
        analysis = {
            "corpus_size": len(papers),
            "common_themes": [],
            "methodological_trends": {},
            "research_gaps": [],
            "temporal_distribution": {},
            "field_distribution": {}
        }

        if not papers:
            return analysis

        # 统计年份分布
        years = {}
        fields = {}

        for paper in papers:
            year = paper.get("year")
            if year:
                years[year] = years.get(year, 0) + 1

            # 这里可以添加更复杂的领域分析
            # 暂时简化处理

        analysis["temporal_distribution"] = years
        analysis["field_distribution"] = fields

        if generate_insights:
            # 生成高层洞见
            insights_prompt = f"""分析这组{len(papers)}篇论文的整体特征：

论文标题列表：
{chr(10).join([f"- {p.get('title', '')}" for p in papers[:10]])}

请识别：
1. 共同研究主题
2. 方法论趋势
3. 潜在研究空白

请用中文回复。"""

            try:
                insights = self.llm.chat([
                    {"role": "system", "content": "你是科研趋势分析师"},
                    {"role": "user", "content": insights_prompt}
                ])

                # 简单解析insights（可以改进）
                analysis["insights_summary"] = insights

            except Exception as e:
                logger.error(f"生成洞见失败: {e}")
                analysis["insights_summary"] = "洞见生成失败"

        return analysis
