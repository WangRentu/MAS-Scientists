from __future__ import annotations

import logging
from typing import Optional, Dict, Any, List

import gradio as gr

from evoverse.literature.unified_search import UnifiedLiteratureSearch
from evoverse.literature.base_client import PaperMetadata
from evoverse.literature.pdf_extractor import get_pdf_extractor
from evoverse.core.llm_client import LLMClient, create_message
from evoverse.config import get_config


logger = logging.getLogger(__name__)
llm = LLMClient()


def _ensure_searcher() -> UnifiedLiteratureSearch:
    """Create a unified searcher with config-driven keys."""
    cfg = get_config().literature
    return UnifiedLiteratureSearch(
        arxiv_enabled=True,
        semantic_scholar_enabled=bool(cfg.semantic_scholar_api_key),
        pubmed_enabled=True,
        semantic_scholar_api_key=cfg.semantic_scholar_api_key,
        pubmed_api_key=cfg.pubmed_api_key,
        pubmed_email=cfg.pubmed_email,
    )


def _paper_to_brief_dict(p: PaperMetadata) -> Dict[str, Any]:
    """Convert PaperMetadata to a lightweight dict for display."""
    return {
        "id": p.primary_identifier,
        "source": p.source.value,
        "title": p.title,
        "year": p.year,
        "authors": ", ".join(p.author_names),
        "doi": p.doi,
        "arxiv_id": p.arxiv_id,
        "pubmed_id": p.pubmed_id,
        "url": p.url,
        "pdf_url": p.pdf_url,
        "abstract": p.abstract,
    }


def _update_outputs(result: Dict[str, Any]) -> tuple[str, str, str, str, str, str, str, str]:
    """Map内部结果dict到8个Gradio输出组件."""
    status_md = f"**状态：** {result.get('status', '')}"
    p = result.get("paper") or {}
    text = (result.get("text") or "")[:20000]  # 避免一次性展示过长文本

    title = p.get("title") or ""
    authors = p.get("authors") or ""
    source_year = ""
    if p:
        src = p.get("source") or ""
        year = p.get("year") or ""
        source_year = f"{src} · {year}" if year else src

    ids = []
    if p.get("doi"):
        ids.append(f"DOI: {p['doi']}")
    if p.get("arxiv_id"):
        ids.append(f"arXiv: {p['arxiv_id']}")
    if p.get("pubmed_id"):
        ids.append(f"PMID: {p['pubmed_id']}")

    return (
        status_md,
        title,
        authors,
        source_year,
        "\n".join(ids),
        p.get("url") or "",
        p.get("abstract") or "",
        text,
    )


def _pick_best_paper(papers: List[PaperMetadata]) -> Optional[PaperMetadata]:
    """Pick the best candidate paper from search results (simple heuristic)."""
    if not papers:
        return None
    # Already ranked by UnifiedLiteratureSearch; just take first
    return papers[0]


def handle_query_mode(
    query: str,
    max_results: int,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, str, str, str]:
    """Handle '通过标题/关键词检索' 模式."""
    progress(0, desc="初始化检索器...")
    searcher = _ensure_searcher()

    query = (query or "").strip()
    if not query:
        return {"status": "请输入检索关键词或论文标题。"}

    progress(0.2, desc="检索候选论文...")
    papers = searcher.search(
        query=query,
        max_results_per_source=max_results,
        total_max_results=max_results,
        deduplicate=True,
        extract_full_text=False,
    )

    if not papers:
        return {"status": "未找到相关论文，请尝试调整关键词。"}

    progress(0.5, desc="选择最佳候选论文...")
    best = _pick_best_paper(papers)
    if not best:
        return {"status": "检索到结果但无法选择合适论文。"}

    extractor = get_pdf_extractor()
    progress(0.7, desc="尝试下载并提取 PDF 正文...")
    text = extractor.extract_paper_text(best)

    progress(1.0, desc="完成")
    result = {
        "status": "成功从检索结果中选取论文并提取文本。",
        "paper": _paper_to_brief_dict(best),
        "text": text or best.abstract or "",
    }
    return _update_outputs(result)


def handle_id_mode(
    identifier: str,
    id_type: str,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, str, str, str]:
    """Handle '通过 DOI / arXiv / PMID' 模式."""
    identifier = (identifier or "").strip()
    if not identifier:
        return {"status": "请输入 DOI / arXiv ID / PMID。"}

    progress(0, desc="初始化检索器...")
    searcher = _ensure_searcher()

    # Normalize identifier a bit
    if id_type == "arxiv" and not identifier.lower().startswith("arxiv:"):
        identifier = identifier.replace("arxiv:", "").strip()

    paper: Optional[PaperMetadata] = None
    progress(0.3, desc="根据标识符查找论文...")

    try:
        if id_type == "doi":
            paper = searcher.search_by_doi(identifier)
        elif id_type == "arxiv":
            paper = searcher.search_by_arxiv_id(identifier)
        else:
            # 对于 PubMed，直接用 search，ID 通常也能搜到
            hits = searcher.search(identifier, max_results_per_source=1, total_max_results=1)
            paper = hits[0] if hits else None
    except Exception as exc:  # noqa: BLE001
        logger.warning("ID 模式检索失败: %s", exc)

    if not paper:
        return {"status": "未找到对应论文，请检查 ID 是否正确。"}

    progress(0.7, desc="尝试下载并提取 PDF 正文...")
    extractor = get_pdf_extractor()
    text = extractor.extract_paper_text(paper)

    progress(1.0, desc="完成")
    result = {
        "status": "成功根据标识符获取论文并提取文本。",
        "paper": _paper_to_brief_dict(paper),
        "text": text or paper.abstract or "",
    }
    return _update_outputs(result)


def handle_pdf_mode(
    file: gr.File,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str, str, str, str, str, str, str]:
    """Handle '上传本地 PDF' 模式."""
    if file is None:
        return _update_outputs({"status": "请上传 PDF 文件。"})

    progress(0.2, desc="读取并提取 PDF 文本...")
    extractor = get_pdf_extractor()
    text = extractor.extract_from_file(file.name)

    if not text:
        return _update_outputs(
            {"status": "未能从 PDF 中提取出有效文本，可能是扫描版或加密文件。", "text": ""}
        )

    progress(1.0, desc="完成")
    result = {
        "status": "成功从本地 PDF 中提取文本。",
        "paper": None,
        "text": text,
    }
    return _update_outputs(result)


def handle_cross_fields(description: str) -> str:
    """
    处理“两个领域交叉”的纯文字描述需求。

    例子：\"计算机科学与流行病学的交叉\"，输出更细化的研究方向、关键问题和建议的检索策略。
    """
    description = (description or "").strip()
    if not description:
        return "请先用几句话描述你想要的交叉方向，例如：计算机视觉与流行病学的交叉。"

    system_prompt = (
        "你是一名善于做跨学科桥接的科研规划师。"
        "用户会给出一个模糊的“两个领域的交叉”需求，请你帮他：\n"
        "1) 明确几个具体的研究问题（尽量可实证/可仿真）；\n"
        "2) 为每个问题给出可能的技术路线；\n"
        "3) 提出建议检索的关键词组合（英文为主，方便查文献）；\n"
        "4) 指出这一交叉方向中值得注意的风险和难点。\n"
        "请用中文分点、分模块清晰输出。"
    )
    user_prompt = (
        f"用户的交叉需求描述如下：\n\n{description}\n\n"
        "请按照上面系统提示中的 1-4 点结构化给出建议。"
    )

    resp = llm.chat(
        [
            create_message("system", system_prompt),
            create_message("user", user_prompt),
        ]
    )
    return resp


def handle_cross_with_paper(paper_text: str, description: str) -> str:
    """
    处理“文献 + 文字描述”的交叉需求。

    用户提供一段论文摘要/正文片段，以及自己的意图描述，例如：
    “我想把强化学习方法用到这篇流行病建模论文上”。
    """
    paper_text = (paper_text or "").strip()
    description = (description or "").strip()

    if not paper_text and not description:
        return "请至少提供论文摘要/片段，或者你的交叉意图描述。"
    if not paper_text:
        return "请粘贴论文的摘要或一小段正文，便于理解你要交叉的对象。"
    if not description:
        return "请简要描述你想和这篇论文做怎样的交叉，例如：用图神经网络改进其建模方法。"

    system_prompt = (
        "你是一名跨学科研究设计专家，擅长在给定论文的基础上，"
        "引入第二个领域（例如计算机科学、控制理论等）提出新的研究方向。"
        "请基于用户提供的论文内容和意图描述：\n"
        "1) 先用 3-5 句话复述论文在做什么；\n"
        "2) 识别出论文中可以被第二领域方法“插入”或“增强”的关键环节；\n"
        "3) 给出 2-3 个具体的交叉研究方向，每个都包含：研究假设、技术路线、可验证的实验设计；\n"
        "4) 为后续查文献提供中英文关键词建议。"
    )
    user_prompt = (
        "【论文内容（可为摘要或正文片段）】\n"
        f"{paper_text[:4000]}\n\n"
        "【你的交叉意图描述】\n"
        f"{description}\n\n"
        "请按照系统提示中的 1-4 点结构化输出，用中文回答。"
    )

    resp = llm.chat(
        [
            create_message("system", system_prompt),
            create_message("user", user_prompt),
        ]
    )
    return resp


def build_interface() -> gr.Blocks:
    """Build a richer Gradio UI for paper ingestion."""
    with gr.Blocks(title="EvoVerse 文献入口 | Paper Ingestion") as demo:
        gr.Markdown(
            "## 📚 EvoVerse 文献入口\n"
            "输入论文标题/关键词、DOI / arXiv / PMID，或直接上传 PDF，系统将尽量获取论文的正文文本并展示基本信息。\n"
            "当前界面主要做 **文献入口 & 文本获取**，之后可以将文本喂给 MAS 科学家小镇进行进一步讨论。"
        )

        with gr.Row():
            with gr.Column(scale=1):
                with gr.Tab("标题 / 关键词", id=0):
                    query = gr.Textbox(
                        label="论文标题或检索关键词",
                        placeholder="例如：Scaling Laws for Neural Language Models",
                    )
                    max_results = gr.Slider(
                        minimum=5,
                        maximum=50,
                        value=20,
                        step=5,
                        label="每个源最多返回多少条结果",
                    )
                    btn_query = gr.Button("检索并提取文本", variant="primary")

                with gr.Tab("DOI / arXiv / PMID", id=1):
                    id_type = gr.Radio(
                        ["doi", "arxiv", "pmid"],
                        value="doi",
                        label="标识符类型",
                    )
                    identifier = gr.Textbox(
                        label="标识符",
                        placeholder="例如：10.48550/arXiv.2001.08361 或 2001.08361",
                    )
                    btn_id = gr.Button("根据标识符获取文本", variant="primary")

                with gr.Tab("上传 PDF", id=2):
                    file = gr.File(
                        label="上传 PDF 文件",
                        file_types=[".pdf"],
                    )
                    btn_pdf = gr.Button("从 PDF 中提取文本", variant="primary")

            with gr.Column(scale=2):
                status = gr.Markdown(label="处理状态")

                with gr.Tab("论文信息"):
                    paper_title = gr.Textbox(label="标题", interactive=False)
                    paper_authors = gr.Textbox(label="作者", interactive=False)
                    paper_source = gr.Textbox(label="来源 & 年份", interactive=False)
                    paper_ids = gr.Textbox(label="ID (DOI / arXiv / PMID)", interactive=False)
                    paper_url = gr.Textbox(label="URL", interactive=False)
                    paper_abstract = gr.Textbox(label="摘要", lines=6, interactive=False)

                with gr.Tab("提取的正文文本"):
                    text_box = gr.Textbox(
                        label="正文文本（前若干字符，避免界面卡顿）",
                        lines=20,
                    )

                with gr.Tab("跨学科需求设计"):
                    with gr.Row():
                        with gr.Column():
                            cross_desc = gr.Textbox(
                                label="两个领域交叉的文字描述",
                                placeholder="例如：计算机科学与流行病学的交叉，用于疫情预测和干预策略优化。",
                                lines=4,
                            )
                            btn_cross_desc = gr.Button(
                                "生成跨学科研究方向建议（纯文字描述）",
                                variant="secondary",
                            )
                        with gr.Column():
                            cross_paper_text = gr.Textbox(
                                label="论文摘要或正文片段",
                                placeholder="将上面提取到的论文摘要/正文片段复制到这里，或者粘贴任意一篇你关心的论文摘要。",
                                lines=6,
                            )
                            cross_paper_desc = gr.Textbox(
                                label="你想和这篇论文做怎样的交叉",
                                placeholder="例如：我想用图神经网络方法改进这篇论文中的传播模型。",
                                lines=3,
                            )
                            btn_cross_paper = gr.Button(
                                "生成“论文 × 领域/方法”的交叉建议",
                                variant="secondary",
                            )

                    cross_output = gr.Markdown(
                        label="跨学科研究建议",
                    )

        btn_query.click(
            fn=handle_query_mode,
            inputs=[query, max_results],
            outputs=[
                status,
                paper_title,
                paper_authors,
                paper_source,
                paper_ids,
                paper_url,
                paper_abstract,
                text_box,
            ],
        )

        btn_id.click(
            fn=handle_id_mode,
            inputs=[identifier, id_type],
            outputs=[
                status,
                paper_title,
                paper_authors,
                paper_source,
                paper_ids,
                paper_url,
                paper_abstract,
                text_box,
            ],
        )

        btn_pdf.click(
            fn=handle_pdf_mode,
            inputs=[file],
            outputs=[
                status,
                paper_title,
                paper_authors,
                paper_source,
                paper_ids,
                paper_url,
                paper_abstract,
                text_box,
            ],
        )

        btn_cross_desc.click(
            fn=handle_cross_fields,
            inputs=[cross_desc],
            outputs=[cross_output],
        )

        btn_cross_paper.click(
            fn=handle_cross_with_paper,
            inputs=[cross_paper_text, cross_paper_desc],
            outputs=[cross_output],
        )

    return demo


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    demo = build_interface()
    demo.launch()


if __name__ == "__main__":
    main()
