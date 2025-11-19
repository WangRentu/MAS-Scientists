"""
Literature management for EvoVerse.

Provides:
- arXiv API client
- Semantic Scholar API client
- PubMed API client
- PDF download and extraction
- Unified literature search
- Citation/reference helpers
- Disk-based caching
"""

from evoverse.literature.base_client import (
    BaseLiteratureClient,
    PaperMetadata,
    PaperSource,
    Author
)
from evoverse.literature.cache import (
    LiteratureCache,
    get_cache,
    reset_cache
)
from evoverse.literature.arxiv_client import ArxivClient
from evoverse.literature.semantic_scholar import SemanticScholarClient
from evoverse.literature.pubmed_client import PubMedClient
from evoverse.literature.pdf_extractor import (
    PDFExtractor,
    get_pdf_extractor,
    reset_pdf_extractor
)
from evoverse.literature.unified_search import UnifiedLiteratureSearch
from evoverse.literature.citations import (
    CitationFormatter,
    papers_to_bibtex,
    papers_to_ris
)
from evoverse.literature.reference_manager import ReferenceManager

__all__ = [
    "BaseLiteratureClient",
    "PaperMetadata",
    "PaperSource",
    "Author",
    "LiteratureCache",
    "get_cache",
    "reset_cache",
    "ArxivClient",
    "SemanticScholarClient",
    "PubMedClient",
    "PDFExtractor",
    "get_pdf_extractor",
    "reset_pdf_extractor",
    "UnifiedLiteratureSearch",
    "CitationFormatter",
    "papers_to_bibtex",
    "papers_to_ris",
    "ReferenceManager",
]
