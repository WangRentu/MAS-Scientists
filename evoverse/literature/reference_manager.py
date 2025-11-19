"""
Reference management with advanced deduplication.

Provides:
- Reference collection management
- Multi-level deduplication (DOI, arXiv, PubMed, fuzzy title)
- Citation relationship tracking
- Library export to multiple formats
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
from difflib import SequenceMatcher
import hashlib

from evoverse.literature.base_client import PaperMetadata
from evoverse.literature.citations import CitationFormatter, papers_to_bibtex, papers_to_ris

logger = logging.getLogger(__name__)


class ReferenceManager:
    """
    Manage collections of references with deduplication.

    Provides storage, search, deduplication, and export capabilities.
    """

    def __init__(
        self,
        storage_path: Optional[str] = None,
        auto_deduplicate: bool = True
    ):
        """
        Initialize reference manager.

        Args:
            storage_path: Path to persistent storage file
            auto_deduplicate: Whether to auto-deduplicate on add

        Example:
            ```python
            manager = ReferenceManager(
                storage_path="my_library.json",
                auto_deduplicate=True
            )
            ```
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.auto_deduplicate = auto_deduplicate

        # Storage: ref_id -> PaperMetadata
        self.references: Dict[str, PaperMetadata] = {}

        # Citation relationships: citing_id -> [cited_ids]
        self.citation_links: Dict[str, List[str]] = {}

        # Deduplication engine
        self.dedup_engine = DeduplicationEngine()

        # Load from storage if exists
        if self.storage_path and self.storage_path.exists():
            self._load_from_storage()

        logger.info(f"Initialized ReferenceManager (refs={len(self.references)})")

    def add_reference(self, paper: PaperMetadata) -> str:
        """
        Add single reference.

        Args:
            paper: PaperMetadata object

        Returns:
            Reference ID

        Example:
            ```python
            ref_id = manager.add_reference(paper)
            ```
        """
        # Check for duplicates if auto-dedup enabled
        if self.auto_deduplicate:
            existing_id = self._find_duplicate(paper)
            if existing_id:
                logger.debug(f"Duplicate found, merging with {existing_id}")
                self._merge_papers(existing_id, paper)
                return existing_id

        # Generate reference ID
        ref_id = self._generate_ref_id(paper)

        # Store reference
        self.references[ref_id] = paper

        # Save to storage
        if self.storage_path:
            self._save_to_storage()

        logger.debug(f"Added reference: {ref_id}")
        return ref_id

    def add_references(self, papers: List[PaperMetadata]) -> List[str]:
        """
        Batch add references.

        Args:
            papers: List of PaperMetadata objects

        Returns:
            List of reference IDs

        Example:
            ```python
            ref_ids = manager.add_references(papers)
            print(f"Added {len(ref_ids)} references")
            ```
        """
        ref_ids = []

        for paper in papers:
            ref_id = self.add_reference(paper)
            ref_ids.append(ref_id)

        logger.info(f"Added {len(ref_ids)} references")
        return ref_ids

    def get_reference(self, ref_id: str) -> Optional[PaperMetadata]:
        """
        Retrieve reference by ID.

        Args:
            ref_id: Reference identifier

        Returns:
            PaperMetadata or None if not found
        """
        return self.references.get(ref_id)

    def search_references(
        self,
        query: str,
        fields: List[str] = ["title", "authors", "abstract"]
    ) -> List[PaperMetadata]:
        """
        Search within reference collection.

        Args:
            query: Search query
            fields: Fields to search in

        Returns:
            List of matching papers

        Example:
            ```python
            results = manager.search_references(
                "machine learning",
                fields=["title", "abstract"]
            )
            ```
        """
        query_lower = query.lower()
        matches = []

        for paper in self.references.values():
            # Check each field
            for field in fields:
                if field == "title" and paper.title:
                    if query_lower in paper.title.lower():
                        matches.append(paper)
                        break
                elif field == "authors" and paper.authors:
                    author_names = " ".join([a.name for a in paper.authors]).lower()
                    if query_lower in author_names:
                        matches.append(paper)
                        break
                elif field == "abstract" and paper.abstract:
                    if query_lower in paper.abstract.lower():
                        matches.append(paper)
                        break

        logger.info(f"Search '{query}' found {len(matches)} results")
        return matches

    def deduplicate_references(
        self,
        strategy: str = "comprehensive"
    ) -> Dict[str, Any]:
        """
        Remove duplicate references.

        Args:
            strategy: Deduplication strategy
                - "doi": Only DOI matching
                - "fuzzy": Fuzzy title matching
                - "comprehensive": Multi-level deduplication

        Returns:
            Deduplication report

        Example:
            ```python
            report = manager.deduplicate_references(strategy="comprehensive")
            print(f"Removed {report['duplicates_removed']} duplicates")
            ```
        """
        papers_list = list(self.references.values())
        original_count = len(papers_list)

        if strategy == "doi":
            unique_papers, duplicate_groups = self.dedup_engine.deduplicate_by_doi(papers_list)
        elif strategy == "fuzzy":
            unique_papers, duplicate_groups = self.dedup_engine.deduplicate_by_title(
                papers_list,
                threshold=0.9
            )
        else:  # comprehensive
            unique_papers, duplicate_groups = self.dedup_engine.comprehensive_deduplication(
                papers_list
            )

        # Rebuild references dict
        self.references = {}
        for paper in unique_papers:
            ref_id = self._generate_ref_id(paper)
            self.references[ref_id] = paper

        # Save changes
        if self.storage_path:
            self._save_to_storage()

        duplicates_removed = original_count - len(unique_papers)

        report = {
            "original_count": original_count,
            "unique_count": len(unique_papers),
            "duplicates_removed": duplicates_removed,
            "duplicate_groups": len(duplicate_groups),
            "strategy": strategy
        }

        logger.info(f"Deduplication complete: removed {duplicates_removed} duplicates")
        return report

    def merge_duplicates(
        self,
        ref_id1: str,
        ref_id2: str,
        keep_id: str
    ):
        """
        Manually merge two references.

        Args:
            ref_id1: First reference ID
            ref_id2: Second reference ID
            keep_id: ID to keep (ref_id1 or ref_id2)

        Example:
            ```python
            manager.merge_duplicates("ref1", "ref2", keep_id="ref1")
            ```
        """
        if ref_id1 not in self.references or ref_id2 not in self.references:
            logger.error("Reference not found for merging")
            return

        if keep_id not in [ref_id1, ref_id2]:
            logger.error("keep_id must be one of the reference IDs")
            return

        discard_id = ref_id2 if keep_id == ref_id1 else ref_id1

        # Merge papers
        kept_paper = self.references[keep_id]
        discarded_paper = self.references[discard_id]

        merged_paper = self.dedup_engine.merge_paper_metadata([kept_paper, discarded_paper])

        # Update references
        self.references[keep_id] = merged_paper
        del self.references[discard_id]

        # Update citation links
        if discard_id in self.citation_links:
            if keep_id in self.citation_links:
                self.citation_links[keep_id].extend(self.citation_links[discard_id])
            else:
                self.citation_links[keep_id] = self.citation_links[discard_id]
            del self.citation_links[discard_id]

        # Save changes
        if self.storage_path:
            self._save_to_storage()

        logger.info(f"Merged {discard_id} into {keep_id}")

    def export_library(
        self,
        output_file: str,
        format: str = "bibtex"
    ):
        """
        Export reference library.

        Args:
            output_file: Output file path
            format: Export format (bibtex, ris, json, csv)

        Example:
            ```python
            manager.export_library("my_library.bib", format="bibtex")
            manager.export_library("my_library.ris", format="ris")
            manager.export_library("my_library.json", format="json")
            ```
        """
        papers = list(self.references.values())

        if format == "bibtex":
            papers_to_bibtex(papers, output_file)
        elif format == "ris":
            papers_to_ris(papers, output_file)
        elif format == "json":
            self._export_json(papers, output_file)
        elif format == "csv":
            self._export_csv(papers, output_file)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Exported {len(papers)} references to {output_file}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get library statistics.

        Returns:
            Statistics dictionary

        Example:
            ```python
            stats = manager.get_statistics()
            print(f"Total references: {stats['total_count']}")
            print(f"With DOI: {stats['doi_count']}")
            ```
        """
        stats = {
            "total_count": len(self.references),
            "doi_count": sum(1 for p in self.references.values() if p.doi),
            "arxiv_count": sum(1 for p in self.references.values() if p.arxiv_id),
            "pubmed_count": sum(1 for p in self.references.values() if p.pubmed_id),
            "year_distribution": {},
            "citation_links": len(self.citation_links)
        }

        # Year distribution
        for paper in self.references.values():
            if paper.year:
                stats["year_distribution"][paper.year] = stats["year_distribution"].get(paper.year, 0) + 1

        return stats

    # Helper methods

    def _generate_ref_id(self, paper: PaperMetadata) -> str:
        """Generate unique reference ID."""
        # Use primary identifier if available
        if paper.doi:
            return f"doi_{hashlib.md5(paper.doi.encode()).hexdigest()[:8]}"
        elif paper.arxiv_id:
            return f"arxiv_{hashlib.md5(paper.arxiv_id.encode()).hexdigest()[:8]}"
        elif paper.pubmed_id:
            return f"pubmed_{hashlib.md5(paper.pubmed_id.encode()).hexdigest()[:8]}"
        else:
            # Hash title
            title_hash = hashlib.md5(paper.title.encode()).hexdigest()[:8]
            return f"ref_{title_hash}"

    def _find_duplicate(self, paper: PaperMetadata) -> Optional[str]:
        """Find if paper is a duplicate of existing reference."""
        for ref_id, existing_paper in self.references.items():
            if self.dedup_engine.is_duplicate(existing_paper, paper):
                return ref_id
        return None

    def _merge_papers(self, ref_id: str, new_paper: PaperMetadata):
        """Merge new paper data into existing reference."""
        existing_paper = self.references[ref_id]
        merged_paper = self.dedup_engine.merge_paper_metadata([existing_paper, new_paper])
        self.references[ref_id] = merged_paper

    def _save_to_storage(self):
        """Save references to disk."""
        try:
            data = {
                "references": {},
                "citation_links": self.citation_links
            }

            # Serialize papers
            for ref_id, paper in self.references.items():
                data["references"][ref_id] = {
                    "id": paper.id,
                    "source": paper.source.value,
                    "title": paper.title,
                    "abstract": paper.abstract,
                    "authors": [{"name": a.name} for a in paper.authors],
                    "year": paper.year,
                    "doi": paper.doi,
                    "arxiv_id": paper.arxiv_id,
                    "pubmed_id": paper.pubmed_id,
                    "url": paper.url,
                    "journal": paper.journal,
                    "citation_count": paper.citation_count
                }

            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save to storage: {e}")

    def _load_from_storage(self):
        """Load references from disk."""
        try:
            with open(self.storage_path, 'r') as f:
                data = json.load(f)

            # Deserialize papers
            from evoverse.literature.base_client import PaperSource, Author

            for ref_id, paper_data in data.get("references", {}).items():
                authors = [Author(name=a["name"]) for a in paper_data.get("authors", [])]

                paper = PaperMetadata(
                    id=paper_data.get("id", ""),
                    source=PaperSource(paper_data.get("source", "other")),
                    title=paper_data.get("title", ""),
                    abstract=paper_data.get("abstract", ""),
                    authors=authors,
                    year=paper_data.get("year"),
                    doi=paper_data.get("doi"),
                    arxiv_id=paper_data.get("arxiv_id"),
                    pubmed_id=paper_data.get("pubmed_id"),
                    url=paper_data.get("url"),
                    journal=paper_data.get("journal"),
                    citation_count=paper_data.get("citation_count", 0)
                )

                self.references[ref_id] = paper

            self.citation_links = data.get("citation_links", {})

            logger.info(f"Loaded {len(self.references)} references from storage")

        except Exception as e:
            logger.error(f"Failed to load from storage: {e}")

    def _export_json(self, papers: List[PaperMetadata], output_file: str):
        """Export to JSON format."""
        data = []

        for paper in papers:
            paper_dict = {
                "title": paper.title,
                "authors": [a.name for a in paper.authors],
                "year": paper.year,
                "abstract": paper.abstract,
                "doi": paper.doi,
                "arxiv_id": paper.arxiv_id,
                "url": paper.url,
                "journal": paper.journal,
                "citation_count": paper.citation_count
            }
            data.append(paper_dict)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, papers: List[PaperMetadata], output_file: str):
        """Export to CSV format."""
        import csv

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['title', 'authors', 'year', 'journal', 'doi', 'citation_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for paper in papers:
                writer.writerow({
                    'title': paper.title,
                    'authors': "; ".join([a.name for a in paper.authors]),
                    'year': paper.year or '',
                    'journal': paper.journal or '',
                    'doi': paper.doi or '',
                    'citation_count': paper.citation_count
                })


class DeduplicationEngine:
    """
    Advanced reference deduplication.

    Supports multiple strategies for duplicate detection.
    """

    def __init__(self):
        """Initialize deduplication engine."""
        logger.debug("Initialized DeduplicationEngine")

    def deduplicate_by_doi(
        self,
        papers: List[PaperMetadata]
    ) -> Tuple[List[PaperMetadata], Dict[str, List[str]]]:
        """
        Deduplicate by DOI.

        Args:
            papers: List of papers

        Returns:
            Tuple of (unique_papers, duplicate_groups)
        """
        seen_dois = {}
        unique_papers = []
        duplicate_groups = {}

        for paper in papers:
            if paper.doi:
                if paper.doi in seen_dois:
                    # Duplicate found
                    group_key = seen_dois[paper.doi]
                    duplicate_groups[group_key].append(paper.title)
                else:
                    # First occurrence
                    seen_dois[paper.doi] = paper.title
                    unique_papers.append(paper)
                    duplicate_groups[paper.title] = [paper.title]
            else:
                # No DOI, keep it
                unique_papers.append(paper)

        return unique_papers, duplicate_groups

    def deduplicate_by_title(
        self,
        papers: List[PaperMetadata],
        threshold: float = 0.9
    ) -> Tuple[List[PaperMetadata], Dict[str, List[str]]]:
        """
        Deduplicate by fuzzy title matching.

        Args:
            papers: List of papers
            threshold: Similarity threshold (0-1)

        Returns:
            Tuple of (unique_papers, duplicate_groups)
        """
        unique_papers = []
        duplicate_groups = {}
        processed_titles = []

        for paper in papers:
            is_duplicate = False

            for i, existing_title in enumerate(processed_titles):
                similarity = self._title_similarity(paper.title, existing_title)

                if similarity >= threshold:
                    # Duplicate found
                    is_duplicate = True
                    group_key = existing_title
                    if group_key in duplicate_groups:
                        duplicate_groups[group_key].append(paper.title)
                    break

            if not is_duplicate:
                unique_papers.append(paper)
                processed_titles.append(paper.title)
                duplicate_groups[paper.title] = [paper.title]

        return unique_papers, duplicate_groups

    def comprehensive_deduplication(
        self,
        papers: List[PaperMetadata]
    ) -> Tuple[List[PaperMetadata], Dict[str, List[str]]]:
        """
        Multi-level deduplication.

        Priority: DOI > arXiv > PubMed > fuzzy title

        Args:
            papers: List of papers

        Returns:
            Tuple of (unique_papers, duplicate_groups)
        """
        # Track seen identifiers
        seen_dois = set()
        seen_arxiv = set()
        seen_pubmed = set()
        seen_titles = []

        unique_papers = []
        duplicate_groups = {}

        for paper in papers:
            is_duplicate = False
            group_key = None

            # Check DOI
            if paper.doi and paper.doi in seen_dois:
                is_duplicate = True
                group_key = f"DOI:{paper.doi}"

            # Check arXiv
            elif paper.arxiv_id and paper.arxiv_id in seen_arxiv:
                is_duplicate = True
                group_key = f"arXiv:{paper.arxiv_id}"

            # Check PubMed
            elif paper.pubmed_id and paper.pubmed_id in seen_pubmed:
                is_duplicate = True
                group_key = f"PubMed:{paper.pubmed_id}"

            # Check fuzzy title
            elif paper.title:
                for existing_title in seen_titles:
                    if self._title_similarity(paper.title, existing_title) >= 0.9:
                        is_duplicate = True
                        group_key = f"Title:{existing_title}"
                        break

            if is_duplicate:
                # Add to duplicate group
                if group_key not in duplicate_groups:
                    duplicate_groups[group_key] = []
                duplicate_groups[group_key].append(paper.title)
            else:
                # Add to unique papers
                unique_papers.append(paper)

                # Track identifiers
                if paper.doi:
                    seen_dois.add(paper.doi)
                if paper.arxiv_id:
                    seen_arxiv.add(paper.arxiv_id)
                if paper.pubmed_id:
                    seen_pubmed.add(paper.pubmed_id)
                if paper.title:
                    seen_titles.append(paper.title)

                # Initialize group
                duplicate_groups[paper.title] = [paper.title]

        return unique_papers, duplicate_groups

    def merge_paper_metadata(
        self,
        papers: List[PaperMetadata]
    ) -> PaperMetadata:
        """
        Merge duplicate papers, keeping best data.

        Priority: complete fields > higher citation counts > most recent

        Args:
            papers: List of duplicate papers

        Returns:
            Merged PaperMetadata
        """
        if not papers:
            raise ValueError("No papers to merge")

        if len(papers) == 1:
            return papers[0]

        # Start with first paper
        merged = papers[0]

        for paper in papers[1:]:
            # Prefer non-empty fields
            if not merged.title and paper.title:
                merged.title = paper.title
            if not merged.abstract and paper.abstract:
                merged.abstract = paper.abstract
            if not merged.doi and paper.doi:
                merged.doi = paper.doi
            if not merged.arxiv_id and paper.arxiv_id:
                merged.arxiv_id = paper.arxiv_id
            if not merged.pubmed_id and paper.pubmed_id:
                merged.pubmed_id = paper.pubmed_id
            if not merged.url and paper.url:
                merged.url = paper.url
            if not merged.journal and paper.journal:
                merged.journal = paper.journal

            # Prefer higher citation count
            if paper.citation_count > merged.citation_count:
                merged.citation_count = paper.citation_count

            # Prefer more recent year
            if paper.year and (not merged.year or paper.year > merged.year):
                merged.year = paper.year

            # Merge authors (deduplicate by name)
            if paper.authors:
                existing_names = {a.name for a in merged.authors}
                for author in paper.authors:
                    if author.name not in existing_names:
                        merged.authors.append(author)

        return merged

    def is_duplicate(
        self,
        paper1: PaperMetadata,
        paper2: PaperMetadata
    ) -> bool:
        """
        Check if two papers are duplicates.

        Args:
            paper1: First paper
            paper2: Second paper

        Returns:
            True if duplicate, False otherwise
        """
        # Check DOI
        if paper1.doi and paper2.doi and paper1.doi == paper2.doi:
            return True

        # Check arXiv
        if paper1.arxiv_id and paper2.arxiv_id and paper1.arxiv_id == paper2.arxiv_id:
            return True

        # Check PubMed
        if paper1.pubmed_id and paper2.pubmed_id and paper1.pubmed_id == paper2.pubmed_id:
            return True

        # Check fuzzy title
        if paper1.title and paper2.title:
            if self._title_similarity(paper1.title, paper2.title) >= 0.9:
                return True

        return False

    def _title_similarity(self, title1: str, title2: str) -> float:
        """Calculate title similarity using SequenceMatcher."""
        if not title1 or not title2:
            return 0.0

        # Normalize titles
        t1 = title1.lower().strip()
        t2 = title2.lower().strip()

        # Calculate similarity
        return SequenceMatcher(None, t1, t2).ratio()


# Singleton instance
_reference_manager: Optional[ReferenceManager] = None


def get_reference_manager(
    storage_path: Optional[str] = None,
    auto_deduplicate: bool = True,
    reset: bool = False
) -> ReferenceManager:
    """
    Get or create the singleton reference manager instance.

    Args:
        storage_path: Path to storage file
        auto_deduplicate: Whether to auto-deduplicate
        reset: Whether to reset the singleton

    Returns:
        ReferenceManager instance
    """
    global _reference_manager
    if _reference_manager is None or reset:
        _reference_manager = ReferenceManager(
            storage_path=storage_path,
            auto_deduplicate=auto_deduplicate
        )
    return _reference_manager


def reset_reference_manager():
    """Reset the singleton reference manager (useful for testing)."""
    global _reference_manager
    _reference_manager = None
