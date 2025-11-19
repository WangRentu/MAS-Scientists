"""
PDF text extraction utilities using PyMuPDF (fitz).

Downloads and extracts text from scientific PDFs with error handling
and fallback to abstract when full text is unavailable.
"""

import fitz  # PyMuPDF
import httpx
from pathlib import Path
from typing import Optional, Dict, Any
import logging
import re
from datetime import datetime

from evoverse.literature.base_client import PaperMetadata
from evoverse.config import get_config

logger = logging.getLogger(__name__)


class PDFExtractionError(Exception):
    """Exception raised for PDF extraction errors."""
    pass


class PDFExtractor:
    """
    PDF text extraction utility using PyMuPDF.

    Supports:
    - URL-based PDF download
    - Local PDF file extraction
    - Text cleaning and normalization
    - Metadata extraction
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the PDF extractor.

        Args:
            cache_dir: Directory to cache downloaded PDFs
        """
        config = get_config()
        cache_dir = cache_dir or str(Path(config.literature.cache_dir) / "pdfs")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.download_timeout = config.literature.pdf_download_timeout

        logger.info(f"Initialized PDF extractor (cache_dir={cache_dir})")

    def extract_from_url(self, url: str, paper_id: Optional[str] = None) -> Optional[str]:
        """
        Download and extract text from a PDF URL.

        Args:
            url: URL to PDF file
            paper_id: Optional paper ID for caching

        Returns:
            Extracted text or None if extraction fails

        Example:
            ```python
            extractor = PDFExtractor()
            text = extractor.extract_from_url("https://arxiv.org/pdf/2103.00020.pdf")
            ```
        """
        try:
            # Check cache
            if paper_id:
                cached_pdf = self.cache_dir / f"{paper_id}.pdf"
                if cached_pdf.exists():
                    logger.debug(f"Using cached PDF for {paper_id}")
                    return self.extract_from_file(str(cached_pdf))

            # Download PDF
            logger.info(f"Downloading PDF from {url}")
            pdf_bytes = self._download_pdf(url)

            if not pdf_bytes:
                return None

            # Save to cache if paper_id provided
            if paper_id:
                cached_pdf = self.cache_dir / f"{paper_id}.pdf"
                with open(cached_pdf, 'wb') as f:
                    f.write(pdf_bytes)

            # Extract text
            text = self._extract_text_from_bytes(pdf_bytes)

            if text:
                logger.info(f"Successfully extracted {len(text)} characters from PDF")

            return text

        except Exception as e:
            logger.error(f"Error extracting from URL {url}: {e}")
            return None

    def extract_from_file(self, file_path: str) -> Optional[str]:
        """
        Extract text from a local PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Extracted text or None if extraction fails

        Example:
            ```python
            text = extractor.extract_from_file("/path/to/paper.pdf")
            ```
        """
        try:
            pdf_path = Path(file_path)

            if not pdf_path.exists():
                logger.error(f"PDF file not found: {file_path}")
                return None

            with open(pdf_path, 'rb') as f:
                pdf_bytes = f.read()

            text = self._extract_text_from_bytes(pdf_bytes)

            if text:
                logger.info(f"Successfully extracted {len(text)} characters from {file_path}")

            return text

        except Exception as e:
            logger.error(f"Error extracting from file {file_path}: {e}")
            return None

    def extract_with_metadata(self, url_or_path: str) -> Dict[str, Any]:
        """
        Extract both text and metadata from PDF.

        Args:
            url_or_path: URL or file path to PDF

        Returns:
            Dictionary with 'text' and 'metadata' keys

        Example:
            ```python
            result = extractor.extract_with_metadata("paper.pdf")
            print(result['metadata']['author'])
            print(result['text'][:100])
            ```
        """
        # Determine if URL or file path
        is_url = url_or_path.startswith(("http://", "https://"))

        if is_url:
            pdf_bytes = self._download_pdf(url_or_path)
        else:
            with open(url_or_path, 'rb') as f:
                pdf_bytes = f.read()

        if not pdf_bytes:
            return {"text": None, "metadata": {}}

        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Extract text
            text = self._extract_text_from_doc(doc)

            # Extract metadata
            metadata = self._extract_metadata(doc)

            doc.close()

            return {"text": text, "metadata": metadata}

        except Exception as e:
            logger.error(f"Error extracting with metadata: {e}")
            return {"text": None, "metadata": {}}

    def extract_paper_text(self, paper: PaperMetadata) -> str:
        """
        Extract full text for a PaperMetadata object.

        Falls back to abstract if PDF extraction fails.

        Args:
            paper: PaperMetadata object

        Returns:
            Full text or abstract

        Example:
            ```python
            text = extractor.extract_paper_text(paper)
            ```
        """
        # Try PDF extraction if URL available
        if paper.pdf_url:
            full_text = self.extract_from_url(
                paper.pdf_url,
                paper_id=paper.primary_identifier.replace("/", "_").replace(":", "_")
            )

            if full_text:
                # Store in paper object
                paper.full_text = full_text
                return full_text

        # Fallback to abstract
        logger.debug(f"No PDF available for {paper.id}, using abstract")
        paper.full_text = paper.abstract
        return paper.abstract

    def _download_pdf(self, url: str) -> Optional[bytes]:
        """
        Download PDF from URL.

        Args:
            url: PDF URL

        Returns:
            PDF bytes or None if download fails
        """
        try:
            with httpx.Client(timeout=self.download_timeout, follow_redirects=True) as client:
                response = client.get(url)
                response.raise_for_status()

                # Verify it's a PDF
                content_type = response.headers.get("content-type", "")
                if "pdf" not in content_type.lower():
                    logger.warning(f"URL does not appear to be a PDF: {url}")

                return response.content

        except httpx.TimeoutException:
            logger.error(f"Timeout downloading PDF from {url}")
            return None
        except httpx.HTTPError as e:
            logger.error(f"HTTP error downloading PDF from {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error downloading PDF from {url}: {e}")
            return None

    def _extract_text_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """
        Extract text from PDF bytes.

        Args:
            pdf_bytes: PDF file bytes

        Returns:
            Extracted text or None
        """
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = self._extract_text_from_doc(doc)
            doc.close()
            return text

        except Exception as e:
            logger.error(f"Error extracting text from PDF bytes: {e}")
            return None

    def _extract_text_from_doc(self, doc: fitz.Document) -> Optional[str]:
        """
        Extract text from PyMuPDF document.

        Args:
            doc: PyMuPDF Document object

        Returns:
            Extracted and cleaned text
        """
        try:
            text_parts = []

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                text_parts.append(text)

            full_text = "\n\n".join(text_parts)

            # Clean text
            cleaned_text = self._clean_text(full_text)

            if len(cleaned_text.strip()) < 100:
                logger.warning("Extracted text is too short, may be scanned/image-based PDF")
                return None

            return cleaned_text

        except Exception as e:
            logger.error(f"Error extracting text from document: {e}")
            return None

    def _extract_metadata(self, doc: fitz.Document) -> Dict[str, Any]:
        """
        Extract metadata from PDF.

        Args:
            doc: PyMuPDF Document object

        Returns:
            Dictionary with metadata fields
        """
        try:
            metadata = doc.metadata

            return {
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "keywords": metadata.get("keywords", ""),
                "creator": metadata.get("creator", ""),
                "producer": metadata.get("producer", ""),
                "creation_date": metadata.get("creationDate", ""),
                "mod_date": metadata.get("modDate", ""),
                "page_count": len(doc)
            }

        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            return {}

    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text.

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers (common patterns)
        text = re.sub(r'\n\d+\n', '\n', text)

        # Remove common PDF artifacts
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII

        # Normalize line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)

        return text.strip()

    def get_cache_size(self) -> Dict[str, Any]:
        """
        Get information about the PDF cache.

        Returns:
            Dictionary with cache statistics
        """
        pdf_files = list(self.cache_dir.glob("*.pdf"))
        total_size_mb = sum(f.stat().st_size for f in pdf_files) / (1024 * 1024)

        return {
            "cache_dir": str(self.cache_dir),
            "file_count": len(pdf_files),
            "size_mb": round(total_size_mb, 2)
        }

    def clear_cache(self):
        """Clear all cached PDFs."""
        count = 0
        for pdf_file in self.cache_dir.glob("*.pdf"):
            pdf_file.unlink()
            count += 1

        logger.info(f"Cleared {count} PDFs from cache")


# Singleton extractor instance
_extractor: Optional[PDFExtractor] = None


def get_pdf_extractor(cache_dir: Optional[str] = None) -> PDFExtractor:
    """
    Get or create the singleton PDF extractor instance.

    Args:
        cache_dir: Directory to cache downloaded PDFs

    Returns:
        PDFExtractor instance
    """
    global _extractor
    if _extractor is None:
        _extractor = PDFExtractor(cache_dir=cache_dir)
    return _extractor


def reset_pdf_extractor():
    """Reset the singleton extractor (useful for testing)."""
    global _extractor
    _extractor = None
