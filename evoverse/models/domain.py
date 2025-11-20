"""
Domain classification and routing models for EvoVerse (MAS-Scientists).

Lightweight adaptation of Kosmos' domain models:
- ScientificDomain / DomainConfidence enums
- DomainClassification: result of classifying a research question
- DomainRoute: minimal routing decision (domains + strategy + hints)
"""

from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field


class ScientificDomain(str, Enum):
    """Comprehensive scientific research domains for multi-agent AI Scientist systems."""

    # Life Sciences
    BIOLOGY = "biology"
    MOLECULAR_BIOLOGY = "molecular_biology"
    GENETICS = "genetics"
    BIOCHEMISTRY = "biochemistry"
    NEUROSCIENCE = "neuroscience"
    IMMUNOLOGY = "immunology"
    MICROBIOLOGY = "microbiology"
    EPIDEMIOLOGY = "epidemiology"
    PHARMACOLOGY = "pharmacology"
    MEDICINE = "medicine"
    PUBLIC_HEALTH = "public_health"

    # Chemistry & Materials
    CHEMISTRY = "chemistry"
    ORGANIC_CHEMISTRY = "organic_chemistry"
    PHYSICAL_CHEMISTRY = "physical_chemistry"
    MATERIALS_SCIENCE = "materials_science"
    NANOTECHNOLOGY = "nanotechnology"
    CHEMICAL_ENGINEERING = "chemical_engineering"

    # Physics
    PHYSICS = "physics"
    ASTROPHYSICS = "astrophysics"
    QUANTUM_PHYSICS = "quantum_physics"
    THERMODYNAMICS = "thermodynamics"
    OPTICS = "optics"
    GEOPHYSICS = "geophysics"
    SPACE_SCIENCE = "space_science"

    # Computer & Information Sciences
    COMPUTER_SCIENCE = "computer_science"
    ARTIFICIAL_INTELLIGENCE = "artificial_intelligence"
    MACHINE_LEARNING = "machine_learning"
    DATA_SCIENCE = "data_science"
    INFORMATION_SCIENCE = "information_science"
    ROBOTICS = "robotics"

    # Mathematics & Statistics
    MATHEMATICS = "mathematics"
    APPLIED_MATHEMATICS = "applied_mathematics"
    STATISTICS = "statistics"

    # Engineering
    ELECTRICAL_ENGINEERING = "electrical_engineering"
    MECHANICAL_ENGINEERING = "mechanical_engineering"
    BIOMEDICAL_ENGINEERING = "biomedical_engineering"
    CIVIL_ENGINEERING = "civil_engineering"
    AEROSPACE_ENGINEERING = "aerospace_engineering"
    MATERIALS_ENGINEERING = "materials_engineering"
    ENERGY_ENGINEERING = "energy_engineering"

    # Earth & Environmental Sciences
    EARTH_SCIENCE = "earth_science"
    ENVIRONMENTAL_SCIENCE = "environmental_science"
    CLIMATE_SCIENCE = "climate_science"
    ECOLOGY = "ecology"
    AGRICULTURE = "agriculture"
    OCEANOGRAPHY = "oceanography"

    # Social Sciences & Humanities
    SOCIAL_SCIENCE = "social_science"
    ECONOMICS = "economics"
    PSYCHOLOGY = "psychology"
    SOCIOLOGY = "sociology"
    POLITICAL_SCIENCE = "political_science"
    EDUCATION = "education"
    LINGUISTICS = "linguistics"

    # Interdisciplinary Fields
    COMPUTATIONAL_SCIENCE = "computational_science"
    SYSTEMS_SCIENCE = "systems_science"
    COMPLEX_SYSTEMS = "complex_systems"
    BIOINFORMATICS = "bioinformatics"
    COMPUTATIONAL_NEUROSCIENCE = "computational_neuroscience"
    COMPUTATIONAL_SOCIAL_SCIENCE = "computational_social_science"
    QUANTITATIVE_FINANCE = "quantitative_finance"

    # Default / general
    GENERAL = "general"

    MATERIALS = "materials"
    ASTRONOMY = "astronomy"


class DomainConfidence(str, Enum):
    """Confidence levels for domain classification."""

    VERY_HIGH = "very_high"  # > 0.9
    HIGH = "high"            # 0.7 - 0.9
    MEDIUM = "medium"        # 0.5 - 0.7
    LOW = "low"              # 0.3 - 0.5
    VERY_LOW = "very_low"    # < 0.3


class DomainClassification(BaseModel):
    """Result of classifying a research question or hypothesis to a domain."""

    # Primary classification
    primary_domain: ScientificDomain = Field(
        description="The primary scientific domain identified"
    )
    confidence: DomainConfidence = Field(
        description="Confidence level in the primary classification"
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Numeric confidence score (0-1)",
    )

    # Secondary domains (for multi-domain questions)
    secondary_domains: List[ScientificDomain] = Field(
        default_factory=list,
        description="Additional relevant domains (for cross-domain research)",
    )
    domain_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence scores for all considered domains",
    )

    # Classification details
    key_terms: List[str] = Field(
        default_factory=list,
        description="Key terms that influenced classification",
    )
    classification_reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of why this domain was chosen",
    )

    # Multi-domain detection
    is_multi_domain: bool = Field(
        default=False,
        description="Whether this question spans multiple domains",
    )
    cross_domain_rationale: Optional[str] = Field(
        default=None,
        description="Explanation if multi-domain research is detected",
    )

    # Metadata
    classified_at: datetime = Field(default_factory=datetime.now)
    classifier_model: str = Field(
        default="mas-llm",
        description="Identifier for the classifier model used",
    )

    def to_domain_list(self) -> List[ScientificDomain]:
        """Get all relevant domains (primary + secondary) as a list."""
        domains: List[ScientificDomain] = [self.primary_domain]
        domains.extend(self.secondary_domains)
        # Remove duplicates while preserving order
        seen = set()
        deduped: List[ScientificDomain] = []
        for d in domains:
            if d not in seen:
                seen.add(d)
                deduped.append(d)
        return deduped

    def requires_cross_domain_synthesis(self) -> bool:
        """Check if cross-domain synthesis is needed."""
        return self.is_multi_domain and len(self.secondary_domains) > 0


class DomainRoute(BaseModel):
    """Routing decision for a research question to domain-specific agents and tools."""

    # Classification
    classification: DomainClassification

    # Selected route
    selected_domains: List[ScientificDomain] = Field(
        description="Domains selected for this research"
    )
    routing_strategy: str = Field(
        description=(
            "Routing strategy: single_domain, "
            "parallel_multi_domain, sequential_multi_domain"
        )
    )

    # Agent selection (hints only; not auto-instantiated)
    assigned_agents: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Agent type hints per domain (domain -> agent_types)",
    )

    # Tool selection (hints only)
    required_tools: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Required tools per domain (domain -> tool_names)",
    )
    recommended_templates: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Recommended experiment templates per domain",
    )

    # Cross-domain synthesis
    synthesis_required: bool = Field(
        default=False,
        description="Whether cross-domain synthesis is needed",
    )
    synthesis_strategy: Optional[str] = Field(
        default=None,
        description="Strategy for synthesizing cross-domain results",
    )

    # Routing metadata
    routed_at: datetime = Field(default_factory=datetime.now)
    routing_reasoning: Optional[str] = Field(
        default=None,
        description="Explanation of routing decision",
    )

    def get_all_tools(self) -> List[str]:
        """Get all required tools across all domains."""
        tools: List[str] = []
        for tool_list in self.required_tools.values():
            tools.extend(tool_list)
        # De-duplicate
        return sorted(set(tools))

    def get_all_templates(self) -> List[str]:
        """Get all recommended templates across all domains."""
        templates: List[str] = []
        for template_list in self.recommended_templates.values():
            templates.extend(template_list)
        return sorted(set(templates))

    def is_single_domain(self) -> bool:
        """Check if this is single-domain research."""
        return len(self.selected_domains) == 1

    def get_primary_domain(self) -> ScientificDomain:
        """Get the primary domain for routing."""
        return self.classification.primary_domain

