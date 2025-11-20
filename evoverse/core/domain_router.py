"""
Domain Router for EvoVerse (MAS-Scientists).

Adapted from Kosmos' DomainRouter:
- Uses EvoVerse's LLMClient instead of ClaudeClient
- Keeps prompt format and parsing logic for compatibility

Main capabilities:
- Classify research questions into one or more ScientificDomain values
- Provide a lightweight routing decision (DomainRoute) for orchestration
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any

from evoverse.core.llm_client import LLMClient, create_message
from evoverse.models.domain import (
    ScientificDomain,
    DomainClassification,
    DomainConfidence,
    DomainRoute,
)

logger = logging.getLogger(__name__)


class DomainRouter:
    """
    Routes research questions to appropriate scientific domains.

    Capabilities:
    - Domain classification using LLM
    - Multi-domain research detection
    - Lightweight routing hints (agents / tools / templates)
    """

    # Domain keywords for classification hints (fallback)
    DOMAIN_KEYWORDS: Dict[ScientificDomain, List[str]] = {
        ScientificDomain.BIOLOGY: [
            "gene",
            "protein",
            "dna",
            "rna",
            "cell",
            "organism",
            "species",
            "metabolite",
            "pathway",
            "genome",
            "expression",
            "mutation",
            "evolution",
            "ecology",
            "metabolism",
            "enzyme",
            "gwas",
            "snp",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "neuron",
            "brain",
            "synapse",
            "neural",
            "cognitive",
            "neuronal",
            "cortex",
            "hippocampus",
            "alzheimer",
            "parkinson",
            "connectome",
            "spike",
            "fmri",
            "eeg",
            "neurotransmitter",
            "plasticity",
        ],
        ScientificDomain.MATERIALS: [
            "material",
            "crystal",
            "structure",
            "property",
            "synthesis",
            "perovskite",
            "solar cell",
            "semiconductor",
            "composite",
            "conductivity",
            "strength",
            "optimization",
            "parameter",
        ],
        ScientificDomain.PHYSICS: [
            "force",
            "energy",
            "momentum",
            "particle",
            "wave",
            "field",
            "quantum",
            "thermodynamic",
            "mechanics",
            "electromagnetic",
            "relativity",
            "optics",
            "plasma",
            "cosmology",
        ],
        ScientificDomain.CHEMISTRY: [
            "molecule",
            "reaction",
            "compound",
            "synthesis",
            "catalyst",
            "bond",
            "electron",
            "oxidation",
            "reduction",
            "spectroscopy",
            "chromatography",
            "polymer",
            "organic",
            "inorganic",
        ],
        ScientificDomain.ASTRONOMY: [
            "star",
            "planet",
            "galaxy",
            "universe",
            "cosmic",
            "telescope",
            "orbit",
            "redshift",
            "black hole",
            "nebula",
            "exoplanet",
            "supernova",
            "dark matter",
            "constellation",
        ],
        ScientificDomain.SOCIAL_SCIENCE: [
            "society",
            "behavior",
            "population",
            "survey",
            "demographic",
            "psychology",
            "sociology",
            "economics",
            "anthropology",
            "culture",
            "policy",
            "intervention",
            "cohort study",
        ],
    }

    # Domain-specific agent type hints (string labels only)
    DOMAIN_AGENTS: Dict[ScientificDomain, List[str]] = {
        ScientificDomain.BIOLOGY: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.MATERIALS: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.PHYSICS: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.CHEMISTRY: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.SOCIAL_SCIENCE: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.ASTRONOMY: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
        ScientificDomain.GENERAL: [
            "ScientistAgent",
            "LiteratureAgent",
        ],
    }

    # Domain-specific templates (high-level analysis patterns)
    DOMAIN_TEMPLATES: Dict[ScientificDomain, List[str]] = {
        ScientificDomain.BIOLOGY: [
            "metabolomics_comparison",
            "gwas_multimodal",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "connectome_scaling",
            "differential_expression",
        ],
        ScientificDomain.MATERIALS: [
            "parameter_correlation",
            "optimization",
        ],
        ScientificDomain.PHYSICS: [
            "simulation_study",
            "theoretical_analysis",
        ],
        ScientificDomain.CHEMISTRY: [
            "reaction_optimization",
            "property_prediction",
        ],
        ScientificDomain.SOCIAL_SCIENCE: [
            "survey_analysis",
            "policy_evaluation",
        ],
        ScientificDomain.ASTRONOMY: [
            "observation_analysis",
            "simulation_comparison",
        ],
        ScientificDomain.GENERAL: [
            "ttest_comparison",
            "correlation_analysis",
            "log_log_analysis",
        ],
    }

    # Domain-specific tools/APIs (hints; MAS itself may not implement all)
    DOMAIN_TOOLS: Dict[ScientificDomain, List[str]] = {
        ScientificDomain.BIOLOGY: [
            "KEGGClient",
            "GWASCatalogClient",
        ],
        ScientificDomain.NEUROSCIENCE: [
            "AllenBrainClient",
        ],
        ScientificDomain.MATERIALS: [
            "MaterialsProjectClient",
        ],
    }

    def __init__(self, llm_client: Optional[LLMClient] = None):
        """
        Initialize domain router.

        Args:
            llm_client: Optional EvoVerse LLM client (creates new one if not provided)
        """
        # Limit history for classification calls to keep prompts light
        self.llm = llm_client or LLMClient(max_history=8)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify_research_question(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> DomainClassification:
        """
        Classify a research question to scientific domain(s).

        Args:
            question: The research question to classify
            context: Optional context (hypothesis, data description, etc.)
        """
        logger.info("Classifying research question: %s", question[:100])

        # Build classification prompt
        prompt = self._build_classification_prompt(question, context)

        # Get classification from LLM
        try:
            response = self.llm.chat(
                [
                    create_message(
                        "system",
                        "You are a scientific domain classifier. "
                        "Follow the requested output format exactly.",
                    ),
                    create_message("user", prompt),
                ]
            )

            classification = self._parse_classification_response(response, question)

            logger.info(
                "Classified to %s (confidence: %s, score=%.2f)",
                classification.primary_domain.value,
                classification.confidence.value,
                classification.confidence_score,
            )

            return classification

        except Exception as exc:  # noqa: BLE001
            logger.error("Domain classification failed, falling back to keywords: %s", exc)
            return self._keyword_based_classification(question)

    def _build_classification_prompt(
        self,
        question: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build prompt for LLM to classify research question."""
        domains_list = ", ".join([d.value for d in ScientificDomain])

        prompt = (
            "Translate the question into English at first.\n\n"
            "Then, Classify the following research question into one or more scientific domains.\n\n"
            "Research Question:\n"
            f"{question}\n\n"
        )

        if context:
            prompt += "Additional Context:\n"
            prompt += f"{context}\n\n"

        prompt += "Available Domains:\n"
        prompt += f"{domains_list}\n\n"
        prompt += (
            "Instructions:\n"
            "1. Identify ALL scientific domains that are meaningfully involved in this question (1-3 domains).\n"
            "   - PRIMARY DOMAIN must be the most central domain.\n"
            "   - SECONDARY DOMAINS should list up to 2 additional domains in descending order of relevance.\n"
            "2. Assign a confidence level for the PRIMARY DOMAIN: very_high (>0.9), high (0.7-0.9), "
            "medium (0.5-0.7), low (0.3-0.5), very_low (<0.3).\n"
            "3. Extract key terms that influenced your classification.\n"
            "4. Explain your reasoning, especially if multiple domains are involved.\n"
            "5. If more than 3 domains seem relevant, keep ONLY the 3 most relevant domains.\n\n"
            "Respond in the following format (one item per line):\n\n"
            "PRIMARY DOMAIN: <domain_name>\n"
            "CONFIDENCE: <confidence_level>\n"
            "CONFIDENCE_SCORE: <0-1 numeric score>\n"
            "SECONDARY DOMAINS: <comma-separated list (max 2) or \"none\">\n"
            "KEY TERMS: <comma-separated key terms>\n"
            "IS MULTI-DOMAIN: <yes/no>\n"
            "REASONING: <your explanation>\n"
        )

        return prompt

    def _parse_classification_response(
        self,
        response: str,
        question: str,
    ) -> DomainClassification:
        """Parse LLM classification response into DomainClassification object."""
        lines = (response or "").strip().split("\n")
        data: Dict[str, str] = {}

        for line in lines:
            if ":" in line:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()

        # Extract primary domain
        primary_domain_str = data.get("PRIMARY DOMAIN", "general").lower()
        try:
            primary_domain = ScientificDomain(primary_domain_str)
        except ValueError:
            primary_domain = ScientificDomain.GENERAL

        # Extract confidence enum
        confidence_str = data.get("CONFIDENCE", "medium").lower().replace(" ", "_")
        try:
            confidence = DomainConfidence(confidence_str)
        except ValueError:
            confidence = DomainConfidence.MEDIUM

        # Confidence score
        try:
            confidence_score = float(data.get("CONFIDENCE_SCORE", "0.6"))
        except ValueError:
            confidence_score = 0.6

        # Secondary domains
        secondary_str = data.get("SECONDARY DOMAINS", "none")
        secondary_domains: List[ScientificDomain] = []
        if secondary_str.lower() != "none":
            for domain_str in secondary_str.split(","):
                d_str = domain_str.strip().lower()
                if not d_str:
                    continue
                try:
                    d = ScientificDomain(d_str)
                    if d != primary_domain:
                        secondary_domains.append(d)
                except ValueError:
                    # Ignore unknown domain labels
                    continue
        # 限制副领域数量：最多 2 个（加上 primary 一共最多 3 个领域）
        secondary_domains = secondary_domains[:2]

        # Key terms
        key_terms_str = data.get("KEY TERMS", "")
        key_terms = [term.strip() for term in key_terms_str.split(",") if term.strip()]

        # Multi-domain flag
        is_multi_domain = data.get("IS MULTI-DOMAIN", "no").lower() in ["yes", "true"]

        reasoning = data.get("REASONING", "")

        # Domain scores (simple assignment)
        domain_scores: Dict[str, float] = {
            primary_domain.value: confidence_score,
        }
        for i, sec_domain in enumerate(secondary_domains):
            domain_scores[sec_domain.value] = confidence_score * (0.7 - i * 0.1)

        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            confidence_score=confidence_score,
            secondary_domains=secondary_domains,
            domain_scores=domain_scores,
            key_terms=key_terms,
            classification_reasoning=reasoning or question[:200],
            is_multi_domain=is_multi_domain,
            cross_domain_rationale=reasoning if is_multi_domain else None,
        )

    def _keyword_based_classification(
        self,
        question: str,
    ) -> DomainClassification:
        """Fallback classification using simple keyword matching."""
        question_lower = (question or "").lower()

        # Calculate scores for each domain based on keyword matches
        domain_scores: Dict[str, float] = {}
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            matches = sum(1 for keyword in keywords if keyword in question_lower)
            domain_scores[domain.value] = (
                matches / len(keywords) if keywords else 0.0
            )

        # Get primary domain (highest score)
        if not domain_scores or max(domain_scores.values()) == 0:
            primary_domain = ScientificDomain.GENERAL
            confidence_score = 0.5
        else:
            primary_domain_str = max(domain_scores, key=domain_scores.get)
            primary_domain = ScientificDomain(primary_domain_str)
            confidence_score = domain_scores[primary_domain_str]

        # Map score to confidence level
        if confidence_score > 0.9:
            confidence = DomainConfidence.VERY_HIGH
        elif confidence_score > 0.7:
            confidence = DomainConfidence.HIGH
        elif confidence_score > 0.5:
            confidence = DomainConfidence.MEDIUM
        elif confidence_score > 0.3:
            confidence = DomainConfidence.LOW
        else:
            confidence = DomainConfidence.VERY_LOW

        # Secondary domains (scores > 0.3, excluding primary)
        secondary_domains: List[ScientificDomain] = [
            ScientificDomain(domain_str)
            for domain_str, score in domain_scores.items()
            if score > 0.3 and ScientificDomain(domain_str) != primary_domain
        ]

        return DomainClassification(
            primary_domain=primary_domain,
            confidence=confidence,
            confidence_score=confidence_score,
            secondary_domains=secondary_domains[:2],
            domain_scores=domain_scores,
            key_terms=[],
            classification_reasoning="Keyword-based fallback classification",
            is_multi_domain=len(secondary_domains) > 0,
            cross_domain_rationale=None,
        )

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(
        self,
        question: str,
        classification: Optional[DomainClassification] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> DomainRoute:
        """
        Create a routing decision for a research question.

        Args:
            question: Research question to route
            classification: Optional pre-computed classification
            context: Optional context for routing decisions
        """
        if classification is None:
            classification = self.classify_research_question(question, context)

        # Determine routing strategy
        if classification.is_multi_domain and classification.secondary_domains:
            selected_domains = classification.to_domain_list()
            routing_strategy = self._determine_multi_domain_strategy(
                classification, context
            )
        else:
            selected_domains = [classification.primary_domain]
            routing_strategy = "single_domain"

        # Select agent/tool/template hints per domain
        assigned_agents: Dict[str, List[str]] = {}
        required_tools: Dict[str, List[str]] = {}
        recommended_templates: Dict[str, List[str]] = {}

        for domain in selected_domains:
            key = domain.value
            assigned_agents[key] = self.DOMAIN_AGENTS.get(domain, [])
            required_tools[key] = self.DOMAIN_TOOLS.get(domain, [])
            recommended_templates[key] = self.DOMAIN_TEMPLATES.get(domain, [])

        # Cross-domain synthesis
        synthesis_required = classification.requires_cross_domain_synthesis()
        synthesis_strategy = (
            self._determine_synthesis_strategy(classification)
            if synthesis_required
            else None
        )

        route = DomainRoute(
            classification=classification,
            selected_domains=selected_domains,
            routing_strategy=routing_strategy,
            assigned_agents=assigned_agents,
            required_tools=required_tools,
            recommended_templates=recommended_templates,
            synthesis_required=synthesis_required,
            synthesis_strategy=synthesis_strategy,
            routing_reasoning=self._build_routing_reasoning(
                classification, routing_strategy, selected_domains
            ),
        )

        logger.info(
            "Routing decision: %d domain(s) with %s strategy",
            len(selected_domains),
            routing_strategy,
        )
        return route

    def _determine_multi_domain_strategy(
        self,
        classification: DomainClassification,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Determine routing strategy for multi-domain research.

        Returns:
            'parallel_multi_domain' or 'sequential_multi_domain'
        """
        # For now, default to parallel execution (most efficient).
        # Future: use LLM + context to infer dependencies between domains.
        return "parallel_multi_domain"

    def _determine_synthesis_strategy(
        self,
        classification: DomainClassification,
    ) -> str:
        """Determine strategy for synthesizing cross-domain results."""
        domains = classification.to_domain_list()
        domain_names = sorted([d.value for d in domains])

        if "biology" in domain_names and "neuroscience" in domain_names:
            return "biological_neural_integration"
        if "materials" in domain_names and "physics" in domain_names:
            return "materials_physics_integration"
        if "chemistry" in domain_names and "biology" in domain_names:
            return "biochemical_integration"
        return "general_cross_domain_synthesis"

    def _build_routing_reasoning(
        self,
        classification: DomainClassification,
        strategy: str,
        domains: List[ScientificDomain],
    ) -> str:
        """Build human-readable reasoning for routing decision."""
        domain_list = ", ".join([d.value for d in domains])
        reasoning = (
            f"Routing to {len(domains)} domain(s): {domain_list}. "
            f"Strategy: {strategy}. "
            f"Classification confidence: {classification.confidence.value} "
            f"({classification.confidence_score:.2f}). "
        )
        if classification.is_multi_domain:
            reasoning += "Multi-domain research detected. "
        return reasoning
