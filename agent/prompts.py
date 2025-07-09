"""
Prompts for Multi-Registry Carbon Project Eligibility Evaluation

This module contains all the prompts used by the evaluator to maintain clean code organization.
All prompts have been moved here from evaluator.py for better maintainability and separation of concerns.

Key Components:
- Registry-specific system prompts (Puro.earth, Verra, etc.)
- EVALUATION_TASK_INSTRUCTIONS: Response format instructions
- create_evaluation_prompt(): Dynamic prompt builder with token optimization
"""

from typing import Dict, List, Any


# Registry-specific system prompts
PURO_SYSTEM_PROMPT = """You are an expert evaluator for Puro.earth biochar project eligibility.

Your task is to analyze document evidence against specific Puro.earth requirements and determine:
1. STATUS: "present", "missing", or "unclear"
2. REASON: Clear explanation of your decision
3. EVIDENCE: What specific evidence you found (if any)
4. MISSING: What evidence is still needed (if any)
5. CONFIDENCE: Your confidence level (0.0-1.0)

EVALUATION CRITERIA:
- "present": Clear evidence that fully satisfies the requirement
- "missing": No relevant evidence found or evidence clearly shows non-compliance
- "unclear": Some evidence found but insufficient or ambiguous

KEY PURO.EARTH DEFINITIONS:

- H/Corg Ratio: Indicator of biochar stability; must be < 0.7 for carbon permanence.
- Additionality: Project must depend on carbon revenue and not be required by law.
- End-use: Biochar must not be used as fuel or reductant; must be verifiably used in carbon-retaining applications like soil amendment, construction materials, or insulation.
- Emissions Monitoring: Must use ISO 14040/44-compliant LCA covering biomass, production, distribution, and use phases.
- Biomass Source: Must be waste or sustainably sourced, aligned with EBC positive lists or IPCC guidelines.
- Proof Requirements: Look for offtake agreements, stakeholder consultations, accredited lab tests, reactor specifications, environmental permits, and LCA data.
- Biochar Stability: Long-term carbon storage capability, typically demonstrated through H/Corg ratio and permanence factors.
- Safety Protocols: Comprehensive procedures for biochar handling, storage, and transport including fire control measures.

Use these definitions when interpreting vague or partially matching content in documents. Be thorough but concise. Focus on factual evidence from the documents."""


VERRA_SYSTEM_PROMPT = """You are an expert evaluator for Verra (VCS) VM0047 Afforestation, Reforestation and Revegetation project eligibility.

Your task is to analyze document evidence against specific Verra VM0047 requirements and determine:
1. STATUS: "present", "missing", or "unclear"
2. REASON: Clear explanation of your decision
3. EVIDENCE: What specific evidence you found (if any)
4. MISSING: What evidence is still needed (if any)
5. CONFIDENCE: Your confidence level (0.0-1.0)

EVALUATION CRITERIA:
- "present": Clear evidence that fully satisfies the requirement
- "missing": No relevant evidence found or evidence clearly shows non-compliance
- "unclear": Some evidence found but insufficient or ambiguous

KEY VERRA VM0047 DEFINITIONS:

- Area-based Approach: Uses plot-based biomass sampling and remote sensing with performance benchmark for baseline.
- Census-based Approach: Direct planting only with full census tracking, zero baseline, no land use change.
- Additionality: Must demonstrate regulatory surplus; investment analysis required if carbon is sole income.
- Performance Benchmark: Baseline methodology for area-based projects using regional data.
- Regulatory Surplus: Project activities exceed what is required by law or regulation.
- Stock Difference Method: Method for estimating biomass change over time.
- Uncertainty Deduction: Minimum 10% deduction applied to account for measurement uncertainties.
- Leakage: Displacement of emissions outside project boundary (area-based only).
- Monitoring Plan: Defines tasks, boundaries, roles, and data handling procedures.
- VM0036: Methodology for wetland restoration and conservation (used for wetland components).
- VMD0054: Module for estimating leakage from displaced agricultural activities.

Use these definitions when interpreting vague or partially matching content in documents. Be thorough but concise. Focus on factual evidence from the documents."""


# Default system prompt (can be overridden by registry-specific prompts)
SYSTEM_PROMPT = PURO_SYSTEM_PROMPT


# Evaluation task instructions
EVALUATION_TASK_INSTRUCTIONS = """=== EVALUATION TASK ===
Based on the document evidence above, evaluate this requirement and respond in this exact format:

STATUS: [present/missing/unclear]
REASON: [Your detailed reasoning]
EVIDENCE: [Specific evidence found, or 'None']
MISSING: [What evidence is still needed, or 'None']
CONFIDENCE: [Your confidence level 0.0-1.0]

Remember:
- 'present': Clear evidence that fully satisfies the requirement
- 'missing': No relevant evidence or evidence shows non-compliance
- 'unclear': Some evidence but insufficient or ambiguous"""


def get_system_prompt_for_registry(registry: str = "puro") -> str:
    """
    Get the appropriate system prompt based on registry
    
    Args:
        registry: Registry name ("puro", "verra", etc.)
        
    Returns:
        str: Registry-specific system prompt
    """
    registry_prompts = {
        "puro": PURO_SYSTEM_PROMPT,
        "verra": VERRA_SYSTEM_PROMPT,
        "vcs": VERRA_SYSTEM_PROMPT,  # Alias for Verra
    }
    
    return registry_prompts.get(registry.lower(), PURO_SYSTEM_PROMPT)


def create_evaluation_prompt(
    checklist_item: Dict[str, Any], 
    context: List[Dict[str, Any]],
    api_provider: str = "openai",
    registry: str = "puro"
) -> str:
    """
    Create the evaluation prompt for LLM
    
    Args:
        checklist_item: The checklist item being evaluated
        context: Retrieved document context
        api_provider: API provider ("openai" or "groq") for token optimization
        registry: Registry type ("puro", "verra", etc.) for field mapping
        
    Returns:
        str: Formatted prompt for evaluation
    """
    
    # Handle both "requirement" and "parameter" fields
    requirement = checklist_item.get('requirement', checklist_item.get('parameter', 'Unknown requirement'))
    
    # Registry-specific field mapping
    if registry.lower() in ["verra", "vcs"]:
        # Verra uses different field names
        registry_requires = checklist_item.get('verraWillCheckFor', checklist_item.get('requirement', 'Not specified'))
        registry_checks_for = checklist_item.get('verraWillCheckFor', 'Not specified')
        registry_name = "Verra"
    else:
        # Puro.earth field names
        registry_requires = checklist_item.get('puroRequires', checklist_item.get('puroLooksFor', checklist_item.get('requirement', 'Not specified')))
        registry_checks_for = checklist_item.get('puroWillCheckFor', checklist_item.get('puroLooksFor', checklist_item.get('puroWillCheck', 'Not specified')))
        registry_name = "Puro.earth"
    
    documents_needed = checklist_item.get('documentsNeeded', 'Not specified')
    
    prompt_parts = [
        "=== REQUIREMENT TO EVALUATE ===",
        f"Requirement: {requirement}",
        f"{registry_name} Requires: {registry_requires}",
        f"Documents Needed: {documents_needed}",
        f"{registry_name} Will Check For: {registry_checks_for}",
    ]
    
    # Standard fields that are handled above
    standard_fields = {
        'requirement', 'parameter', 'puroRequires', 'puroLooksFor', 'puroWillCheckFor', 
        'puroWillCheck', 'verraWillCheckFor', 'documentsNeeded', 'searchKeywords'
    }
    
    # Add any additional context fields dynamically
    context_fields_added = []
    for key, value in checklist_item.items():
        if key not in standard_fields and value:
            # Format field name for display (convert camelCase to Title Case)
            display_name = ''.join([' ' + c if c.isupper() else c for c in key]).strip().title()
            
            if isinstance(value, (list, dict)):
                # Handle complex data structures
                import json
                formatted_value = json.dumps(value, indent=2)
                prompt_parts.append(f"{display_name}:\n{formatted_value}")
            else:
                # Handle simple strings/numbers
                prompt_parts.append(f"{display_name}: {value}")
            context_fields_added.append(key)
    
    # Add spacing if context fields were added
    if context_fields_added:
        prompt_parts.append("")
        
    prompt_parts.extend([
        "",
        "=== DOCUMENT EVIDENCE ===",
    ])
    
    if not context:
        prompt_parts.append("No relevant document evidence found.")
    else:
        for i, ctx in enumerate(context, 1):
            # Limit context content to reduce token usage for Groq
            content_limit = 600 if api_provider == "groq" else 800
            content = ctx['content'][:content_limit]
            if len(ctx['content']) > content_limit:
                content += "... [truncated]"
            
            prompt_parts.extend([
                f"Evidence {i} (from {ctx['source']}):",
                f"Query used: {ctx['query']}",
                f"Content: {content}",
                ""
            ])
    
    # Add evaluation task instructions
    prompt_parts.extend([
        "",
        EVALUATION_TASK_INSTRUCTIONS
    ])
    
    return "\n".join(prompt_parts)
