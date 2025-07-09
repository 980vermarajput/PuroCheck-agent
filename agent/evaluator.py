"""
Checklist Evaluator - RAG-based requirement evaluation

This module handles the evaluation of individual checklist items using 
RAG (Retrieval-Augmented Generation) with GPT-4 for reasoning.
"""

import logging
import re
import os
import time
from typing import Dict, List, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

from .nodes import get_relevant_docs
from .prompts import SYSTEM_PROMPT, create_evaluation_prompt, get_system_prompt_for_registry

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class ChecklistEvaluator:
    """
    Evaluates checklist requirements using RAG + GPT-4 reasoning
    
    For each requirement, this class:
    1. Retrieves relevant document context using RAG
    2. Uses GPT-4 to analyze the context against the requirement
    3. Returns structured evaluation results
    """
    
    def __init__(self, vector_store, model_name: str = None, api_provider: str = "auto", registry: str = "puro"):
        self.vector_store = vector_store
        self.registry = registry.lower()
        
        # Determine API provider and model based on available keys
        self.api_provider, self.model_name = self._determine_api_provider(api_provider, model_name)
        
        # Initialize LLM based on provider
        if self.api_provider == "groq":
            self.llm = self._init_groq_llm()
        else:  # OpenAI
            self.llm = self._init_openai_llm()
        
        logger.info(f"Initialized evaluator for {self.registry.upper()} registry with {self.api_provider} using model: {self.model_name}")
    
    def _determine_api_provider(self, api_provider: str, model_name: str = None) -> tuple[str, str]:
        """Determine which API provider to use based on available keys and preferences"""
        groq_key = os.getenv('GROQ_API_KEY')
        openai_key = os.getenv('OPENAI_API_KEY')
        
        if api_provider == "groq" and groq_key:
            return "groq", model_name or "llama-3.1-8b-instant"
        elif api_provider == "openai" and openai_key:
            return "openai", model_name or "gpt-4o"
        elif api_provider == "auto":
            # Auto-detect based on available keys, prefer Groq for cost efficiency
            if groq_key:
                return "groq", model_name or "llama-3.1-8b-instant"
            elif openai_key:
                return "openai", model_name or "gpt-4o"
            else:
                raise ValueError("No API keys found. Please set GROQ_API_KEY or OPENAI_API_KEY environment variable.")
        else:
            raise ValueError(f"Invalid API provider '{api_provider}' or missing API key")
    
    def _init_groq_llm(self):
        """Initialize Groq LLM"""
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model=self.model_name,
                temperature=0.1,
                max_tokens=1000,
                groq_api_key=os.getenv('GROQ_API_KEY')
            )
        except ImportError:
            logger.error("langchain-groq not installed. Install with: pip install langchain-groq")
            raise ImportError("Please install langchain-groq: pip install langchain-groq")
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM"""
        return ChatOpenAI(
            model=self.model_name,
            temperature=0.1,
            max_tokens=1000,
            openai_api_key=os.getenv('OPENAI_API_KEY')
        )
    
    def evaluate_requirement(self, checklist_item: Dict[str, Any]) -> 'EvaluationResult':
        """
        Evaluate a single checklist requirement using RAG + GPT-4
        
        Args:
            checklist_item: Dict containing requirement details from checklist
            
        Returns:
            EvaluationResult with evaluation details
        """
        from .agent import EvaluationResult  # Import here to avoid circular import
        
        # Handle both "requirement" and "parameter" fields
        requirement = checklist_item.get('requirement', checklist_item.get('parameter', 'Unknown requirement'))
        
        # Handle different JSON key structures
        puro_requires = checklist_item.get('puroRequires', checklist_item.get('puroLooksFor', checklist_item.get('requirement', 'Not specified')))
        documents_needed = checklist_item.get('documentsNeeded', 'Not specified')
        puro_checks_for = checklist_item.get('puroWillCheckFor', checklist_item.get('puroLooksFor', checklist_item.get('puroWillCheck', 'Not specified')))
        
        logger.debug(f"Evaluating requirement: {requirement}")
        
        # Step 1: Use RAG to get relevant document context
        search_queries = self._generate_search_queries(checklist_item)
        relevant_context = []
        
        for query in search_queries:
            # Use fewer documents for Groq to reduce token usage
            k = 2 if self.api_provider == "groq" else 3
            docs = get_relevant_docs(query, self.vector_store, k=k)
            for doc in docs:
                relevant_context.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', 'unknown'),
                    'query': query
                })
        
        # Step 2: Create evaluation prompt
        evaluation_prompt = create_evaluation_prompt(
            checklist_item, relevant_context, self.api_provider, self.registry
        )
        
        # Step 3: Get GPT-4 evaluation with retry logic
        evaluation_result = self._evaluate_with_retry(
            checklist_item, evaluation_prompt, requirement, relevant_context
        )
        
        return evaluation_result
    
    def _evaluate_with_retry(
        self, 
        checklist_item: Dict[str, Any], 
        evaluation_prompt: str, 
        requirement: str, 
        relevant_context: List[Dict[str, Any]],
        max_retries: int = 5
    ) -> 'EvaluationResult':
        """
        Evaluate with retry logic for API errors (especially Groq rate limits and token limits)
        
        Args:
            checklist_item: The checklist item being evaluated
            evaluation_prompt: The prompt to send to the LLM
            requirement: The requirement text
            relevant_context: The retrieved context
            max_retries: Maximum number of retry attempts (default: 5)
            
        Returns:
            EvaluationResult: The evaluation result or error result if all retries fail
        """
        from .agent import EvaluationResult  # Import here to avoid circular import
        
        documents_needed = checklist_item.get('documentsNeeded', 'Not specified')
        
        for attempt in range(max_retries + 1):  # +1 because we want max_retries actual retries after the first attempt
            try:
                logger.debug(f"Evaluation attempt {attempt + 1}/{max_retries + 1} for requirement: {requirement}")
                
                response = self.llm.invoke([
                    SystemMessage(content=get_system_prompt_for_registry(self.registry)),
                    HumanMessage(content=evaluation_prompt)
                ])
                
                # If we get here, the API call was successful
                evaluation_result = self._parse_evaluation_response(
                    response.content, requirement, relevant_context
                )
                
                if attempt > 0:
                    logger.info(f"Successfully evaluated requirement '{requirement}' after {attempt} retries")
                
                return evaluation_result
                
            except Exception as e:
                error_message = str(e).lower()
                
                # Check if this is a recoverable API error
                is_rate_limit_error = any(code in error_message for code in ['429', 'rate limit', 'rate_limit'])
                is_token_limit_error = any(code in error_message for code in ['413', 'token limit', 'token_limit', 'context length'])
                is_api_error = any(code in error_message for code in ['400', '401', '403', '500', '502', '503', '504'])
                
                is_recoverable_error = is_rate_limit_error or is_token_limit_error or is_api_error
                
                if attempt < max_retries and is_recoverable_error:
                    # Calculate exponential backoff delay
                    delay = min(60, (2 ** attempt) + (0.1 * attempt))  # Cap at 60 seconds
                    
                    logger.warning(
                        f"API error on attempt {attempt + 1}/{max_retries + 1} for requirement '{requirement}': {e}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )
                    
                    time.sleep(delay)
                    continue
                else:
                    # Either we've exhausted retries or it's a non-recoverable error
                    if is_recoverable_error:
                        logger.error(
                            f"Failed to evaluate requirement '{requirement}' after {max_retries} retries. "
                            f"Final error: {e}"
                        )
                        error_reason = f"API error after {max_retries} retries: {str(e)}"
                    else:
                        logger.error(f"Non-recoverable error evaluating requirement '{requirement}': {e}")
                        error_reason = f"Non-recoverable error: {str(e)}"
                    
                    # Return error result - this will terminate evaluation for this requirement
                    return EvaluationResult(
                        requirement=requirement,
                        status="unclear",
                        reason=error_reason,
                        evidence_found=[],
                        missing_evidence=[documents_needed],
                        confidence_score=0.0
                    )
        
        # This should never be reached, but just in case
        return EvaluationResult(
            requirement=requirement,
            status="unclear",
            reason="Unknown error: Maximum retries exceeded",
            evidence_found=[],
            missing_evidence=[documents_needed],
            confidence_score=0.0
        )
    
    def _generate_search_queries(self, checklist_item: Dict[str, Any]) -> List[str]:
        """Generate search queries for RAG retrieval using checklist-defined keywords and context"""
        # Handle both "requirement" and "parameter" fields
        requirement = checklist_item.get('requirement', checklist_item.get('parameter', ''))
        puro_requires = checklist_item.get('puroRequires', checklist_item.get('puroLooksFor', checklist_item.get('requirement', '')))
        
        # Start with basic searches
        queries = [
            # Direct requirement search
            requirement,
            # Specific Puro requirement (if available)
            puro_requires,
            # Combined search
            f"{requirement} {puro_requires}",
        ]
        
        # Filter out empty queries
        queries = [q.strip() for q in queries if q.strip()]
        
        # Add search keywords from checklist if available
        search_keywords = checklist_item.get('searchKeywords', [])
        if search_keywords:
            queries.extend(search_keywords)
        else:
            # Fallback: if no keywords in checklist, use basic keyword extraction
            # This maintains backward compatibility with old checklists
            queries.extend(self._extract_fallback_keywords(requirement, puro_requires))
        
        # Extract additional search terms from context fields
        context_search_terms = self._extract_context_search_terms(checklist_item)
        queries.extend(context_search_terms)
        
        return list(set(queries))  # Remove duplicates
    
    def _extract_context_search_terms(self, checklist_item: Dict[str, Any]) -> List[str]:
        """Extract search terms from additional context fields"""
        search_terms = []
        
        # Fields that might contain searchable terms
        searchable_fields = [
            'context', 'evaluationContext', 'notes', 'technicalDetails',
            'acceptableEvidence', 'commonIssues'
        ]
        
        for field in searchable_fields:
            value = checklist_item.get(field)
            if not value:
                continue
                
            if isinstance(value, str):
                # Extract key terms from text (avoid too generic words)
                words = value.split()
                meaningful_terms = [
                    word.strip('.,()[]') for word in words 
                    if len(word) > 4 and word.lower() not in {
                        'should', 'would', 'could', 'these', 'those', 'their', 'there', 
                        'where', 'which', 'while', 'during', 'before', 'after'
                    }
                ]
                search_terms.extend(meaningful_terms[:5])  # Limit to avoid too many terms
                
            elif isinstance(value, list):
                # Handle list of strings
                for item in value:
                    if isinstance(item, str):
                        search_terms.append(item)
                        
            elif isinstance(value, dict):
                # Handle nested dictionaries (like technicalDetails)
                for subkey, subvalue in value.items():
                    if isinstance(subvalue, (str, list)):
                        if isinstance(subvalue, str):
                            search_terms.append(subvalue)
                        else:
                            search_terms.extend([str(item) for item in subvalue])
        
        return search_terms[:10]  # Limit total additional terms
    
    def _extract_fallback_keywords(self, requirement: str, puro_requires: str) -> List[str]:
        """Fallback keyword extraction for backward compatibility"""
        req_lower = requirement.lower()
        puro_requires_lower = puro_requires.lower()
        fallback_keywords = []
        
        if 'ratio' in req_lower or 'h/c' in req_lower:
            fallback_keywords.extend(['carbon ratio', 'H/C ratio', 'H/Corg ratio', 'oxygen carbon', 'molar ratio'])
        elif 'biomass' in req_lower or 'feedstock' in req_lower:
            fallback_keywords.extend(['feedstock', 'biomass source', 'raw material', 'wood chips', 'agricultural residue'])
        elif 'process' in req_lower or 'pyrolysis' in req_lower:
            fallback_keywords.extend(['pyrolysis', 'gasification', 'reactor', 'process', 'combustion', 'temperature'])
        elif 'additionality' in req_lower:
            fallback_keywords.extend(['project not mandated by law', 'carbon finance dependency', 'baseline analysis', 'financial viability', 'IRR'])
        elif 'emission' in req_lower:
            fallback_keywords.extend(['LCA', 'ISO 14040', 'ISO 14044', 'greenhouse gases', 'emissions', 'pollutants', 'environmental'])
        elif 'stability' in req_lower:
            fallback_keywords.extend(['H/Corg ratio', 'carbon permanence', 'Fp factor', 'organic carbon', 'biochar stability'])
        elif 'end-use' in req_lower or 'offtake' in req_lower:
            fallback_keywords.extend(['offtake agreement', 'non-fuel use', 'use in soil or construction', 'biochar application', 'soil amendment'])
        elif 'social' in req_lower or 'stakeholder' in req_lower:
            fallback_keywords.extend(['EIA', 'community consent', 'grievance policy', 'stakeholder consultation', 'public support'])
        elif 'safety' in req_lower:
            fallback_keywords.extend(['safety protocols', 'handling procedures', 'storage', 'transport', 'fire control'])
        elif 'waste' in req_lower or 'management' in req_lower:
            fallback_keywords.extend(['waste management', 'oil', 'tars', 'wastewater', 'ash', 'environmental impact'])
        elif 'yield' in req_lower:
            fallback_keywords.extend(['yield', 'production', 'output', 'efficiency'])
        elif 'sampling' in req_lower:
            fallback_keywords.extend(['sampling protocol', 'sample collection', 'testing procedure', 'quality control'])
        elif 'monitoring' in req_lower:
            fallback_keywords.extend(['monitoring', 'tracking', 'measurement', 'data collection', 'MRV'])
        
        return fallback_keywords
    
    def _parse_evaluation_response(
        self, 
        response: str, 
        requirement: str, 
        context: List[Dict[str, Any]]
    ) -> 'EvaluationResult':
        """Parse GPT-4 response into structured result with improved regex parsing"""
        from .agent import EvaluationResult  # Import here to avoid circular import
        
        # Extract unique document sources from context for reference
        document_sources = list(set([ctx.get('source', 'unknown') for ctx in context]))
        source_info = f" (Sources: {', '.join([src.split('/')[-1] for src in document_sources[:3]])}{'...' if len(document_sources) > 3 else ''})" if document_sources else ""
        
        # Default values
        status = "unclear"
        reason = "Unable to parse evaluation response"
        evidence_found = []
        missing_evidence = []
        confidence_score = 0.0
        
        try:
            # Use regex for more robust parsing
            status_match = re.search(r"STATUS:\s*(present|missing|unclear)", response, re.IGNORECASE)
            if status_match:
                status = status_match.group(1).lower()
            
            reason_match = re.search(r"REASON:\s*(.+?)(?=\n(?:EVIDENCE|MISSING|CONFIDENCE|$))", response, re.DOTALL | re.IGNORECASE)
            if reason_match:
                reason = reason_match.group(1).strip()
            
            evidence_match = re.search(r"EVIDENCE:\s*(.+?)(?=\n(?:MISSING|CONFIDENCE|$))", response, re.DOTALL | re.IGNORECASE)
            if evidence_match:
                evidence_text = evidence_match.group(1).strip()
                if evidence_text.lower() not in ['none', 'n/a', 'not found']:
                    # Add source information to evidence
                    evidence_with_sources = evidence_text + source_info
                    evidence_found = [evidence_with_sources]
            
            missing_match = re.search(r"MISSING:\s*(.+?)(?=\n(?:CONFIDENCE|$))", response, re.DOTALL | re.IGNORECASE)
            if missing_match:
                missing_text = missing_match.group(1).strip()
                if missing_text.lower() not in ['none', 'n/a']:
                    missing_evidence = [missing_text]
            
            confidence_match = re.search(r"CONFIDENCE:\s*([0-9]*\.?[0-9]+)", response, re.IGNORECASE)
            if confidence_match:
                try:
                    confidence_score = float(confidence_match.group(1))
                    # Ensure confidence is between 0.0 and 1.0
                    confidence_score = max(0.0, min(1.0, confidence_score))
                except ValueError:
                    confidence_score = 0.5
            
            # Fallback to line-by-line parsing if regex fails
            if not status_match:
                lines = response.strip().split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('STATUS:'):
                        status = line.split(':', 1)[1].strip().lower()
                    elif line.startswith('REASON:'):
                        reason = line.split(':', 1)[1].strip()
                    elif line.startswith('EVIDENCE:'):
                        evidence_text = line.split(':', 1)[1].strip()
                        if evidence_text.lower() not in ['none', 'n/a']:
                            # Add source information to evidence in fallback parsing
                            evidence_with_sources = evidence_text + source_info
                            evidence_found = [evidence_with_sources]
                    elif line.startswith('MISSING:'):
                        missing_text = line.split(':', 1)[1].strip()
                        if missing_text.lower() not in ['none', 'n/a']:
                            missing_evidence = [missing_text]
                    elif line.startswith('CONFIDENCE:'):
                        try:
                            confidence_score = float(line.split(':', 1)[1].strip())
                            confidence_score = max(0.0, min(1.0, confidence_score))
                        except ValueError:
                            confidence_score = 0.5
            
            # Validate status
            if status not in ['present', 'missing', 'unclear']:
                status = 'unclear'
                
        except Exception as e:
            logger.error(f"Error parsing evaluation response: {e}")
            reason = f"Error parsing response: {str(e)}"
        
        return EvaluationResult(
            requirement=requirement,
            status=status,
            reason=reason,
            evidence_found=evidence_found,
            missing_evidence=missing_evidence,
            confidence_score=confidence_score
        )
