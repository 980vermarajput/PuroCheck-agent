"""
PuroCheck AI Agent - Main Orchestrator

This module contains the core agent logic that orchestrates the entire
biochar project eligibility checking process according to Puro.earth requirements.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

from .nodes import (
    load_documents_from_folder,
    chunk_documents,
    create_vector_store,
    load_vector_store,
    get_relevant_docs
)
from .evaluator import ChecklistEvaluator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Data class for storing evaluation results for each checklist item"""
    requirement: str
    status: str  # 'present', 'missing', 'unclear'
    reason: str
    evidence_found: List[str]
    missing_evidence: List[str]
    confidence_score: float


class PuroCheckAgent:
    """
    Main agent orchestrator for biochar project eligibility checking.
    
    This agent coordinates the entire pipeline:
    1. Document processing (FR1, FR2)
    2. Checklist loading (FR3)
    3. RAG-based evaluation (FR4)
    4. Results compilation (FR5)
    5. Output generation (FR6)
    """
    
    def __init__(
        self,
        data_dir: str = "data/",
        checklist_path: str = None,
        registry: str = "puro",
        vector_store_dir: str = "chroma_db/",
        force_rebuild_vectorstore: bool = False,
        api_provider: str = "auto",
        model_name: str = None
    ):
        self.data_dir = Path(data_dir)
        self.registry = registry.lower()
        
        # Set default checklist path based on registry if not provided
        if checklist_path is None:
            checklist_path = self._get_default_checklist_path(self.registry)
        self.checklist_path = Path(checklist_path)
        
        self.vector_store_dir = vector_store_dir
        self.force_rebuild_vectorstore = force_rebuild_vectorstore
        self.api_provider = api_provider
        self.model_name = model_name
        
        # Initialize components
        self.vector_store = None
        self.checklist = None
        self.evaluator = None
        
        logger.info(f"🚀 PuroCheck Agent initialized for {self.registry.upper()} registry with API provider: {api_provider}")
    
    def initialize(self) -> None:
        """Initialize all components of the agent"""
        logger.info("🔧 Initializing agent components...")
        
        # Step 1: Load and process documents (FR1, FR2)
        self._setup_vector_store()
        
        # Step 2: Load checklist (FR3)
        self._load_checklist()
        
        # Step 3: Initialize evaluator (FR4)
        self._setup_evaluator()
        
        logger.info("✅ Agent initialization complete")
    
    def _setup_vector_store(self) -> None:
        """Setup vector store with document processing (FR1, FR2)"""
        logger.info("📄 Processing documents and setting up vector store...")
        
        try:
            # Check if vector store exists and we don't want to rebuild
            vector_store_path = Path(self.vector_store_dir)
            if vector_store_path.exists() and not self.force_rebuild_vectorstore:
                logger.info("📚 Loading existing vector store...")
                self.vector_store = load_vector_store(self.vector_store_dir, self.api_provider)
            else:
                logger.info("🔨 Building new vector store from documents...")
                
                # FR1: Parse all documents in the /data directory
                docs = load_documents_from_folder(str(self.data_dir))
                if not docs:
                    raise ValueError(f"No documents found in {self.data_dir}")
                
                logger.info(f"📖 Loaded {len(docs)} documents")
                
                # FR2: Chunk and embed content using specified API provider and store in Chroma
                chunks = chunk_documents(docs)
                logger.info(f"✂️ Created {len(chunks)} document chunks")
                
                self.vector_store = create_vector_store(
                    chunks, 
                    self.vector_store_dir, 
                    self.api_provider, 
                    force_rebuild=self.force_rebuild_vectorstore
                )
                logger.info("💾 Vector store created and persisted")
        except Exception as e:
            logger.error(f"Error setting up vector store: {str(e)}", exc_info=True)
            raise
    
    def _load_checklist(self) -> None:
        """Load checklist items from JSON (FR3)"""
        logger.info("📋 Loading checklist...")
        
        if not self.checklist_path.exists():
            raise FileNotFoundError(f"Checklist file not found: {self.checklist_path}")
        
        with open(self.checklist_path, 'r') as f:
            self.checklist = json.load(f)
        
        # Count total checklist items
        total_items = sum(len(section['items']) for section in self.checklist['sections'])
        logger.info(f"📝 Loaded checklist with {total_items} requirements across {len(self.checklist['sections'])} sections")
    
    def _setup_evaluator(self) -> None:
        """Initialize the checklist evaluator (FR4)"""
        logger.info("🧠 Setting up checklist evaluator...")
        self.evaluator = ChecklistEvaluator(
            self.vector_store, 
            model_name=self.model_name, 
            api_provider=self.api_provider,
            registry=self.registry
        )
    
    def _is_critical_api_failure(self, evaluation: EvaluationResult) -> bool:
        """
        Check if an evaluation result indicates a critical API failure that should terminate evaluation
        
        Args:
            evaluation: The evaluation result to check
            
        Returns:
            bool: True if this is a critical failure that should stop evaluation
        """
        # Check if the reason indicates an API error after retries
        reason_lower = evaluation.reason.lower()
        is_api_error_after_retries = (
            "api error after" in reason_lower and "retries" in reason_lower
        )
        
        # Also check for non-recoverable errors
        is_non_recoverable = "non-recoverable error" in reason_lower
        
        return is_api_error_after_retries or is_non_recoverable
    def evaluate_project(self, progress_callback=None) -> Dict[str, Any]:
        """
        Main method to evaluate project against checklist (FR4, FR5)
        
        Args:
            progress_callback: Optional callback function to receive progress updates
        
        Returns:
            Dict containing evaluation results for all checklist items
        """
        if not all([self.vector_store, self.checklist, self.evaluator]):
            raise RuntimeError("Agent not properly initialized. Call initialize() first.")
        
        logger.info("🔍 Starting project evaluation...")
        
        results = {
            "summary": {
                "total_requirements": 0,
                "present": 0,
                "missing": 0,
                "unclear": 0,
                "overall_status": "unknown",
                "evaluation_completed": True,
                "termination_reason": None
            },
            "sections": []
        }
        
        # Iterate through all checklist sections and items (FR3)
        for section in self.checklist['sections']:
            logger.info(f"📊 Evaluating section: {section['title']}")
            
            # Send section start progress update
            if progress_callback:
                progress_callback({
                    'type': 'section_start',
                    'section_title': section['title'],
                    'total_sections': len(self.checklist['sections'])
                })
            
            section_results = {
                "title": section['title'],
                "items": []
            }
            
            for item in section['items']:
                logger.info(f"  🔎 Evaluating: {item['requirement']}")
                
                # Send item start progress update
                if progress_callback:
                    progress_callback({
                        'type': 'item_start',
                        'requirement': item['requirement'],
                        'current_item': results['summary']['total_requirements'] + 1
                    })
                
                # FR4: Use relevant document context (via RAG) to evaluate each checklist item
                evaluation = self.evaluator.evaluate_requirement(item)
                
                # Check if this is a critical API failure that should terminate evaluation
                if self._is_critical_api_failure(evaluation):
                    logger.error(f"💥 Critical API failure detected. Terminating evaluation.")
                    logger.error(f"   Failed requirement: {evaluation.requirement}")
                    logger.error(f"   Failure reason: {evaluation.reason}")
                    
                    # Add the failed evaluation to results
                    section_results['items'].append(evaluation.__dict__)
                    results['summary']['total_requirements'] += 1
                    results['summary'][evaluation.status] += 1
                    
                    # Mark evaluation as incomplete and set termination reason
                    results['summary']['evaluation_completed'] = False
                    results['summary']['termination_reason'] = f"API failure on requirement: {evaluation.requirement}"
                    results['summary']['overall_status'] = "evaluation_failed"
                    
                    # Add current section to results (even if incomplete)
                    results['sections'].append(section_results)
                    
                    logger.warning("⚠️  Evaluation terminated due to API failure. Partial results available.")
                    return results
                
                section_results['items'].append(evaluation.__dict__)
                
                # Update summary counters
                results['summary']['total_requirements'] += 1
                results['summary'][evaluation.status] += 1
                
                # Send item completion progress update
                if progress_callback:
                    progress_callback({
                        'type': 'item_complete',
                        'requirement': evaluation.requirement,
                        'status': evaluation.status,
                        'reason': evaluation.reason,
                        'evidence_found': evaluation.evidence_found,
                        'missing_evidence': evaluation.missing_evidence,
                        'confidence_score': float(evaluation.confidence_score),  # Ensure it's a float
                        'current_item': results['summary']['total_requirements'],
                        'summary': {
                            'present': results['summary']['present'],
                            'missing': results['summary']['missing'],
                            'unclear': results['summary']['unclear'],
                            'total': results['summary']['total_requirements']
                        }
                    })
                
                # Log result
                status_emoji = {"present": "✅", "missing": "❌", "unclear": "⚠️"}
                logger.info(f"    {status_emoji.get(evaluation.status, '❓')} {evaluation.status.upper()}: {evaluation.reason}")
            
            results['sections'].append(section_results)
        
        # Calculate overall status (only if evaluation completed successfully)
        if results['summary']['evaluation_completed']:
            present_ratio = results['summary']['present'] / results['summary']['total_requirements']
            if present_ratio >= 0.8:
                results['summary']['overall_status'] = "likely_eligible"
            elif present_ratio >= 0.5:
                results['summary']['overall_status'] = "needs_review"
            else:
                results['summary']['overall_status'] = "likely_ineligible"
        
        logger.info("🏁 Project evaluation completed")
        return results
    
    def generate_report(self, results: Dict[str, Any], output_format: str = "both") -> Dict[str, str]:
        """
        Generate output report (FR6)
        
        Args:
            results: Evaluation results from evaluate_project()
            output_format: "cli", "json", or "both"
        
        Returns:
            Dict containing report content in requested formats
        """
        reports = {}
        
        if output_format in ["cli", "both"]:
            reports["cli"] = self._generate_cli_report(results)
        
        if output_format in ["json", "both"]:
            reports["json"] = json.dumps(results, indent=2)
        
        return reports
    
    def _generate_cli_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable CLI report"""
        summary = results['summary']
        
        report = []
        report.append("\n" + "="*80)
        report.append("🎯 PURO.EARTH BIOCHAR PROJECT EVALUATION REPORT")
        report.append("="*80)
        
        # Summary section
        report.append(f"\n📊 SUMMARY:")
        report.append(f"  Total Requirements: {summary['total_requirements']}")
        report.append(f"  ✅ Present: {summary['present']}")
        report.append(f"  ❌ Missing: {summary['missing']}")
        report.append(f"  ⚠️  Unclear: {summary['unclear']}")
        report.append(f"  🎯 Overall Status: {summary['overall_status'].replace('_', ' ').title()}")
        
        # Add evaluation completion status
        if not summary.get('evaluation_completed', True):
            report.append(f"  ⛔ Evaluation Status: INCOMPLETE")
            if summary.get('termination_reason'):
                report.append(f"  🚨 Termination Reason: {summary['termination_reason']}")
        else:
            report.append(f"  ✅ Evaluation Status: COMPLETED")
        
        # Detailed results by section
        for section in results['sections']:
            report.append(f"\n📋 {section['title'].upper()}:")
            report.append("-" * 60)
            
            for item in section['items']:
                status_emoji = {"present": "✅", "missing": "❌", "unclear": "⚠️"}
                emoji = status_emoji.get(item['status'], '❓')
                
                report.append(f"\n{emoji} {item['requirement']}")
                report.append(f"   Status: {item['status'].upper()}")
                report.append(f"   Reason: {item['reason']}")
                
                if item['evidence_found']:
                    report.append(f"   Evidence: {', '.join(item['evidence_found'][:2])}...")
                
                if item['missing_evidence']:
                    report.append(f"   Missing: {', '.join(item['missing_evidence'][:2])}...")
        
        report.append("\n" + "="*80)
        return "\n".join(report)
    
    def run_full_evaluation(self, output_format: str = "both") -> Dict[str, str]:
        """
        Complete end-to-end evaluation pipeline
        
        Args:
            output_format: "cli", "json", or "both"
        
        Returns:
            Dict containing reports in requested formats
        """
        logger.info("🎬 Starting full evaluation pipeline...")
        
        # Initialize if not already done
        if not all([self.vector_store, self.checklist, self.evaluator]):
            self.initialize()
        
        # Run evaluation
        results = self.evaluate_project()
        
        # Generate reports
        reports = self.generate_report(results, output_format)
        
        logger.info("🎉 Full evaluation pipeline completed!")
        return reports
    
    def _get_default_checklist_path(self, registry: str) -> str:
        """Get the default checklist path for a given registry"""
        registry_checklists = {
            "puro": "checklist/puro_biochar_checklist.json",
            "verra": "checklist/verra_vm0047_checklist.json",
            "vcs": "checklist/verra_vm0047_checklist.json",  # Alias for Verra
        }
        return registry_checklists.get(registry, "checklist/puro_biochar_checklist.json")


def main():
    """Main entry point for the agent"""
    agent = PuroCheckAgent()
    reports = agent.run_full_evaluation()
    
    # Print CLI report if available
    if "cli" in reports:
        print(reports["cli"])
    
    # Save JSON report if available
    if "json" in reports:
        with open("evaluation_results.json", "w") as f:
            f.write(reports["json"])
        print("\n💾 Detailed results saved to evaluation_results.json")


if __name__ == "__main__":
    main()
