from agent.agent import PuroCheckAgent

def main():
    """
    Main entry point for PuroCheck AI Agent
    
    This implements the complete pipeline according to the PRD:
    - FR1: Parse PDFs from /data directory
    - FR2: Chunk and embed with OpenAI + Chroma
    - FR3: Load checklist items from JSON
    - FR4: Use RAG to evaluate each requirement
    - FR5: Return status, reason, and evidence for each item
    - FR6: Output results in CLI and JSON formats
    """
    
    print("🚀 Starting PuroCheck AI - Biochar Project Eligibility Agent")
    print("=" * 70)
    
    try:
        # Initialize the agent
        agent = PuroCheckAgent(
            data_dir="data/",
            checklist_path="checklist/sample_checklist.json",
            vector_store_dir="chroma_db/",
            force_rebuild_vectorstore=True  # Set to False since we've already built it
        )
        
        # Run the complete evaluation pipeline
        reports = agent.run_full_evaluation(output_format="both")
        
        # Display CLI report
        if "cli" in reports:
            print(reports["cli"])
        
        # Save JSON report
        if "json" in reports:
            with open("evaluation_results.json", "w") as f:
                f.write(reports["json"])
            print("\n💾 Detailed results saved to: evaluation_results.json")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
