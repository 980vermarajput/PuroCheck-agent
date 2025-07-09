from agent.agent import PuroCheckAgent
import sys

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
    
    Usage:
        python main.py [registry]
        
    Where registry can be:
        - puro (default): Use Puro.earth biochar checklist
        - verra: Use Verra VM0047 checklist
    """
    
    # Parse command line arguments
    registry = "puro"  # Default
    if len(sys.argv) > 1:
        registry = sys.argv[1].lower()
        if registry not in ["puro", "verra", "vcs"]:
            print(f"❌ Unknown registry: {registry}")
            print("Available registries: puro, verra")
            return 1
    
    print(f"🚀 Starting PuroCheck AI - {registry.upper()} Registry Eligibility Agent")
    print("=" * 70)
    
    try:
        # Initialize the agent with registry selection
        agent = PuroCheckAgent(
            data_dir="data/",
            registry=registry,  # Dynamic registry selection
            vector_store_dir="chroma_db/",
            force_rebuild_vectorstore=False  # Set to False since we've already built it
        )
        
        # Run the complete evaluation pipeline
        reports = agent.run_full_evaluation(output_format="both")
        
        # Display CLI report
        if "cli" in reports:
            print(reports["cli"])
        
        # Save JSON report
        output_file = f"evaluation_results_{registry}.json"
        if "json" in reports:
            with open(output_file, "w") as f:
                f.write(reports["json"])
            print(f"\n💾 Detailed results saved to: {output_file}")
        
        print("\n🎉 Evaluation completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
