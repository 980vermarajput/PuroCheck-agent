# PuroCheck Agent

A sophisticated RAG (Retrieval-Augmented Generation) agent that evaluates biochar project eligibility against Puro.earth standards. The agent uses advanced document parsing, vector search, and LLM reasoning to assess compliance with biochar methodology requirements.

## Features

- 🔍 **Multi-format Document Processing**: Supports PDF, DOCX, TXT, CSV, and Excel files
- 🤖 **Dual LLM Support**: Compatible with OpenAI GPT and Groq models
- 📊 **Vector Search**: ChromaDB-powered semantic search for relevant document sections
- ✅ **Compliance Evaluation**: Automated assessment against Puro.earth biochar methodology
- 📈 **Detailed Reporting**: Comprehensive evaluation results with evidence tracking
- 🔄 **Flexible Configuration**: Customizable checklists and search parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd purocheck-agent
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the root directory and add your API keys:

```bash
# For OpenAI (if using GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# For Groq (if using Groq models)
GROQ_API_KEY=your_groq_api_key_here

# Optional: Set default model provider (auto, openai, or groq)
DEFAULT_API_PROVIDER=auto
```

**Note**: You need at least one API key. The agent will automatically detect which provider to use based on available keys.

### Step 5: Prepare Your Documents

Place your project documents in the `data/` folder. Supported formats:

- PDF files (`.pdf`)
- Word documents (`.docx`)
- Text files (`.txt`)
- CSV files (`.csv`)
- Excel files (`.xlsx`)

## Usage

### Basic Usage

Run the agent with default settings:

```bash
python main.py
```

### Advanced Usage

You can customize the evaluation by modifying the parameters in `main.py` or by using command-line arguments:

```python
# Example: Force rebuild vector store and use specific model
results = agent.evaluate_checklist(
    checklist_path="checklist/sample_checklist.json",
    force_rebuild_vectorstore=True,
    api_provider="openai",  # or "groq" or "auto"
    model_name="gpt-4"      # or specific Groq model
)
```

### Configuration Options

- **`force_rebuild_vectorstore`**: Set to `True` to rebuild the vector database from scratch
- **`api_provider`**: Choose between "openai", "groq", or "auto" (default)
- **`model_name`**: Specify the exact model to use (e.g., "gpt-4", "mixtral-8x7b-32768")

## Project Structure

```
purocheck-agent/
├── agent/                      # Core agent implementation
│   ├── agent.py               # Main agent orchestrator
│   ├── evaluator.py          # Evaluation logic
│   ├── nodes.py              # Document processing nodes
│   ├── graph.py              # Workflow graph definition
│   └── prompts.py            # LLM prompts
├── checklist/                 # Evaluation criteria
│   ├── sample_checklist.json # Default Puro.earth checklist
│   └── schema.py             # Checklist validation schema
├── data/                      # Document storage (ignored by git)
├── chroma_db/                # Vector database (ignored by git)
├── main.py                   # Application entry point
├── requirements.txt          # Python dependencies
└── .env                      # Environment variables (create this)
```

## API Keys Setup

### OpenAI API Key

1. Go to [OpenAI Platform](https://platform.openai.com/)
2. Create an account or log in
3. Navigate to API Keys section
4. Create a new API key
5. Add it to your `.env` file

### Groq API Key

1. Go to [Groq Console](https://console.groq.com/)
2. Create an account or log in
3. Generate an API key
4. Add it to your `.env` file

## Customizing Evaluation Criteria

You can modify the evaluation checklist by editing `checklist/sample_checklist.json`. Each requirement can include:

- **`requirement`**: Description of what needs to be checked
- **`searchKeywords`**: Keywords to guide document search
- **`category`**: Grouping for organization

## Output

The agent generates detailed evaluation results including:

- **Summary**: Overall project status and statistics
- **Section-by-section analysis**: Detailed findings for each requirement
- **Evidence tracking**: Source documents and confidence scores
- **Missing evidence**: What documentation might be needed

Results are saved to `evaluation_results.json`.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed via `pip install -r requirements.txt`
2. **API Key Errors**: Verify your API keys are correctly set in the `.env` file
3. **Vector Store Issues**: Try setting `force_rebuild_vectorstore=True` to rebuild the database
4. **Document Processing Errors**: Ensure documents are in supported formats and not corrupted

### Debug Mode

For detailed logging, you can modify the logging level in the code or check the console output for diagnostic information.
