
# 📄 Product Requirements Document (PRD)
**Project:** PuroCheck AI – Biochar Project Eligibility Agent  
**Owner:** Abhishek Verma  
**Created:** July 2025  

---

## 🎯 1. Objective

Develop an AI agent that can:
- Analyze one or more uploaded documents (project proposals, lab reports, certifications).
- Cross-check them against the **Puro.earth Biochar Methodology** requirements.
- Identify:
  - ✅ Which checklist items are satisfied.
  - ❌ Which are **missing** or **unclear**.
  - 📌 What additional documents or clarifications are needed.

---

## 🧱 2. Core Features

| Feature                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| Document Upload (local or UI)    | Supports uploading PDFs into `data/` folder.                               |
| Document Parsing + Chunking      | Extract text from PDF, split into searchable chunks.                       |
| Vector Store + Retrieval (RAG)   | Use Chroma + OpenAI embeddings to find relevant evidence.                  |
| Checklist Evaluation             | Each checklist item evaluated via RAG + GPT-4.                             |
| Output Report                    | JSON/table of checklist items with status (`present`, `missing`, `unclear`).|
| CLI-based MVP                   | Runs from `main.py` for now (later extensible to Streamlit or API).        |

---

## 🔍 3. Inputs

### 📄 Document Types:
- Technical specs (reactor design, feedstock, etc.)
- Lab reports (carbon ratio, heavy metals)
- Environmental permits
- LCA reports
- Stakeholder meeting notes
- Offtake contracts

### 📋 Checklist Source:
- `sample_checklist.json` (mirrors Puro.earth criteria)

---

## 🧠 4. AI/Tech Components

| Component          | Technology / Model                 |
|--------------------|------------------------------------|
| Language Model     | OpenAI GPT-4 (`gpt-4` or `gpt-4o`)  |
| Embeddings         | `text-embedding-3-small`           |
| RAG                | Chroma DB with LangChain Retriever |
| Orchestration      | LangGraph (optional, for future)   |
| Prompting          | Custom checklist-based prompt templates |
| Chunking           | RecursiveCharacterTextSplitter     |

---

## ✅ 5. Functional Requirements

| ID   | Requirement Description                                                                 |
|------|-------------------------------------------------------------------------------------------|
| FR1  | System shall parse all PDFs in the `/data` directory.                                    |
| FR2  | System shall chunk and embed content using OpenAI and store in Chroma.                   |
| FR3  | System shall load and iterate through checklist items in JSON format.                    |
| FR4  | System shall use relevant document context (via RAG) to evaluate each checklist item.    |
| FR5  | System shall return for each item: status, reason, and missing evidence if any.          |
| FR6  | System shall output results in both CLI log and structured JSON or table.                |

---

## 🚫 6. Non-Requirements (for MVP)

- No UI yet (Streamlit/dashboard comes later).
- No web uploading of files.
- No multi-user session handling.
- No external database (just local Chroma DB).

