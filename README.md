# ResumeGraph

ResumeGraph autonomously generates JD-tailored, ATS-optimized resumes using LLMs, graph-based execution (LangGraph), and hybrid vector search (Qdrant). It prevents hallucination by retrieving only verified facts from your knowledge base and matching them to specific job requirements.

## Pipeline

```
JD Text → [Extract Requirements] → [Retrieve Bullets] → [Draft Resume] → [Generate PDF]
```

| Node | What it does | Tech |
|------|-------------|------|
| **1 — Extract** | Parses JD into structured skills, experience, responsibilities | Gemini 2.5 Flash + Pydantic |
| **2 — Retrieve** | Fetches top-3 bullets/company, top-2/project from vector DB; reorders skills | Qdrant Hybrid Search |
| **3 — Draft** | Rewrites bullets for ATS alignment without hallucination | Gemini 2.5 Flash + Structured Output |
| **PDF Gen** | Merges drafted content with static KB, compiles LaTeX | Jinja2 + pdflatex (MiKTeX) |

## Quick Start

### Prerequisites
- Python 3.10+
- Docker (for Qdrant)
- MiKTeX (`winget install MiKTeX.MiKTeX`)
- Gemini API key in `.env` file (`GOOGLE_API_KEY=...`)

### Setup
```powershell
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start Qdrant
docker run -d -p 6333:6333 -p 6334:6334 -v F:\ResumeGraph\qdrant_storage:/qdrant/storage --name qdrant qdrant/qdrant

# Ingest your resume knowledge base
python src\ingest.py
```

### Run
```powershell
# Make sure Qdrant is running
docker start qdrant

# Generate a tailored resume
.\venv\Scripts\python.exe src\workflow.py
```

Output → `out/final_resume.pdf`

## Project Structure

```
ResumeGraph/
├── src/
│   ├── state.py          # TypedDict state + Pydantic models
│   ├── nodes.py          # LangGraph nodes (extract, retrieve, draft)
│   ├── workflow.py       # Graph definition + orchestration entry point
│   ├── ingest.py         # Qdrant data ingestion script
│   └── generate_pdf.py   # Jinja2 → LaTeX → PDF compilation
├── resume_kb/            # Your resume knowledge base
│   ├── personal_info.json
│   ├── education.json
│   ├── skills.json
│   ├── publications.json
│   ├── experience/       # One JSON per company
│   └── projects/         # One JSON per project
├── latex_template/
│   └── jinja_template.tex  # Unified Jinja2 LaTeX template
├── out/                  # Generated output (PDF, TEX)
├── requirements.txt
├── .env                  # API keys (not committed)
└── TECHNICAL_README.md   # Deep technical architecture guide
```

## Task Status

- [x] Vector Database Setup & Data Ingestion (Qdrant)
- [x] LangGraph Workflow — Nodes 1, 2, 3
- [x] LaTeX PDF Generation via Jinja2 + pdflatex
- [ ] Critique & Fact-Check Node (hallucination prevention loop)