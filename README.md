# Agentic AI RAG Application

A Retrieval-Augmented Generation (RAG) system built using Python, LangGraph, and FastAPI.

---

## 🚀 Getting Started

Ensure your virtual environment is active and your `.env` file is configured before running these commands.

### 1. Run the Main Script (Terminal Mode)
```bash
uv run main.py
```

### 2. Launch LangGraph Development Studio
```bash
uv run langgraph dev
```

### 3. Query the Local LangGraph Developer Instance
```bash
uv run python query_agent.py
```

### 4. Start the Self-Hosted Production API Gateway
```bash
uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Test the Self-Hosted API Gateway Stream
```bash
uv run python test_api.py
```