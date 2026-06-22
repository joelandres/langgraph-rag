uv run main.py 
uv run langgraph dev

uv run python query_agent.py

uv run uvicorn server:app --host 0.0.0.0 --port 8000 --reload
uv run python test_api.py