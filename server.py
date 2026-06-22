import os
from dotenv import load_dotenv

# 1. Load environment variables before doing ANY local imports
load_dotenv()  

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# 2. Import the graph generator function from your file
from src.graph import create_graph

# 3. Instantiate your compiled graph globally inside the server script
graph = create_graph()

app = FastAPI(title="My Self-Hosted RAG Agent")

class QueryPayload(BaseModel):
    message: str

@app.post("/query")
async def query_agent(payload: QueryPayload):
    async def event_generator():
        # Because your graph uses MessagesState, it expects a list of message objects
        inputs = {"messages": [{"role": "user", "content": payload.message}]}
        
        async for event in graph.astream(inputs, stream_mode="updates"):
            yield f"data: {event}\n\n"
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")