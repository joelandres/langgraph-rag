import asyncio
from langgraph_sdk import get_client

async def main():
    # 1. Connect to your local LangGraph dev server
    client = get_client(url="http://127.0.0.1:2024")
    
    # 2. Get the assistant/graph name you configured in langgraph.json
    # (Using "agent" as defined in your json config)
    assistant_id = "my_agent"
    
    # 3. Create a unique thread to maintain conversational history
    thread = await client.threads.create()
    
    # 4. Define your RAG input query
    # Note: Ensure the key matching your graph's expected state input (e.g., 'messages' or 'query')
    inputs = {
        "messages": [
            {
                "role": "user", 
                "content": "What does Lilian Weng say about types of reward hacking?"
            }
        ]
    }
    
    print("🚀 Sending query to LangGraph RAG Agent...")
    print("-" * 50)
    
    # 5. Stream the results step-by-step as nodes finish executing
    async for chunk in client.runs.stream(
        thread_id=thread["thread_id"],
        assistant_id=assistant_id,
        input=inputs,
        stream_mode="updates" # 'updates' streams the state changes after each node finishes
    ):
        # Print out what each node is outputting
        if chunk.event == "updates":
            print(f"\n📍 Node Executed: {list(chunk.data.keys())}")
            print(chunk.data)

if __name__ == "__main__":
    asyncio.run(main())