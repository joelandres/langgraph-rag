import requests
import json

url = "http://127.0.0.1:8000/query"
payload = {"message": "What does Lilian Weng say about types of reward hacking?"}

print("📡 Sending request to self-hosted RAG API...")
print("-" * 60)

# Use stream=True to handle the live server-sent events chunk by chunk
with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            # Decode the server bytes into a readable string
            decoded_line = line.decode('utf-8')
            
            # Filter out standard SSE data prefixes
            if decoded_line.startswith("data: "):
                clean_json_str = decoded_line.replace("data: ", "")
                
                try:
                    # Parse the string into a readable Python dictionary
                    node_data = json.loads(clean_json_str.replace("'", '"'))
                    
                    for node_name, state in node_data.items():
                        print(f"\n📍 Node Executed: {node_name}")
                        print("-" * 30)
                        # Print the latest messages appended to the graph state
                        if "messages" in state:
                            last_msg = state["messages"][-1]
                            print(f"[{last_msg.get('type', 'message')}]: {last_msg.get('content')}")
                except Exception:
                    # Fallback to printing raw text if it contains custom objects
                    print(decoded_line)