from typing import Any

from langgraph import graph


def run_agentic_rag(query: str, graph: graph) -> dict[str, Any]:
    stream = graph.stream_events(
        {
            "messages": [
                {
                    "role": "user",
                    "content": f"{query}",
                }
            ]
        },
        version="v3",
    )

    # Tool-call requests from generate_query_or_respond have no text to
    # stream, so including it alongside generate_answer is safe: it only
    # ever contributes text when the model answers directly without
    # retrieving (skipping generate_answer entirely).
    final_answer_nodes = {"generate_query_or_respond", "generate_answer"}
    for message in stream.messages:
        if message.node not in final_answer_nodes:
            continue
        for token in message.text:
            print(token, end="", flush=True)
    print()

    # Iterating stream.messages above already drives the run to
    # completion, so this just returns the state that's already final.
    return stream.output
