from langchain_core.messages import convert_to_messages
from langgraph.graph import END

from src.nodes import (
    generate_answer,
    generate_query_or_respond,
    grade_documents,
    rewrite_question,
    route_on_tool_calls,
)
from src.tools import retrieve_blog_posts

def validate_setup(graph) -> None:
    # Validate graph node callables are defined.
    for fn in (
        retrieve_blog_posts,
        generate_query_or_respond,
        grade_documents,
        rewrite_question,
        generate_answer,
    ):
        assert callable(fn) or hasattr(fn, "invoke")

    # Validate routing helper behavior without hitting external APIs.
    no_tool_state = convert_to_messages([{"role": "assistant", "content": "hello"}])
    assert route_on_tool_calls({"messages": no_tool_state}) == END

    tool_call_state = convert_to_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            }
        ]
    )
    assert route_on_tool_calls({"messages": tool_call_state}) == "tools"

    # Validate graph object exists and compiled successfully.
    assert graph is not None
    assert graph.get_graph() is not None
