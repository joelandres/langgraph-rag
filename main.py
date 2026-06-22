import warnings

from dotenv import load_dotenv
load_dotenv()

from langchain_core._api.beta_decorator import LangChainBetaWarning

from src import graph
from src.chunks import split_docs
from src.documents import load_web_page
from src.rag import run_agentic_rag
from src.validate import validate_setup
from src.vectordb import init_retriever

# Known benign warning: langchain-core's with_structured_output() stores the
# parsed model in a field pydantic's schema types as None, which trips
# pydantic's serializer during tracing. Doesn't affect the parsed result.
warnings.filterwarnings(
    "ignore", message="Pydantic serializer warnings", category=UserWarning
)
# We intentionally use graph.stream_events(version="v3"), which is still
# marked experimental/beta in langgraph's Pregel implementation.
warnings.filterwarnings("ignore", category=LangChainBetaWarning)


def main():
    workflow = graph.create_graph()
    print(workflow.get_graph().draw_ascii())

    # validate_setup(workflow)

    print(f"Results:\n")
    query = "What does Lilian Weng say about types of reward hacking?"
    final_state = run_agentic_rag(query, workflow)

    print("\nFinal state:")
    for message in final_state["messages"]:
        message.pretty_print()


if __name__ == "__main__":
    main()
