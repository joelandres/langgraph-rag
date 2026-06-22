from typing import Literal

from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage
from langgraph.graph import END, MessagesState

from src.prompts import GENERATE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
from src.schemas import GradeDocuments
from src.tools import retrieve_blog_posts

response_model = init_chat_model("openai:gpt-4o-mini", temperature=0)
grader_model = init_chat_model("openai:gpt-4o-mini", temperature=0)
retriever_tool = retrieve_blog_posts


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state.
    Given the question, it will decide to retrieve using the retriever tool,
    or simply respond to the user.
    """
    print("GENERATE QUERY OR RESPOND")
    response = response_model.bind_tools([retriever_tool]).invoke(state["messages"])
    response.name = "generate_query_or_respond"
    return {"messages": [response]}


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    print("GRADE DOCUMENTS")
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = grader_model.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )
    print(response.binary_score)
    if response.binary_score == "yes":
        print("GENERATE_ANSWER")
        return "generate_answer"
    print("REWRITE QUESTION")
    return "rewrite_question"


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    print("REWRITE QUESTION")
    question = state["messages"][0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    response.name = "rewrite_question"
    return {"messages": [HumanMessage(content=response.content)]}


def generate_answer(state: MessagesState):
    """Generate an answer from question and retrieved context."""
    print("GENERATE ANSWER")
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    response.name = "generate_answer"
    return {"messages": [response]}


# Route based on whether the model requested tool calls.
def route_on_tool_calls(state: MessagesState):
    print("ROUTE ON TOOL CALLS")
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        print("TOOLS")
        return "tools"
    print("END")
    return END
