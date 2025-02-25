from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from huggingface_hub import login
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# Define the tools for the agent to use
@tool
def search(query: str):
    '''Define the tools for the agent to use'''
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."

# @tool
# def multiply(first_int: int, second_int: int) -> int:
#     """Multiply two integers together."""
#     return first_int * second_int


@tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

tools = [add, multiply]
tool_node = ToolNode(tools=tools)


llm_with_tools = ChatOllama(
    model = "llama3.1",
    temperature = 0,
    num_predict = 256,
).bind_tools([add, multiply])

def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


# Define the function that calls the model
def call_model(state: MessagesState):
    messages = state['messages']
    response = llm_with_tools.invoke(messages)
    
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


# define a new graph

workflow = StateGraph(MessagesState)

# Define the two nodes we will cycle between
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# Set the entrypoint as `agent`
# This means that this node is the first one called
workflow.add_edge(START, "agent")

# We now add a conditional edge
workflow.add_conditional_edges(
    # First, we define the start node. We use `agent`.
    # This means these are the edges taken after the `agent` node is called.
    "agent",
    # Next, we pass in the function that will determine which node is called next.
    should_continue,
)

workflow.add_edge("tools", "agent")

# Initialize memory to persist state between graph runs
checkpointer = MemorySaver()

# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable.
# Note that we're (optionally) passing the memory when compiling the graph
app = workflow.compile(checkpointer=checkpointer)

final_state = app.invoke(
     {"messages": [HumanMessage(content="4 + 23")]},
    config={"configurable": {"thread_id": 42}}
)

print(final_state)