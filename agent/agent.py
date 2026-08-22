import os
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq

from agent.utils.prompt import CHAT_SYSTEM_PROMPT, WELCOME_MESSAGE
from agent.utils.state import StateSchema
from agent.utils.tools import TOOLS_CHAT


def create_agent_graph(checkpointer=None):

    llm = ChatGroq(
        temperature=0,
        model_name="qwen/qwen3.6-27b",
        api_key=os.getenv("GROQ_API_KEY"),
        max_retries=3,
        timeout=None
    )

    graph = StateGraph(state_schema=StateSchema)

    # --- NODES ---

    def welcome_node(state: StateSchema) -> dict:
        """Envia boas-vindas apenas na primeira mensagem."""
        if not state.get("messages"):
            return {
                "messages": [AIMessage(content=WELCOME_MESSAGE)],
                "confirmation": False
            }
        return {}

    def chat_node(state: StateSchema) -> dict:
        """Nó principal de chat com acesso às tools."""
        system_prompt = SystemMessage(content=CHAT_SYSTEM_PROMPT)
        response = llm.bind_tools(tools=TOOLS_CHAT).invoke(
            [system_prompt, *state["messages"]]
        )
        return {"messages": [response]}

    tool_node = ToolNode(tools=TOOLS_CHAT, name="tools_chat")

    # --- ROTEAMENTO ---

    def should_use_tools(state: StateSchema) -> str:
        """Decide se o LLM quer usar uma tool ou finalizar."""
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools_chat"
        return "end"

    # --- GRAFO ---

    graph.add_node("welcome_node", welcome_node)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools_chat", tool_node)

    graph.add_edge(START, "welcome_node")
    graph.add_edge("welcome_node", "chat_node")

    graph.add_conditional_edges(
        "chat_node",
        should_use_tools,
        {
            "tools_chat": "tools_chat",  # LLM quer usar tool
            "end": END                   # LLM respondeu direto
        }
    )

    graph.add_edge("tools_chat", "chat_node")  # após tool, volta pro chat

    return graph.compile(checkpointer=checkpointer)


graph = create_agent_graph()