"""
Defines the LangGraph StateGraph that powers the AutoStream conversational
sales agent.
 
Graph topology
──────────────
  [START]
     │
  classify          ← determines intent; skipped when collecting lead
     │
  ┌──┴──────────────────────┐
  ▼                         ▼
handle_greeting          handle_rag        (inquiry / greeting paths)
  │                         │
 END                       END
 
  ▼ (high_intent or collecting_lead=True)
collect_lead              ← asks for name → email → platform one at a time
     │
  ┌──┴──────────────────┐
  ▼                     ▼
capture_lead            END  (waiting for user to answer the current field)
  │
 END
"""

import os
from typing import Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from .intent import classify_intent
from .rag import load_knowledge_base, retrieve_context
from .tools import mock_lead_capture

load_dotenv()

# Shared knowledge base (loaded once at import time)
_KB: Optional[dict] = None


def _get_kb() -> dict:
    global _KB
    if _KB is None:
        _KB = load_knowledge_base()
    return _KB


# LLM factory
def _get_llm() -> ChatGoogleGenerativeAI:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )



#  State definition

class AgentState(TypedDict):
    # Full conversation history as plain dicts
    messages: list  # [{"role": "user"|"assistant", "content": "..."}]

    # Detected intent for the current turn
    intent: Optional[str]       # "greeting" | "inquiry" | "high_intent"

    # Lead collection fields
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]

    # Workflow control flags
    lead_captured: bool         # True once mock_lead_capture has been called
    collecting_lead: bool       # True from first high-intent signal onwards
    awaiting_field: Optional[str]  # Which field we're currently waiting for



#  Node 1 – classify

def classify_node(state: AgentState) -> AgentState:
    """
    Use the LLM to classify the user's latest message.

    Skipped when we are mid-collection so we don't misclassify
    field answers (like "Jane Doe") as greetings or inquiries.
    """
    # Skip if already in lead-collection mode
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return state

    llm = _get_llm()
    last_msg = state["messages"][-1]["content"]
    intent = classify_intent(last_msg, llm)

    updated = {**state, "intent": intent}
    if intent == "high_intent":
        updated["collecting_lead"] = True

    return updated


# Router after classify

def _route_after_classify(state: AgentState) -> str:
    if state.get("collecting_lead") and not state.get("lead_captured"):
        return "collect_lead"
    intent = state.get("intent", "inquiry")
    if intent == "greeting":
        return "handle_greeting"
    if intent == "high_intent":
        return "collect_lead"
    return "handle_rag"



#  Node 2 – handle_greeting

def handle_greeting_node(state: AgentState) -> AgentState:
    """Respond warmly to casual greetings."""
    llm = _get_llm()
    last_msg = state["messages"][-1]["content"]

    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are a warm, upbeat sales assistant for AutoStream — an AI-powered "
                    "video editing SaaS platform for content creators. "
                    "Respond to the user's greeting with friendliness and enthusiasm. "
                    "Briefly introduce AutoStream in one sentence and invite the user to ask "
                    "about pricing or features. Keep your reply to 2–3 sentences maximum."
                )
            ),
            HumanMessage(content=last_msg),
        ]
    )

    new_messages = state["messages"] + [
        {"role": "assistant", "content": response.content}
    ]
    return {**state, "messages": new_messages}



#  Node 3 – handle_rag

def handle_rag_node(state: AgentState) -> AgentState:
    """
    Answer product / pricing / policy questions using RAG.

    The LLM is strictly instructed to use ONLY the retrieved KB context,
    preventing hallucination of prices or features.
    """
    llm = _get_llm()
    kb = _get_kb()
    last_msg = state["messages"][-1]["content"]
    context = retrieve_context(last_msg, kb)

    # Include up to the last 6 messages as conversation history
    history_msgs = state["messages"][:-1][-6:]
    history = "\n".join(
        f"{'User' if m['role'] == 'user' else 'AutoStream'}: {m['content']}"
        for m in history_msgs
    )

    system_prompt = f"""You are a knowledgeable and friendly sales assistant for AutoStream,
an AI-powered video editing SaaS for content creators.

STRICT RULES:
1. Answer ONLY using the KNOWLEDGE BASE provided below.
2. Do NOT invent, guess, or extrapolate any prices, features, or policies.
3. If the answer is not in the knowledge base, say:
   "I don't have that information right now — please reach out to our support team."
4. Format your response clearly. Use bullet points for plan comparisons.
5. If the user seems interested or satisfied, gently invite them to sign up.

--- KNOWLEDGE BASE ---
{context}

--- RECENT CONVERSATION ---
{history}
"""

    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=last_msg),
        ]
    )

    new_messages = state["messages"] + [
        {"role": "assistant", "content": response.content}
    ]
    return {**state, "messages": new_messages}



#  Node 4 – collect_lead

def collect_lead_node(state: AgentState) -> AgentState:
    """
    Sequentially collects name → email → platform from the user.

    Logic:
      1. If `awaiting_field` is set, the user's latest message IS the answer
        to that field → store it.
      2. Check which field is still missing → ask for it and set `awaiting_field`.
      3. If all three fields are present and `awaiting_field` is cleared →
        return state without appending a message so the router can proceed
        to `capture_lead`.
    """
    last_msg = state["messages"][-1]["content"].strip()
    awaiting = state.get("awaiting_field")

    # Step 1: Extract the answer to the field we were waiting for
    if awaiting == "name":
        state = {**state, "name": last_msg, "awaiting_field": None}
    elif awaiting == "email":
        state = {**state, "email": last_msg, "awaiting_field": None}
    elif awaiting == "platform":
        state = {**state, "platform": last_msg, "awaiting_field": None}

    # Step 2: Ask for the next missing field
    if not state.get("name"):
        reply = (
            "Awesome! I'd love to get you set up with AutoStream. 🎬\n\n"
            "To create your account, I just need a few quick details.\n\n"
            "First — what's your **full name**?"
        )
        state = {**state, "awaiting_field": "name"}

    elif not state.get("email"):
        reply = (
            f"Great to meet you, **{state['name']}**! 😊\n\n"
            f"What's the best **email address** for your account?"
        )
        state = {**state, "awaiting_field": "email"}

    elif not state.get("platform"):
        reply = (
            "Almost there! 🙌\n\n"
            "Which **creator platform** do you primarily publish on?\n"
            "*(e.g., YouTube, Instagram, TikTok, Twitch, Facebook)*"
        )
        state = {**state, "awaiting_field": "platform"}

    else:
        # All three fields collected — do NOT append a message here.
        # The router will send us to capture_lead which will confirm everything.
        return state

    new_messages = state["messages"] + [{"role": "assistant", "content": reply}]
    return {**state, "messages": new_messages}


# Router after collect_lead

def _route_after_collect(state: AgentState) -> str:
    """
    Proceed to capture_lead only when ALL fields are populated
    and we are no longer awaiting a field answer.
    """
    if (
        state.get("name")
        and state.get("email")
        and state.get("platform")
        and not state.get("awaiting_field")
    ):
        return "capture_lead"
    return END



#  Node 5 – capture_lead

def capture_lead_node(state: AgentState) -> AgentState:
    """
    Calls mock_lead_capture() with the collected lead info and
    sends a confirmation message to the user.

    This node is ONLY reached after all three fields are confirmed,
    guaranteeing the tool is never triggered prematurely.
    """
    # Call the mock API
    mock_lead_capture(
        name=state["name"],
        email=state["email"],
        platform=state["platform"],
    )

    # Compose the confirmation message
    reply = (
        f"🎉 You're all set, **{state['name']}**!\n\n"
        f"Here's a summary of your registration:\n"
        f"  • **Name:**     {state['name']}\n"
        f"  • **Email:**    {state['email']}\n"
        f"  • **Platform:** {state['platform']}\n\n"
        f"Our team will reach out to **{state['email']}** shortly with your "
        f"AutoStream account setup details.\n\n"
        f"Welcome aboard — let's take your {state['platform']} content to the next level! 🚀"
    )

    new_messages = state["messages"] + [{"role": "assistant", "content": reply}]
    return {
        **state,
        "messages": new_messages,
        "lead_captured": True,
        "collecting_lead": False,
    }



#  Graph builder

def build_graph():
    """
    Construct and compile the LangGraph StateGraph.

    Returns a compiled graph that accepts an AgentState dict
    and returns an updated AgentState dict.
    """
    workflow = StateGraph(AgentState)

    # Register all nodes
    workflow.add_node("classify",        classify_node)
    workflow.add_node("handle_greeting", handle_greeting_node)
    workflow.add_node("handle_rag",      handle_rag_node)
    workflow.add_node("collect_lead",    collect_lead_node)
    workflow.add_node("capture_lead",    capture_lead_node)

    # Entry point
    workflow.set_entry_point("classify")

    # Edges from classify
    workflow.add_conditional_edges(
        "classify",
        _route_after_classify,
        {
            "handle_greeting": "handle_greeting",
            "handle_rag":      "handle_rag",
            "collect_lead":    "collect_lead",
        },
    )

    # Terminal edges
    workflow.add_edge("handle_greeting", END)
    workflow.add_edge("handle_rag",      END)
    workflow.add_edge("capture_lead",    END)

    # Conditional edge from collect_lead
    workflow.add_conditional_edges(
        "collect_lead",
        _route_after_collect,
        {
            "capture_lead": "capture_lead",
            END:            END,
        },
    )

    return workflow.compile()