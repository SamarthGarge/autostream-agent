"""
Classifies user messages into one of three intent categories using the LLM.

  greeting   – casual hello / small talk
  inquiry    – product / pricing / feature / policy questions
  high_intent – user is ready to sign up, try, or purchase
"""

from langchain_core.messages import HumanMessage, SystemMessage

# Intent labels
VALID_INTENTS = {"greeting", "inquiry", "high_intent"}
DEFAULT_INTENT = "inquiry"

_SYSTEM_PROMPT = """You are an intent classifier for AutoStream, an AI-powered video editing SaaS platform.

Classify the user's message into EXACTLY ONE of these three intents:

  greeting    – The user is just saying hello, hi, how are you, or making casual small talk.
  inquiry     – The user is asking about pricing, features, plans, refund policy, support, or any product details.
  high_intent – The user clearly wants to sign up, try the product, buy a plan, or get started.
                Examples: "I want to subscribe", "Let's do it", "Sign me up", "I'd like to try the Pro plan",
                          "How do I get started?", "I'm ready to join"

Reply with ONLY a single word — one of: greeting, inquiry, high_intent
Do NOT include any punctuation, explanation, or extra text."""


def classify_intent(message: str, llm) -> str:
    """
    Classify *message* into one of: 'greeting', 'inquiry', 'high_intent'.

    Args:
        message : The latest user message text.
        llm     : An instantiated LangChain chat model.

    Returns:
        One of the three intent strings.
    """
    response = llm.invoke(
        [
            SystemMessage(content=_SYSTEM_PROMPT),
            HumanMessage(content=message),
        ]
    )
    intent = response.content.strip().lower().rstrip(".")

    # Defensive fallback if the model returns something unexpected
    if intent not in VALID_INTENTS:
        intent = DEFAULT_INTENT

    return intent