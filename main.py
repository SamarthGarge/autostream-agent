"""
CLI entry point for the AutoStream conversational sales agent.

Run with:
    python main.py
"""

import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()

# Validate API key before importing the graph
if not os.getenv("GOOGLE_API_KEY"):
    print("\n❌  ERROR: GOOGLE_API_KEY not found in environment.")
    print("    1. Copy .env.example  →  .env")
    print("    2. Add your key:  GOOGLE_API_KEY=your_key_here")
    print("    3. Get a free key at: https://aistudio.google.com\n")
    sys.exit(1)

from agent.graph import AgentState, build_graph

# Banner
BANNER = """
╔══════════════════════════════════════════════════════════════╗
║   🎬  AutoStream AI Sales Assistant                          ║
║       Powered by Gemini 2.5 Flash  ×  LangGraph              ║
╚══════════════════════════════════════════════════════════════╝
  Ask about pricing, features, or type "I want to sign up"!
  Type  quit / exit  to end the conversation.
"""

DIVIDER = "─" * 64


# Initial state factory
def _initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": None,
        "name": None,
        "email": None,
        "platform": None,
        "lead_captured": False,
        "collecting_lead": False,
        "awaiting_field": None,
    }


# Main loop 
def main() -> None:
    print(BANNER)

    graph = build_graph()
    state = _initial_state()

    while True:
        # Read user input
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "bye", "q"):
            print("\nThanks for chatting with AutoStream! Goodbye! 👋\n")
            break

        # Append user message to state
        state["messages"] = state["messages"] + [
            {"role": "user", "content": user_input}
        ]

        # Run the graph
        try:
            state = graph.invoke(state)
        except EnvironmentError as e:
            print(f"\n⚠️  Configuration error: {e}\n")
            break
        except Exception as e:
            print(f"\n⚠️  Unexpected error: {e}\n")
            continue

        # Print the latest assistant reply
        assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
        if assistant_msgs:
            reply = assistant_msgs[-1]['content']
            # Strip markdown formatting for clean terminal output
            reply = re.sub(r'^\s*\*(\s)', r'•\1', reply, flags=re.MULTILINE)  # bullet points
            reply = reply.replace('*', '')  # remove all remaining bold/italic markers
            # Add blank line before plan section headings (e.g. "•   Pro Plan:")
            reply = re.sub(r'(?<!\n\n)(^.*Plan:\s*$)', r'\n\1', reply, flags=re.MULTILINE)
            # Add space between a URL and trailing punctuation (e.g. "autostream.io!")
            reply = re.sub(r'(https?://\S+?)([!?.,:;])', r'\1 \2', reply)
            print(f"\nAutoStream: {reply}\n")

        # End gracefully after successful lead capture
        if state.get("lead_captured"):
            print(DIVIDER)
            print("✅  Lead successfully captured in CRM!")
            print("    The AutoStream team will be in touch soon.")
            print(DIVIDER + "\n")
            break


if __name__ == "__main__":
    main()