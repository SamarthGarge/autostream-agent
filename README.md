# 🎬 AutoStream AI Sales Agent

A **Conversational AI Agent** built for *AutoStream* — a fictional AI-powered video editing SaaS platform for content creators. Built as the ServiceHive / Inflx ML Intern assignment.

---

## Features

| Capability | Details |
|---|---|
| **Intent Detection** | Classifies every message into `greeting`, `inquiry`, or `high_intent` |
| **RAG Pipeline** | Answers questions strictly from a local JSON knowledge base |
| **Lead Capture** | Collects name → email → platform, then fires `mock_lead_capture()` |
| **State Memory** | Full conversation state retained across 5–6+ turns via LangGraph |
| **No Premature Tool Call** | Tool is only triggered once all three lead fields are confirmed |

---

## Project Structure

```
autostream-agent/
├── knowledge_base/
│   └── autostream_kb.json     # Pricing, features, policies (RAG source)
├── agent/
│   ├── __init__.py
│   ├── graph.py               # LangGraph StateGraph — core agent logic
│   ├── tools.py               # mock_lead_capture() function
│   ├── rag.py                 # KB loader + keyword-based retriever
│   └── intent.py              # LLM-powered intent classifier
├── main.py                    # CLI entry point
├── requirements.txt
├── .env.example               # Copy this to .env and add your API key
└── README.md
```

---

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

```bash
cp .env.example .env
```

Open `.env` and add your key:

```
GOOGLE_API_KEY=your_google_api_key_here
```

> **Get a FREE key** (no credit card required) at [https://aistudio.google.com](https://aistudio.google.com)

### 5. Run the Agent

```bash
python main.py
```

---

## Example Conversation

```
╔══════════════════════════════════════════════════════════════╗
║   🎬  AutoStream AI Sales Assistant                          ║
║       Powered by Gemini 2.5 Flash  ×  LangGraph              ║
╚══════════════════════════════════════════════════════════════╝

You: Hi there!
AutoStream: Hey! Welcome to AutoStream 🎬 — the smartest way to edit
            videos for your audience. Feel free to ask about our pricing
            or features anytime!

You: What are your pricing plans?
AutoStream: Great question! Here's what we offer:

  • Basic Plan — $29/month
      10 videos/month | 720p resolution | No AI captions
  • Pro Plan — $79/month
      Unlimited videos | 4K resolution | AI captions ✅

You: That sounds great, I want to try the Pro plan for my YouTube channel.
AutoStream: Awesome! I'd love to get you set up with AutoStream. 🎬
            First — what's your full name?

You: Jane Doe
AutoStream: Great to meet you, Jane! 😊
            What's the best email address for your account?

You: jane@example.com
AutoStream: Almost there! 🙌
            Which creator platform do you primarily publish on?
            (e.g., YouTube, Instagram, TikTok, Twitch, Facebook)

You: YouTube

════════════════════════════════════════════════════════
  ✅  Lead captured successfully!
  Name     : Jane Doe
  Email    : jane@example.com
  Platform : YouTube
════════════════════════════════════════════════════════

AutoStream: 🎉 You're all set, Jane!
            Our team will reach out to jane@example.com shortly.
            Welcome to AutoStream — let's take your YouTube content to
            the next level! 🚀

✅  Lead successfully captured in CRM!
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it provides **explicit, deterministic control flow** through a directed acyclic graph. AutoGen is better suited for multi-agent collaboration where agents negotiate, but this project requires a single focused agent that follows a strict sales funnel: greet → inform → qualify → capture. LangGraph's `StateGraph` makes those transitions clear, auditable, and easy to extend.

### How State is Managed

The agent uses a single typed `AgentState` dictionary that is passed through every node of the graph. It tracks: the full message history, the detected intent, the lead fields (`name`, `email`, `platform`), and two boolean control flags — `collecting_lead` and `lead_captured`.

The `classify` node is short-circuited when `collecting_lead=True`, so field answers like "Jane Doe" are never misclassified as intents. The `collect_lead` node uses `awaiting_field` to know exactly which field the user's latest reply is answering, then asks for the next missing field. The `capture_lead` node is only reachable once all three fields are non-null and `awaiting_field` is cleared — **guaranteeing the tool is never triggered prematurely**. Because the state dict is passed on every call, full memory is retained across all 5–6+ conversation turns without any external database.

---

## WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, use the **Twilio WhatsApp API** (or the Meta WhatsApp Business Cloud API) paired with a **FastAPI** (or Flask) webhook server.

### Step 1 — Create the Webhook Server

```python
# whatsapp_server.py
from fastapi import FastAPI, Form
from agent.graph import build_graph, AgentState

app = FastAPI()
graph = build_graph()

# In-memory session store (use Redis in production)
sessions: dict[str, AgentState] = {}

def _initial_state() -> AgentState:
    return {
        "messages": [], "intent": None,
        "name": None, "email": None, "platform": None,
        "lead_captured": False, "collecting_lead": False,
        "awaiting_field": None,
    }

@app.post("/whatsapp")
async def whatsapp_webhook(
    From: str = Form(...),   # e.g. "whatsapp:+91XXXXXXXXXX"
    Body: str = Form(...),   # user's message text
):
    # Load or initialise session for this sender
    state = sessions.get(From, _initial_state())
    state["messages"] = state["messages"] + [{"role": "user", "content": Body}]

    # Run the agent
    state = graph.invoke(state)
    sessions[From] = state

    # Extract reply
    assistant_msgs = [m for m in state["messages"] if m["role"] == "assistant"]
    reply = assistant_msgs[-1]["content"] if assistant_msgs else "Sorry, I couldn't process that."

    # Send reply via Twilio (uncomment and configure)
    # from twilio.rest import Client
    # client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    # client.messages.create(
    #     body=reply,
    #     from_="whatsapp:+14155238886",  # Twilio sandbox number
    #     to=From
    # )

    return {"reply": reply}
```

### Step 2 — Expose Your Server Locally (Development)

```bash
pip install fastapi uvicorn
uvicorn whatsapp_server:app --port 5000

# In a second terminal, expose via ngrok
ngrok http 5000
```

Copy the `https://xxxx.ngrok.io/whatsapp` URL into your **Twilio WhatsApp Sandbox** webhook settings.

### Step 3 — Production Deployment

| Concern | Solution |
|---|---|
| Hosting | Deploy on **Railway**, **Render**, or **AWS Lambda** |
| Session persistence | Replace `sessions = {}` with **Redis** |
| Lead storage | Replace `mock_lead_capture` with **HubSpot / Salesforce API** |
| Security | Validate Twilio's `X-Twilio-Signature` header on every request |
| Rate limiting | Add per-number rate limiting to prevent abuse |

---

## Tech Stack

- **Language:** Python 3.9+
- **Framework:** LangGraph (StateGraph)
- **LLM:** Gemini 2.5 Flash via `langchain-google-genai`
- **Knowledge Base:** Local JSON file with keyword-based RAG
- **Memory:** In-process state dictionary (persistent per session)
- **Entry Point:** CLI (`main.py`) / WhatsApp webhook (`whatsapp_server.py`)

---

## Evaluation Checklist

- [x] Intent detection (greeting / inquiry / high_intent)
- [x] RAG-powered answers from local knowledge base
- [x] State preserved across 5–6 conversation turns
- [x] Lead capture tool fires only after all 3 fields collected
- [x] Clean, modular code structure
- [x] README with setup, architecture, and WhatsApp deployment guide