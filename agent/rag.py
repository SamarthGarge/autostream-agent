"""
Handles loading the local JSON knowledge base and retrieving the most
relevant context for a given user query using keyword-based matching.
"""

import json
import os
from typing import Optional

# Path resolution
_KB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "knowledge_base", "autostream_kb.json"
)


def load_knowledge_base(path: Optional[str] = None) -> dict:
    """Load and return the knowledge base as a Python dict."""
    kb_path = path or _KB_PATH
    with open(kb_path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_context(query: str, kb: dict) -> str:
    """
    Return the most relevant KB sections as a formatted string.

    Strategy:
      - Keyword scan on the lowercased query
      - Match against: pricing/plans, policies, features, company info
      - If no keyword matches → return the full KB so the LLM can decide
    """
    q = query.lower()
    sections = []

    # Pricing / Plan keywords
    pricing_keywords = [
        "price", "pricing", "cost", "how much", "plan", "plans",
        "subscription", "basic", "pro", "monthly", "per month",
    ]
    if any(kw in q for kw in pricing_keywords):
        plans = kb.get("plans", {})
        sections.append("=== PRICING PLANS ===")
        for plan_key, info in plans.items():
            sections.append(f"\n{info['name']}:")
            sections.append(f"  Price            : {info['price']}")
            sections.append(f"  Videos / month   : {info['videos_per_month']}")
            sections.append(f"  Resolution       : {info['resolution']}")
            sections.append(f"  AI Captions      : {'Yes' if info['ai_captions'] else 'No'}")
            sections.append(f"  Support          : {info['support']}")

    # Feature keywords 
    feature_keywords = [
        "feature", "caption", "4k", "720p", "resolution", "template",
        "storage", "export", "edit", "ai", "cloud", "format",
    ]
    if any(kw in q for kw in feature_keywords):
        features = kb.get("features", {})
        sections.append("\n=== FEATURES ===")
        for feat_name, feat_desc in features.items():
            sections.append(f"  {feat_name.replace('_', ' ').title()}: {feat_desc}")

        # Also include plan-level feature differences
        plans = kb.get("plans", {})
        sections.append("\n=== PLAN COMPARISON ===")
        for plan_key, info in plans.items():
            sections.append(f"  {info['name']}: {info['resolution']} | "
                            f"{info['videos_per_month']} videos | "
                            f"AI Captions: {'Yes' if info['ai_captions'] else 'No'}")

    # Policy keywords
    policy_keywords = [
        "refund", "cancel", "policy", "support", "help", "trial",
        "return", "billing", "upgrade", "downgrade", "24/7",
    ]
    if any(kw in q for kw in policy_keywords):
        policies = kb.get("policies", {})
        sections.append("\n=== COMPANY POLICIES ===")
        for policy_key, policy_text in policies.items():
            sections.append(f"  {policy_key.replace('_', ' ').title()}: {policy_text}")

    # Company / general keywords
    company_keywords = ["autostream", "about", "what is", "company", "who are", "product"]
    if any(kw in q for kw in company_keywords):
        company = kb.get("company", {})
        sections.append("\n=== ABOUT AUTOSTREAM ===")
        sections.append(f"  {company.get('description', '')}")
        sections.append(f"  Website: {company.get('website', '')}")

    # Fallback: return everything
    if not sections:
        sections.append("=== FULL AUTOSTREAM KNOWLEDGE BASE ===")
        sections.append(json.dumps(kb, indent=2))

    return "\n".join(sections)