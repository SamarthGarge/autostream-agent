"""
Contains the mock_lead_capture tool that is triggered when a high-intent
user has provided all three required lead fields: name, email, platform.
"""


def mock_lead_capture(name: str, email: str, platform: str) -> dict:
    """
    Mock API function to capture a qualified lead.

    In a real deployment, this would call a CRM API (e.g. HubSpot, Salesforce).
    For this assignment it prints to stdout and returns a success dict.

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Creator platform (YouTube, Instagram, TikTok, etc.)

    Returns:
        dict with status and captured lead data
    """
    print(f"\n{'=' * 52}")
    print(f"  ✅  Lead captured successfully!")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"{'=' * 52}\n")

    return {
        "status": "success",
        "message": f"Lead captured successfully: {name}, {email}, {platform}",
        "lead": {
            "name": name,
            "email": email,
            "platform": platform,
        },
    }