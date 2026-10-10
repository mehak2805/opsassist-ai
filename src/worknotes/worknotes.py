
import os
import streamlit as st
from ollama import Client


def get_ollama_client():
    try:
        api_key = st.secrets["OLLAMA_API_KEY"]
    except Exception:
        api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Add it to Streamlit Cloud Secrets."
        )

    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )


def generate_work_notes(ticket_text, category, resolution):

    prompt = f"""
You are OpsAssist AI, an IT Service Desk Work Notes Generator.

Generate concise ServiceNow work notes using ONLY information
explicitly provided in the ticket and resolution.

Ticket:
{ticket_text}

Category:
{category}

Resolution:
{resolution}

Return ONLY the following format:

Reason for call:
<short reason>

Resolution:
<actual resolution or current status>

Action items:
<Follow up Needed OR Follow up not needed>

Details:
- <fact explicitly stated in the ticket>
- <action explicitly confirmed as completed>
- <actual outcome or current status>

STRICT RULES:

1. Never invent an action performed by the user or IT agent.
2. Never claim an action was performed unless explicitly stated.
3. Recommendations are NOT completed actions.
4. Do not put recommendations under Details or Action items.
5. Action items must contain ONLY:
   "Follow up Needed"
   OR
   "Follow up not needed"
6. If no action was completed, do not invent one.
7. If the issue is unresolved, use "Follow up Needed".
8. If the issue is clearly resolved, use "Follow up not needed".
9. Do not invent URLs, access codes, policies, approvals,
   error messages, or technical details.
10. Do not add explanations outside the four required sections.
11. Do not treat recommended or suggested troubleshooting as
    a completed action.

Return ONLY the four required sections.
"""

    client = get_ollama_client()

    response = client.chat(
        model="gpt-oss:120b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    return response.message.content
