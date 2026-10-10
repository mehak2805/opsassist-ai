
from src.cloud_llm import generate_cloud_response


def generate_recommendation(ticket_text, category, support_response):
    prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Ticket:
{ticket_text}

Ticket Category:
{category}

AI Support Response:
{support_response}

Return exactly this format:

Recommended Resolution:
<most appropriate resolution>

Next Action:
<what the IT support agent should do next>

Escalation:
<Yes or No>

Escalation Reason:
<reason, or Not required>

Rules:
1. Base recommendations on the ticket and support response.
2. Do not invent technical procedures or facts.
3. Recommend escalation only when justified.
4. Keep the response concise and practical.
5. Write for an IT Service Desk agent.
"""

    return generate_cloud_response(prompt)
