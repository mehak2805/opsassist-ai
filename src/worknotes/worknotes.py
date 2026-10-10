
from src.cloud_llm import generate_cloud_response


def generate_work_notes(ticket_text, category, resolution):
    prompt = f"""
You are OpsAssist AI, an IT Service Desk Work Notes Generator.

Ticket:
{ticket_text}

Category:
{category}

Resolution:
{resolution}

Return ONLY this format:

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
1. Never invent actions performed by the user or agent.
2. Recommendations are not completed actions.
3. Do not put recommendations under Details or Action items.
4. Action items must contain only:
   Follow up Needed
   OR
   Follow up not needed
5. If unresolved, use Follow up Needed.
6. If clearly resolved, use Follow up not needed.
7. Never invent URLs, policies, approvals or technical details.
8. Return only the four required sections.
"""

    return generate_cloud_response(prompt)
