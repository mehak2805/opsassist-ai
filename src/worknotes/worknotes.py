import ollama


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
2. Never say the user restarted, tested, verified, entered,
   reinstalled, contacted, or attempted anything unless it is
   explicitly stated in the Ticket or Resolution.
3. Recommendations are NOT completed actions.
4. Do not put recommendations under "Details".
5. Do not put recommendations under "Action items".
6. "Action items" must contain ONLY:
   "Follow up Needed"
   OR
   "Follow up not needed"
7. If no action was actually completed, do not create one.
8. If the issue is unresolved, use:
   Follow up Needed
9. If the issue is clearly resolved, use:
   Follow up not needed
10. Do not invent URLs, access codes, policies, approvals,
    error messages, or technical details.
11. Do not add headings, explanations, introductions, or
    commentary outside the required format.

IMPORTANT:
The Resolution may contain recommended troubleshooting.
Do NOT treat recommended, suggested, should, ask the user to,
or need to as completed actions.

Example of INVALID detail:
- User entered the correct access code.

Example of VALID detail:
- User reported that UKG could not be accessed on a new phone.

Return ONLY the four required sections.
"""

    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]