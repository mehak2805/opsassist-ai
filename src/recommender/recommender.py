import ollama


def generate_recommendation(ticket_text, category, support_response):
    prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Analyze the following IT ticket and provide a practical resolution recommendation.

Ticket:
{ticket_text}

Ticket Category:
{category}

AI Support Response:
{support_response}

Provide the recommendation using exactly this format:

Recommended Resolution:
<most appropriate resolution>

Next Action:
<what the IT support agent should do next>

Escalation:
<Yes or No>

Escalation Reason:
<reason for escalation, or "Not required">

Rules:
1. Base the recommendation on the ticket and support response.
2. Do not invent technical procedures or information.
3. Do not recommend escalation unless there is a valid reason.
4. Keep the recommendation concise and practical.
5. Write it as guidance for an IT Service Desk agent.
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