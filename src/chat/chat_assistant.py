import ollama

from src.rag.rag_pipeline import get_context


def chat_with_assistant(user_message, conversation_history=None):

    if conversation_history is None:
        conversation_history = []

    context, results = get_context(
        user_message,
        top_k=2
    )

    system_prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Your role is to help IT support agents and users troubleshoot
common IT issues.

Use the provided Knowledge Base to give accurate answers.

KNOWLEDGE BASE:
{context}

RULES:
1. Use only relevant information from the Knowledge Base.
2. Do not invent procedures, policies, URLs, access codes, or technical details.
3. Do not recommend unrelated troubleshooting steps.
4. If the Knowledge Base does not contain enough information, clearly say
   that further investigation or escalation is required.
5. Ask a clarifying question when the user's issue is unclear.
6. Keep responses concise and professional.
7. Remember the previous conversation when answering follow-up questions.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    messages.extend(conversation_history)

    messages.append(
        {
            "role": "user",
            "content": user_message
        }
    )

    response = ollama.chat(
        model="llama3.2",
        messages=messages
    )

    return response["message"]["content"]