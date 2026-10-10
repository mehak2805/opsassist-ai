
from src.rag.rag_pipeline import get_context
from src.cloud_llm import generate_cloud_response


def chat_with_assistant(user_message, conversation_history=None):
    if conversation_history is None:
        conversation_history = []

    context, results = get_context(user_message, top_k=2)

    system_prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Knowledge Base:
{context}

Rules:
1. Use relevant knowledge-base information.
2. Never invent procedures, policies, URLs or access codes.
3. If information is insufficient, say further investigation
   or escalation is required.
4. Ask a clarifying question when the issue is unclear.
5. Keep responses concise and professional.
6. Use the supplied conversation history for context.
"""

    messages = [{"role": "system", "content": system_prompt}]

    # Keep only valid conversational messages.
    for message in conversation_history:
        if message.get("role") in ("user", "assistant"):
            messages.append({
                "role": message["role"],
                "content": message["content"],
            })

    return generate_cloud_response(
        user_message,
        conversation_history=messages,
    )
