
import pandas as pd

from src.classifier.classifier import train_classifier, predict_ticket
from src.rag.rag_pipeline import get_context
from src.cloud_llm import generate_cloud_response


def classify_ticket(ticket_text):
    df = pd.read_csv("data/tickets.csv")
    model, vectorizer = train_classifier(df)
    return predict_ticket(model, vectorizer, ticket_text)


def generate_response(ticket_text):
    context, results = get_context(ticket_text, top_k=2)

    prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

USER TICKET:
{ticket_text}

KNOWLEDGE BASE:
{context}

Rules:
1. Use relevant knowledge-base information only.
2. Never invent procedures, URLs, access codes or policies.
3. Do not recommend password reset for unrelated issues.
4. If the knowledge base is insufficient, recommend investigation
   or escalation.
5. Keep guidance concise and professional.

Respond using this exact format:

Issue:
<brief description>

Troubleshooting Steps:
1. <step>
2. <step>
3. <step>

Recommended Action:
<next action>

Escalation:
<Yes/No and reason>
"""

    return generate_cloud_response(prompt)


def process_ticket(ticket_text):
    category = classify_ticket(ticket_text)
    response = generate_response(ticket_text)

    return {
        "ticket": ticket_text,
        "category": category,
        "response": response,
    }
