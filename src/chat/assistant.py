import pandas as pd
import ollama

from src.classifier.classifier import train_classifier, predict_ticket
from src.rag.rag_pipeline import get_context


def classify_ticket(ticket_text):

    df = pd.read_csv(
        "data/tickets.csv"
    )

    model, vectorizer = train_classifier(
        df
    )

    category = predict_ticket(
        model,
        vectorizer,
        ticket_text
    )

    return category


def generate_response(ticket_text):

    context, results = get_context(
        ticket_text,
        top_k=2
    )

    prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Your job is to provide accurate IT support using ONLY
the relevant information contained in the provided
Knowledge Base.

USER TICKET:
{ticket_text}

KNOWLEDGE BASE:
{context}

STRICT RULES:

1. Use only information relevant to the user's ticket.
2. Do not introduce unrelated troubleshooting steps.
3. Do not recommend password reset unless the ticket is
   specifically about password, authentication, or an
   account access problem caused by a password issue.
4. Do not invent procedures, URLs, access codes, policies,
   or technical details.
5. If the Knowledge Base does not contain enough information,
   clearly say that the issue requires further investigation
   or escalation.
6. Give concise and professional Service Desk guidance.
7. Clearly separate troubleshooting, recommended next action,
   and escalation.

Respond using this format:

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


def process_ticket(ticket_text):

    category = classify_ticket(
        ticket_text
    )

    response = generate_response(
        ticket_text
    )

    return {
        "ticket": ticket_text,
        "category": category,
        "response": response
    }