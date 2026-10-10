
import os
import streamlit as st
from ollama import Client

import pandas as pd

from src.classifier.classifier import train_classifier, predict_ticket
from src.rag.rag_pipeline import get_context


def get_ollama_client():
    """Connect to Ollama Cloud using Streamlit Secrets or an environment variable."""

    try:
        api_key = st.secrets["OLLAMA_API_KEY"]
    except Exception:
        api_key = os.getenv("OLLAMA_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. Configure it in Streamlit Cloud Secrets."
        )

    return Client(
        host="https://ollama.com",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=120,
    )


def classify_ticket(ticket_text):
    df = pd.read_csv("data/tickets.csv")
    model, vectorizer = train_classifier(df)
    category = predict_ticket(model, vectorizer, ticket_text)
    return category


def generate_response(ticket_text):
    context, results = get_context(ticket_text, top_k=2)

    prompt = f"""
You are OpsAssist AI, an IT Service Desk Copilot.

Provide accurate IT support using the relevant information
in the provided Knowledge Base.

USER TICKET:
{ticket_text}

KNOWLEDGE BASE:
{context}

STRICT RULES:
1. Use only information relevant to the ticket.
2. Do not introduce unrelated troubleshooting steps.
3. Do not recommend password reset unless the issue concerns
   passwords, authentication, or password-related account access.
4. Do not invent procedures, URLs, access codes, or policies.
5. If the Knowledge Base lacks sufficient information,
   clearly state that further investigation or escalation is needed.
6. Keep guidance concise and professional.
7. Separate troubleshooting, recommended action, and escalation.

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

    client = get_ollama_client()

    response = client.chat(
        model="gpt-oss:120b",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.message.content


def process_ticket(ticket_text):
    category = classify_ticket(ticket_text)
    response = generate_response(ticket_text)

    return {
        "ticket": ticket_text,
        "category": category,
        "response": response,
    }
