
import os
import requests
import streamlit as st


MODEL_NAME = "gpt-oss:120b"
OLLAMA_CHAT_URL = "https://ollama.com/api/chat"


def get_api_key():
    try:
        key = st.secrets.get("OLLAMA_API_KEY", "")
    except Exception:
        key = ""

    return key or os.getenv("OLLAMA_API_KEY", "")


def generate_cloud_response(prompt, conversation_history=None):
    api_key = get_api_key()

    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY missing. Add it in Streamlit Cloud Secrets."
        )

    messages = []

    if conversation_history:
        messages.extend(conversation_history)

    messages.append({"role": "user", "content": prompt})

    try:
        response = requests.post(
            OLLAMA_CHAT_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "stream": False,
            },
            timeout=180,
        )

        if not response.ok:
            raise RuntimeError(
                f"Ollama Cloud API returned HTTP {response.status_code}: "
                f"{response.text[:500]}"
            )

        result = response.json()
        return result["message"]["content"]

    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not connect to Ollama Cloud: {exc}"
        ) from exc
