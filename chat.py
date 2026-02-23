"""Groq-powered chat for market analysis Q&A."""
import os
import streamlit as st

SYSTEM_PROMPT = """You are a precious metals trading analyst assistant for a £2M gold/silver portfolio.
The user is deploying capital over 2026 with a weeks-to-months horizon, driven by macro catalysts
(Fed rate cuts, tariffs, Iran tensions, Asia demand).

You have access to the full current market state below. Answer questions concisely and actionably.
Focus on what the data means for trading decisions. Reference specific indicators and scores.
Use plain English, no jargon without explanation. Keep answers to 2-4 paragraphs max.

CURRENT MARKET STATE:
{context}
"""


def _get_client():
    """Get Groq client, return None if no API key."""
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return None
    try:
        from groq import Groq
        return Groq(api_key=api_key)
    except Exception:
        return None


def is_chat_available() -> bool:
    return _get_client() is not None


def ask(question: str, context: str, history: list[dict] = None) -> str:
    """Send a question to Groq with full market context. Returns answer string."""
    client = _get_client()
    if not client:
        return "Chat unavailable -- add GROQ_API_KEY to Streamlit secrets or environment."

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]
    if history:
        for msg in history[-6:]:
            messages.append(msg)
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"
