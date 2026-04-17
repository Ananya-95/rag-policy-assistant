"""
Groq chat client for the generation step ("G" in RAG).

Requires ``GROQ_API_KEY`` in the environment (see ``config.settings``). Call after retrieval
to answer using a prompt that includes the retrieved policy excerpts.
"""
from langchain_groq import ChatGroq

from config.settings import settings


class GroqClient:
    """Thin wrapper around ``ChatGroq`` returning assistant text from a string prompt."""

    def __init__(self):
        self.client = ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
        )

    def invoke(self, prompt: str):
        """Send ``prompt`` to the configured model and return the message content string."""
        return self.client.invoke(prompt).content
