from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

from .config import get_settings


def get_llm():
    s = get_settings()
    if s.llm_provider == "mistral":
        if not s.mistral_api_key:
            raise RuntimeError("MISTRAL_API_KEY is not set")
        return ChatMistralAI(model=s.mistral_model, api_key=s.mistral_api_key, temperature=0)
    if not s.groq_api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return ChatGroq(model=s.groq_model, api_key=s.groq_api_key, temperature=0)
