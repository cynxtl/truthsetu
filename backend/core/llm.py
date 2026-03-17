from langchain_core.language_models import BaseLLM
from loguru import logger
from backend.core.config import get_settings

settings = get_settings()


def get_llm() -> BaseLLM:
    provider = settings.llm_provider.lower()
    logger.info(f"Loading LLM: {provider}")

    if provider == "groq":
        return _get_groq()
    elif provider == "ollama":
        return _get_ollama()
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {provider}")


def _get_groq():
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            api_key=settings.groq_api_key,
            model=settings.groq_model,
            temperature=0.1,
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-groq")


def _get_ollama():
    try:
        from langchain_ollama import OllamaLLM
        return OllamaLLM(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.1,
        )
    except ImportError:
        raise ImportError("Run: pip install langchain-ollama")