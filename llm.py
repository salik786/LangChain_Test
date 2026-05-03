"""
underwriter_agent/llm.py
-------------------------
Lazy LLM construction — picks the first available provider from env.
"""

from __future__ import annotations

import os
import logging

from langchain_core.language_models import BaseChatModel

logger = logging.getLogger(__name__)

_llm_instance: BaseChatModel | None = None


def get_llm() -> BaseChatModel:
    """Return a shared LLM instance (constructed once)."""
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    if os.getenv("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        _llm_instance = ChatOpenAI(model="gpt-4o", temperature=0)
        logger.info("Using OpenAI GPT-4o")

    elif os.getenv("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        _llm_instance = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
        logger.info("Using Anthropic Claude 3.5 Sonnet")

    elif os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI
        _llm_instance = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            temperature=0,
        )
        logger.info("Using Azure OpenAI")

    else:
        raise EnvironmentError(
            "No LLM provider key found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY, "
            "or AZURE_OPENAI_API_KEY in your .env file."
        )

    return _llm_instance
