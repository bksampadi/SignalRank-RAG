from signalrank.components.llm.base import LLMProvider
from signalrank.components.llm.providers.groq import (
    build_groq_provider,
)
from signalrank.components.llm.providers.openrouter import (
    build_openrouter_provider,
)
from signalrank.config.settings import LLMConfig
from signalrank.services.llm_service import LLMService


def build_llm_service(
    config: LLMConfig,
) -> LLMService:
    providers: list[LLMProvider] = []

    if config.provider == "openrouter":
        provider = build_openrouter_provider(
            model_name=config.model_name,
            fallback_models=config.fallback_models,
            max_retries=config.max_retries,
            max_output_tokens=config.max_output_tokens,
        )

        providers.append(provider)

    elif config.provider == "groq":
        provider = build_groq_provider(
            model_name=config.model_name,
            max_retries=config.max_retries,
            max_output_tokens=config.max_output_tokens,
        )

        providers.append(provider)

    return LLMService(
        providers=providers,
    )
