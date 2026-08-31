from langchain_openrouter import ChatOpenRouter

from signalrank.components.llm.providers.langchain import (
    LangChainLLMProvider,
)


def build_openrouter_provider(
    *,
    model_name: str,
    fallback_models: tuple[str, ...],
    max_retries: int,
    max_output_tokens: int,
) -> LangChainLLMProvider:
    model_kwargs: dict[str, object] = {}

    if fallback_models:
        model_kwargs["models"] = [
            model_name,
            *fallback_models,
        ]

    model = ChatOpenRouter(
        model=model_name,
        temperature=0,
        max_retries=max_retries,
        max_tokens=max_output_tokens,
        route="fallback",
        model_kwargs=model_kwargs,
        openrouter_provider={
            "allow_fallbacks": True,
        },
    )

    return LangChainLLMProvider(
        name="openrouter",
        model=model,
    )
