from langchain_groq import ChatGroq

from signalrank.components.llm.providers.langchain import (
    LangChainLLMProvider,
)


def build_groq_provider(
    *,
    model_name: str,
    max_retries: int,
    max_output_tokens: int,
) -> LangChainLLMProvider:
    model = ChatGroq(
        model=model_name,
        temperature=0,
        max_retries=max_retries,
        max_tokens=max_output_tokens,
    )

    return LangChainLLMProvider(
        name="groq",
        model=model,
    )
