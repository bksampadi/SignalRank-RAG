from collections.abc import Sequence
from typing import TypeVar

import logfire
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

from signalrank.components.llm.base import LLMProvider

StructuredOutput = TypeVar(
    "StructuredOutput",
    bound=BaseModel,
)


class LLMUnavailableError(RuntimeError):
    """Raised when all configured LLM providers fail."""


class LLMService:
    def __init__(
        self,
        providers: Sequence[LLMProvider],
    ) -> None:
        self._providers = tuple(providers)

    @property
    def available(self) -> bool:
        return bool(self._providers)

    def invoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> AIMessage:
        last_error: Exception | None = None

        for provider in self._providers:
            try:
                with logfire.span(
                    "LLM invocation",
                    provider=provider.name,
                ):
                    return provider.invoke(messages)

            except Exception as exc:  # noqa: BLE001
                last_error = exc

                logfire.warn(
                    "LLM profiver failed",
                    provider=provider.name,
                    error_type=type(exc).__name__,
                )

        if last_error is None:
            raise LLMUnavailableError("No LLM providers are configured.")

        raise LLMUnavailableError("All LLM providers failed.") from last_error

    def invoke_structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[StructuredOutput],
    ) -> StructuredOutput:
        last_error: Exception | None = None

        for provider in self._providers:
            try:
                with logfire.span(
                    "Structured LLM invocation",
                    provider=provider.name,
                    schema=schema.__name__,
                ):
                    return provider.invoke_structured(
                        messages,
                        schema,
                    )

            except Exception as exc:  # noqa: BLE001
                last_error = exc

                logfire.warn(
                    "Structured LLM provider failed",
                    provider=provider.name,
                    schema=schema.__name__,
                    error_type=type(exc).__name__,
                )

        if last_error is None:
            raise LLMUnavailableError("No LLM providers are configured.")

        raise LLMUnavailableError("All LLM providers failed.") from last_error
