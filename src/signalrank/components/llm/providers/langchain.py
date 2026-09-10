from collections.abc import Sequence
from typing import Any, TypeVar

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

StructuredOutput = TypeVar(
    "StructuredOutput",
    bound=BaseModel,
)


class LangChainLLMProvider:
    def __init__(
        self,
        *,
        name: str,
        model: BaseChatModel,
        structured_output_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._name = name
        self._model = model
        self._structured_output_kwargs = structured_output_kwargs or {}

    @property
    def name(self) -> str:
        return self._name

    def invoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> AIMessage:
        response = self._model.invoke(
            list(messages),
        )

        if not isinstance(response, AIMessage):
            raise TypeError(f"Expected AIMessage, got {type(response).__name__}")

        return response

    def invoke_structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[StructuredOutput],
    ) -> StructuredOutput:
        structured_model = self._model.with_structured_output(
            schema,
            **self._structured_output_kwargs,
        )

        response = structured_model.invoke(
            list(messages),
        )

        if isinstance(response, schema):
            return response

        return schema.model_validate(response)
