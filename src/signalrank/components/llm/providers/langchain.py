from collections.abc import Sequence
from typing import TypeVar

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
    ) -> None:
        self._name = name
        self._model = model

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
        )

        response = structured_model.invoke(
            list(messages),
        )

        if isinstance(response, schema):
            return response

        return schema.model_validate(response)
