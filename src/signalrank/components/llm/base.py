from collections.abc import Sequence
from typing import Protocol, TypeVar

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel

StructuredOutput = TypeVar(
    "StructuredOutput",
    bound=BaseModel,
)


class LLMProvider(Protocol):
    @property
    def name(self) -> str: ...

    def invoke(
        self,
        messages: Sequence[BaseMessage],
    ) -> AIMessage: ...

    def invoke_structured(
        self,
        messages: Sequence[BaseMessage],
        schema: type[StructuredOutput],
    ) -> StructuredOutput: ...
