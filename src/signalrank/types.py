from typing import Literal

Route = Literal[
    "conversation",
    "retrieval",
]

RetrievalMode = Literal[
    "bm25",
    "dense",
    "hybrid",
]

ResponseMode = Literal[
    "auto",
    "evidence",
    "synthesis",
]

EffectiveResponseMode = Literal[
    "conversation",
    "evidence",
    "synthesis",
]

FallbackReason = Literal["llm_unavailable",]
