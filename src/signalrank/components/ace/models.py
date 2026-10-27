from dataclasses import dataclass
from enum import StrEnum

from signalrank.components.retrieval.result import SearchResult


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    QUALIFIES = "qualifies"
    IRRELEVANT = "irrelevant"


@dataclass(frozen=True)
class StanceEvidence:
    result: SearchResult
    stance: EvidenceStance
