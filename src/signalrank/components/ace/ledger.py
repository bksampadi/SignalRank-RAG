from collections.abc import Iterable
from dataclasses import dataclass

from signalrank.components.ace.models import EvidenceStance, StanceEvidence


@dataclass(frozen=True)
class EvidenceLedger:
    evidence: tuple[StanceEvidence, ...]

    @classmethod
    def from_evidence(
        cls,
        evidence: Iterable[StanceEvidence],
    ) -> "EvidenceLedger":
        return cls(tuple(evidence))

    @property
    def evaluated_depth(self) -> int:
        return len(self.evidence)

    def count(self, stance: EvidenceStance) -> int:
        return sum(item.stance is stance for item in self.evidence)

    @property
    def supports(self) -> int:
        return self.count(EvidenceStance.SUPPORTS)

    @property
    def contradicts(self) -> int:
        return self.count(EvidenceStance.CONTRADICTS)

    @property
    def qualifies(self) -> int:
        return self.count(EvidenceStance.QUALIFIES)

    @property
    def irrelevant(self) -> int:
        return self.count(EvidenceStance.IRRELEVANT)

    @property
    def answer_changing_evidence_found(self) -> bool:
        return self.contradicts > 0 or self.qualifies > 0
