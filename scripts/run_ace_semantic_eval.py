from dataclasses import replace

from signalrank.components.ace.classifier import LLMStanceClassifier
from signalrank.components.ace.models import EvidenceStance
from signalrank.components.llm.factory import build_llm_service
from signalrank.components.retrieval.result import SearchResult
from signalrank.config.configuration import ConfigurationManager
from signalrank.evaluation.ace_eval import ACEEvalCase, evaluate_ace_case


def make_result(
    chunk_id: str,
    rank: int,
    text: str,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        doc_id="remote-work-policy",
        text=text,
        score=1.0,
        rank=rank,
        source_path="remote-work-policy.txt",
    )


def main() -> None:
    config = ConfigurationManager().load()

    if config.llm is None:
        raise RuntimeError("LLM configuration is required for ACE evaluation.")

    print(
        f"Provider: {config.llm.provider} | "
        f"Model: {config.llm.model_name}"
    )

    eval_llm_config = replace(
        config.llm,
        max_output_tokens=1024,
    )

    llm_service = build_llm_service(eval_llm_config)
    classifier = LLMStanceClassifier(llm_service)

    evidence = (
        make_result(
            "support",
            1,
            "Company policy explicitly permits employees to work remotely.",
        ),
        make_result(
            "contradict",
            2,
            (
                "Employees are required to work from the office. "
                "Remote work is not permitted."
            ),
        ),
        make_result(
            "qualify",
            3,
            (
                "Employees may work remotely only after successfully "
                "completing their probation period."
            ),
        ),
        make_result(
            "irrelevant-equipment",
            4,
            (
                "The company provides monitors and keyboards to employees "
                "who maintain a home office."
            ),
        ),
        make_result(
            "irrelevant-booking",
            5,
            (
                "Remote employees can use the company portal to reserve "
                "office desks when visiting headquarters."
            ),
        ),
        make_result(
            "irrelevant-holiday",
            6,
            (
                "Remote employees follow the same public-holiday calendar "
                "as office-based employees."
            ),
        ),
        make_result(
            "irrelevant-guidance",
            7,
            (
                "The company publishes ergonomic guidance for employees "
                "working from home."
            ),
        ),
        make_result(
            "indirect-contradiction",
            8,
            (
                "Employees assigned to regulated customer environments "
                "must perform all work from company premises."
            ),
        ),
        make_result(
            "regulated-laptop",
            9,
            (
                "Employees assigned to regulated customer environments "
                "use company-managed laptops."
            ),
        ),
    )

    expected = (
        EvidenceStance.SUPPORTS,
        EvidenceStance.CONTRADICTS,
        EvidenceStance.QUALIFIES,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.CONTRADICTS,
        EvidenceStance.IRRELEVANT,
    )

    case = ACEEvalCase(
        query="Can employees work remotely?",
        candidate_answer="Employees may work remotely.",
        evidence=evidence,
        expected_stances=expected,
    )

    result = evaluate_ace_case(
        classifier,
        case,
    )

    print()
    print(f"{'chunk_id':<24} {'expected':<14} {'predicted':<14}")
    print("-" * 54)

    for passage, expected_stance, predicted_stance in zip(
        evidence,
        result.expected,
        result.predicted,
        strict=True,
    ):
        print(
            f"{passage.chunk_id:<24} "
            f"{expected_stance.value:<14} "
            f"{predicted_stance.value:<14}"
        )

    print()
    print(f"Stance accuracy: {result.stance_accuracy:.3f}")
    print(f"False flags:     {result.false_flags}/{result.safe_cases}")
    print(f"False-flag rate: {result.false_flag_rate:.3f}")


if __name__ == "__main__":
    main()