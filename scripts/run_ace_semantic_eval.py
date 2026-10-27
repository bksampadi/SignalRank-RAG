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

    print(f"Provider: {config.llm.provider} | Model: {config.llm.model_name}")

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
            "role-exception",
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
        EvidenceStance.QUALIFIES,
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

    hard_evidence = (
        make_result(
            "negation-support",
            1,
            "Company policy states that employees are not prohibited from working remotely.",
        ),
        make_result(
            "irrelevant-leave",
            2,
            "Working remotely does not change an employee's annual leave entitlement.",
        ),
        make_result(
            "irrelevant-lunch",
            3,
            "The company does not reimburse lunch expenses incurred while working remotely.",
        ),
        make_result(
            "irrelevant-training",
            4,
            (
                "Employees who work remotely complete the same annual security "
                "training as office-based employees."
            ),
        ),
        make_result(
            "irrelevant-logs",
            5,
            "Remote-access logs are retained for 90 days for security auditing.",
        ),
        make_result(
            "irrelevant-equipment-rule",
            6,
            (
                "Company monitors may not be taken outside the country, "
                "including when employees are working remotely."
            ),
        ),
        make_result(
            "qualify-role",
            7,
            (
                "Employees may work remotely only if their role has been "
                "designated remote-eligible."
            ),
        ),
        make_result(
            "qualify-location",
            8,
            "Remote work is permitted only from locations within the Netherlands.",
        ),
    )

    hard_expected = (
        EvidenceStance.SUPPORTS,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.IRRELEVANT,
        EvidenceStance.QUALIFIES,
        EvidenceStance.QUALIFIES,
    )

    hard_case = ACEEvalCase(
        query="Can employees work remotely?",
        candidate_answer="Employees may work remotely.",
        evidence=hard_evidence,
        expected_stances=hard_expected,
    )

    hard_result = evaluate_ace_case(
        classifier,
        hard_case,
    )

    print()
    print("Batch 1")
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

    print()
    print("Batch 2 — hard negatives")
    print(f"{'chunk_id':<24} {'expected':<14} {'predicted':<14}")
    print("-" * 54)

    for passage, expected_stance, predicted_stance in zip(
        hard_evidence,
        hard_result.expected,
        hard_result.predicted,
        strict=True,
    ):
        print(
            f"{passage.chunk_id:<24} "
            f"{expected_stance.value:<14} "
            f"{predicted_stance.value:<14}"
        )

    print()
    print(f"Stance accuracy: {hard_result.stance_accuracy:.3f}")
    print(f"False flags:     {hard_result.false_flags}/{hard_result.safe_cases}")
    print(f"False-flag rate: {hard_result.false_flag_rate:.3f}")


if __name__ == "__main__":
    main()
