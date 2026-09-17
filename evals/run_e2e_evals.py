"""CLI entry point for the real end-to-end evaluation benchmark.

    python -m evals.run_e2e_evals

WARNING: this makes REAL OpenAI API calls (classification, address
extraction where needed, and response generation) for every case in
`evals.e2e_cases.E2E_CASES` and consumes API usage. It is never run by
pytest and must be run manually.

Everything else in the workflow (policy, routing, order resolution, action
execution, approval enforcement) remains fully deterministic - only the
three OpenAI-backed components can vary between runs. See the "Reproducing
this benchmark" note printed at the end of a run.
"""

from __future__ import annotations

import sys

from evals.e2e_cases import E2E_CASES
from evals.e2e_runner import EndToEndEvalResult, EndToEndEvalSummary, run_e2e_evals

_METRIC_LABELS = {
    "intent": "Intent accuracy",
    "urgency": "Urgency accuracy",
    "order_resolution": "Order resolution accuracy",
    "policy": "Policy accuracy",
    "route": "Routing accuracy",
    "proposed_action": "Action proposal accuracy",
    "approval_behavior": "Approval behavior accuracy",
    "mutation": "Mutation behavior accuracy",
    "final_state": "Final-state accuracy",
    "response": "Response fact-check pass rate",
}


def _format_rate(rate: float | None) -> str:
    if rate is None:
        return "n/a (no applicable cases)"
    return f"{rate * 100:.1f}%"


def _print_case_result(result: EndToEndEvalResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.case_id}")
    print(f"  question: {result.description}")
    if result.infra_error:
        print(f"  INFRASTRUCTURE ERROR: {result.infra_error}")
        return
    print(f"  intent: actual={result.actual_intent!r}")
    print(f"  urgency: actual={result.actual_urgency!r}")
    print(f"  route: actual={result.actual_route!r}")
    print(f"  final_status: {result.actual_final_status!r}")
    if result.actual_final_response:
        preview = result.actual_final_response
        if len(preview) > 160:
            preview = preview[:157] + "..."
        print(f"  final_response: {preview!r}")
    for component in result.component_results:
        marker = "ok" if component.passed else "FAIL"
        applicability = "" if component.applicable else " (n/a for this metric)"
        line = f"  - {component.name}: {marker}{applicability}"
        if not component.passed and component.detail:
            line += f" -- {component.detail}"
        print(line)
    print()


def _print_summary(summary: EndToEndEvalSummary) -> None:
    print("End-to-End Evaluation")
    print("---------------------")
    print(f"Cases: {summary.total_cases}")
    print()
    for metric in summary.metric_accuracies:
        label = _METRIC_LABELS.get(metric.name, metric.name)
        print(f"{label}: {_format_rate(metric.rate)} ({metric.passed_count}/{metric.applicable_count})")
    print(
        f"Overall case pass rate: {_format_rate(summary.overall_pass_rate)} "
        f"({summary.overall_pass_count}/{summary.total_cases})"
    )
    print()
    print(
        "These metrics apply only to this curated benchmark and are not claims of "
        "general model accuracy or production reliability."
    )
    print(
        "Response checks are deterministic substring/fact checks (never an "
        "LLM-as-a-judge); they are transparent but cannot detect every possible "
        "hallucination or semantic error."
    )
    print(
        "This benchmark makes real OpenAI calls: results can vary across runs "
        "because the model-backed classifier, address extractor, and response "
        "generator are not deterministic. Every other workflow component "
        "(policy, routing, order resolution, action execution, approval "
        "enforcement) is fully deterministic and does not vary."
    )


def main() -> int:
    print("Running the real end-to-end benchmark - this makes real OpenAI API calls.")
    print(f"{len(E2E_CASES)} cases, one interrupt/resume cycle each for approval-required cases.")
    print()

    summary = run_e2e_evals(E2E_CASES)

    for result in summary.results:
        _print_case_result(result)

    _print_summary(summary)

    return 0 if summary.overall_pass_count == summary.total_cases else 1


if __name__ == "__main__":
    sys.exit(main())
