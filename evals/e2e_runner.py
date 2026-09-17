"""Execution and scoring for the end-to-end evaluation benchmark.

`run_single_case` builds a completely FRESH graph, in-memory action store,
and `InMemorySaver` checkpointer for every case - no mutation from one case
can ever be observed by another, and case order never affects results (see
`tests/test_e2e_evals.py::*isolation*`). Approval-required cases genuinely
exercise `interrupt()`/`Command(resume=...)` on a unique `eval-<case_id>`
thread_id - the HITL flow is never bypassed.

Technical/infrastructure failures (a benchmark-initialization bug, a
malformed case) are distinguished from ordinary case-evaluation failures (a
model/workflow mismatch against expectations): the latter are captured per
case so one bad case never aborts the rest of the benchmark; the former are
allowed to raise and fail fast, since silently continuing past a broken
setup would make every subsequent result meaningless.
"""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel

from customer_ops.action_inputs import ActionInputExtractor, OpenAIActionInputExtractor
from customer_ops.classifier import OpenAIRequestClassifier, RequestClassifier
from customer_ops.graph import build_customer_ops_graph
from customer_ops.response_generator import CustomerResponseGenerator, OpenAICustomerResponseGenerator
from evals.e2e_cases import EndToEndEvalCase
from evals.e2e_checks import (
    ComponentCheckResult,
    check_approval_behavior,
    check_final_state,
    check_intent,
    check_mutation,
    check_order_resolution,
    check_policy,
    check_proposed_action,
    check_response,
    check_route,
    check_urgency,
)
from tools.action_store import InMemoryCustomerActionStore


class EndToEndEvalResult(BaseModel):
    """The outcome of running one benchmark case - designed to be easy to diagnose.

    Never includes secrets or raw OpenAI response objects - only the
    already-structured workflow facts and this benchmark's own
    pass/fail judgments.
    """

    case_id: str
    description: str
    passed: bool
    component_results: list[ComponentCheckResult]
    failure_messages: list[str]
    actual_intent: str | None = None
    actual_urgency: str | None = None
    actual_route: str | None = None
    actual_final_status: str | None = None
    actual_final_response: str | None = None
    infra_error: str | None = None


class MetricAccuracy(BaseModel):
    """Applicability-aware accuracy for one named evaluation dimension."""

    name: str
    applicable_count: int
    passed_count: int

    @property
    def rate(self) -> float | None:
        """`None` when no case in this benchmark exercised this dimension."""
        if self.applicable_count == 0:
            return None
        return self.passed_count / self.applicable_count


class EndToEndEvalSummary(BaseModel):
    """Aggregate results across the whole benchmark run."""

    total_cases: int
    overall_pass_count: int
    metric_accuracies: list[MetricAccuracy]
    results: list[EndToEndEvalResult]

    @property
    def overall_pass_rate(self) -> float | None:
        if self.total_cases == 0:
            return None
        return self.overall_pass_count / self.total_cases


def run_single_case(
    case: EndToEndEvalCase,
    *,
    classifier: RequestClassifier,
    action_input_extractor: ActionInputExtractor,
    response_generator: CustomerResponseGenerator,
) -> EndToEndEvalResult:
    """Run exactly one case against a fresh graph/store/checkpointer.

    A technical exception during graph invocation (e.g. an unexpected
    domain error) is captured as a FAILED case with the exception type and
    a concise message, rather than aborting the whole benchmark.
    """
    thread_id = f"eval-{case.case_id}"
    store = InMemoryCustomerActionStore.from_json()

    try:
        orders_before = {
            order.order_id: order.model_dump(mode="json")
            for order in store.list_orders_for_customer(case.customer_id)
        }
    except Exception as exc:  # noqa: BLE001 - recorded as an infra error, not swallowed
        return EndToEndEvalResult(
            case_id=case.case_id,
            description=case.description,
            passed=False,
            component_results=[],
            failure_messages=[f"Could not read baseline orders for {case.customer_id!r}."],
            infra_error=f"{type(exc).__name__}: {exc}",
        )

    graph = build_customer_ops_graph(
        classifier=classifier,
        store=store,
        action_input_extractor=action_input_extractor,
        action_store=store,
        checkpointer=InMemorySaver(),
        response_generator=response_generator,
    )
    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = graph.invoke(
            {
                "request_id": f"req-{case.case_id}",
                "customer_id": case.customer_id,
                "customer_message": case.customer_message,
            },
            config=config,
        )
    except Exception as exc:  # noqa: BLE001 - a case-level failure, not an infra failure
        return EndToEndEvalResult(
            case_id=case.case_id,
            description=case.description,
            passed=False,
            component_results=[],
            failure_messages=[f"Unhandled exception during initial invoke: {type(exc).__name__}: {exc}"],
        )

    interrupted = "__interrupt__" in result
    interrupt_payload: Any = None
    pre_approval_result: dict[str, Any] | None = None

    if interrupted:
        pre_approval_result = result
        interrupt_payload = result["__interrupt__"][0].value
        if case.human_decision is not None:
            try:
                result = graph.invoke(Command(resume={"decision": case.human_decision}), config=config)
            except Exception as exc:  # noqa: BLE001
                return EndToEndEvalResult(
                    case_id=case.case_id,
                    description=case.description,
                    passed=False,
                    component_results=[],
                    failure_messages=[f"Unhandled exception during resume: {type(exc).__name__}: {exc}"],
                    actual_intent=pre_approval_result.get("intent"),
                    actual_urgency=pre_approval_result.get("urgency"),
                    actual_route=pre_approval_result.get("route"),
                )

    component_results = [
        check_intent(case, result),
        check_urgency(case, result),
        check_order_resolution(case, result),
        check_policy(case, result),
        check_route(case, result),
        check_proposed_action(case, result),
        check_approval_behavior(
            case,
            interrupted=interrupted,
            interrupt_payload=interrupt_payload,
            pre_approval_result=pre_approval_result,
            post_result=result,
        ),
        check_mutation(case, orders_before, result.get("order_context"), result.get("action_result")),
        check_final_state(case, result),
        check_response(case, result.get("final_response")),
    ]

    failure_messages = [
        f"{component.name}: {component.detail}" for component in component_results if not component.passed
    ]

    return EndToEndEvalResult(
        case_id=case.case_id,
        description=case.description,
        passed=not failure_messages,
        component_results=component_results,
        failure_messages=failure_messages,
        actual_intent=result.get("intent"),
        actual_urgency=result.get("urgency"),
        actual_route=result.get("route"),
        actual_final_status=result.get("workflow_status"),
        actual_final_response=result.get("final_response"),
    )


def _summarize(results: list[EndToEndEvalResult]) -> EndToEndEvalSummary:
    metric_names = [
        "intent",
        "urgency",
        "order_resolution",
        "policy",
        "route",
        "proposed_action",
        "approval_behavior",
        "mutation",
        "final_state",
        "response",
    ]
    accuracies: list[MetricAccuracy] = []
    for name in metric_names:
        applicable_count = 0
        passed_count = 0
        for result in results:
            for component in result.component_results:
                if component.name == name and component.applicable:
                    applicable_count += 1
                    if component.passed:
                        passed_count += 1
        accuracies.append(MetricAccuracy(name=name, applicable_count=applicable_count, passed_count=passed_count))

    return EndToEndEvalSummary(
        total_cases=len(results),
        overall_pass_count=sum(1 for result in results if result.passed),
        metric_accuracies=accuracies,
        results=results,
    )


def run_e2e_evals(
    cases: list[EndToEndEvalCase] | tuple[EndToEndEvalCase, ...],
    *,
    classifier: RequestClassifier | None = None,
    action_input_extractor: ActionInputExtractor | None = None,
    response_generator: CustomerResponseGenerator | None = None,
) -> EndToEndEvalSummary:
    """Run every case and return the aggregate summary.

    Omitting a dependency defaults to the real OpenAI-backed
    implementation - this is the entry point the real CLI benchmark uses.
    Tests must always inject fakes for all three.
    """
    classifier = classifier or OpenAIRequestClassifier()
    action_input_extractor = action_input_extractor or OpenAIActionInputExtractor()
    response_generator = response_generator or OpenAICustomerResponseGenerator()

    results = [
        run_single_case(
            case,
            classifier=classifier,
            action_input_extractor=action_input_extractor,
            response_generator=response_generator,
        )
        for case in cases
    ]
    return _summarize(results)
