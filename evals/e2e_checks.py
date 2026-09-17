"""Transparent, rule-based checks for the end-to-end evaluation benchmark.

Every check here is explicit Python comparing structured workflow state (or
normalized response text) against an `EndToEndEvalCase`'s declared
expectations - never an LLM-as-a-judge. `ComponentCheckResult.applicable`
governs whether a check counts toward its NAMED per-metric accuracy stat
(e.g. "action proposal accuracy" only counts cases designed to test action
proposal); `ComponentCheckResult.passed` always reflects genuine
correctness for THIS case, applicable or not, so a case that unexpectedly
produces a phantom proposed action (when none was expected) still fails
overall even though it is excluded from the action-proposal accuracy
denominator.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from evals.e2e_cases import EndToEndEvalCase


class ComponentCheckResult(BaseModel):
    """The outcome of one evaluation dimension for one case.

    `applicable=False` means this case does not exercise this dimension
    (e.g. no order-resolution expectation was declared) - such results are
    excluded from that dimension's accuracy denominator, but `passed` still
    reflects whatever trivial/negative correctness could be established
    (e.g. "no proposed_action was present, as expected").
    """

    name: str
    applicable: bool
    passed: bool
    detail: str = ""


# --- Text normalization and fact-group matching -----------------------------------


def normalize_text(text: str) -> str:
    """Normalize text for transparent substring/fact matching.

    Unicode-normalizes, strips accents, casefolds, and collapses
    whitespace. Used only for comparison - never alters the actual stored
    model output.
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return " ".join(without_accents.casefold().split())


def contains_required_fact_groups(
    text: str, groups: Sequence[Sequence[str]]
) -> list[Sequence[str]]:
    """Return the required fact groups for which NO alternative was found.

    Each group is a tuple of acceptable alternative phrasings; a group
    passes if at least one alternative appears (after normalization). An
    empty return value means every group passed.
    """
    normalized_text = normalize_text(text)
    missing: list[Sequence[str]] = []
    for group in groups:
        alternatives = [normalize_text(alt) for alt in group]
        if not any(alt in normalized_text for alt in alternatives):
            missing.append(group)
    return missing


def contains_forbidden_facts(text: str, forbidden: Sequence[str]) -> list[str]:
    """Return which forbidden facts were found in `text` (after normalization).

    An empty return value means none of the forbidden facts were found.
    """
    normalized_text = normalize_text(text)
    return [fact for fact in forbidden if normalize_text(fact) in normalized_text]


# --- Component checks --------------------------------------------------------------


def check_intent(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_intent is None:
        return ComponentCheckResult(name="intent", applicable=False, passed=True)
    actual = result.get("intent")
    passed = actual == case.expected_intent
    detail = "" if passed else f"expected intent={case.expected_intent!r}, got {actual!r}"
    return ComponentCheckResult(name="intent", applicable=True, passed=passed, detail=detail)


def check_urgency(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_urgency is None:
        return ComponentCheckResult(name="urgency", applicable=False, passed=True)
    actual = result.get("urgency")
    passed = actual == case.expected_urgency
    detail = "" if passed else f"expected urgency={case.expected_urgency!r}, got {actual!r}"
    return ComponentCheckResult(name="urgency", applicable=True, passed=passed, detail=detail)


def check_order_resolution(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_order_resolution_status is None:
        return ComponentCheckResult(name="order_resolution", applicable=False, passed=True)
    problems: list[str] = []
    order_resolution = result.get("order_resolution") or {}
    actual_status = order_resolution.get("status")
    if actual_status != case.expected_order_resolution_status:
        problems.append(
            f"expected order_resolution.status={case.expected_order_resolution_status!r}, "
            f"got {actual_status!r}"
        )
    actual_selected = result.get("selected_order_id")
    if actual_selected != case.expected_selected_order_id:
        problems.append(
            f"expected selected_order_id={case.expected_selected_order_id!r}, got {actual_selected!r}"
        )
    return ComponentCheckResult(
        name="order_resolution", applicable=True, passed=not problems, detail="; ".join(problems)
    )


def check_policy(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_policy_outcome is None:
        return ComponentCheckResult(name="policy", applicable=False, passed=True)
    problems: list[str] = []
    policy_assessment = result.get("policy_assessment") or {}
    if policy_assessment.get("outcome") != case.expected_policy_outcome:
        problems.append(
            f"expected policy outcome={case.expected_policy_outcome!r}, "
            f"got {policy_assessment.get('outcome')!r}"
        )
    if case.expected_policy_code is not None and policy_assessment.get("policy_code") != case.expected_policy_code:
        problems.append(
            f"expected policy_code={case.expected_policy_code!r}, got {policy_assessment.get('policy_code')!r}"
        )
    if (
        case.expected_requires_human_approval is not None
        and policy_assessment.get("requires_human_approval") != case.expected_requires_human_approval
    ):
        problems.append(
            f"expected requires_human_approval={case.expected_requires_human_approval!r}, "
            f"got {policy_assessment.get('requires_human_approval')!r}"
        )
    return ComponentCheckResult(name="policy", applicable=True, passed=not problems, detail="; ".join(problems))


def check_route(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_route is None:
        return ComponentCheckResult(name="route", applicable=False, passed=True)
    actual = result.get("route")
    passed = actual == case.expected_route
    detail = "" if passed else f"expected route={case.expected_route!r}, got {actual!r}"
    return ComponentCheckResult(name="route", applicable=True, passed=passed, detail=detail)


def check_proposed_action(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    """Applicable only when an action IS expected - see module docstring."""
    actual = result.get("proposed_action")
    expected = case.expected_proposed_action

    if expected is None:
        passed = actual is None
        detail = "" if passed else f"expected no proposed_action, got {actual!r}"
        return ComponentCheckResult(name="proposed_action", applicable=False, passed=passed, detail=detail)

    if actual is None:
        return ComponentCheckResult(
            name="proposed_action", applicable=True, passed=False, detail="expected a proposed_action, got none."
        )

    problems: list[str] = []
    if actual.get("action_type") != expected.action_type:
        problems.append(f"expected action_type={expected.action_type!r}, got {actual.get('action_type')!r}")
    if actual.get("order_id") != expected.order_id:
        problems.append(f"expected order_id={expected.order_id!r}, got {actual.get('order_id')!r}")
    if actual.get("requires_human_approval") != expected.requires_human_approval:
        problems.append(
            f"expected requires_human_approval={expected.requires_human_approval!r}, "
            f"got {actual.get('requires_human_approval')!r}"
        )
    return ComponentCheckResult(
        name="proposed_action", applicable=True, passed=not problems, detail="; ".join(problems)
    )


def check_approval_behavior(
    case: EndToEndEvalCase,
    *,
    interrupted: bool,
    interrupt_payload: Any,
    pre_approval_result: dict[str, Any] | None,
    post_result: dict[str, Any],
) -> ComponentCheckResult:
    """Applicable only for cases designed to exercise approval - see module docstring."""
    if case.human_decision is None:
        if interrupted:
            return ComponentCheckResult(
                name="approval_behavior",
                applicable=False,
                passed=False,
                detail="Unexpected interrupt for a case not designed to require approval.",
            )
        return ComponentCheckResult(name="approval_behavior", applicable=False, passed=True)

    problems: list[str] = []
    if not interrupted:
        problems.append("expected an interrupt but the graph completed without pausing.")
        return ComponentCheckResult(name="approval_behavior", applicable=True, passed=False, detail="; ".join(problems))

    if not isinstance(interrupt_payload, dict):
        problems.append("interrupt payload was not a JSON-like mapping.")
    if pre_approval_result is not None:
        if "action_result" in pre_approval_result:
            problems.append("action_result present before approval.")
        if "final_response" in pre_approval_result:
            problems.append("final_response present before approval.")

    if post_result.get("human_decision") != case.human_decision:
        problems.append(
            f"expected human_decision={case.human_decision!r}, got {post_result.get('human_decision')!r}"
        )

    if case.human_decision == "approved":
        action_result = post_result.get("action_result") or {}
        if not action_result.get("success"):
            problems.append("expected action_result.success=True after approval.")
    else:
        if "action_result" in post_result:
            problems.append("action_result present after rejection.")

    return ComponentCheckResult(
        name="approval_behavior", applicable=True, passed=not problems, detail="; ".join(problems)
    )


def check_final_state(case: EndToEndEvalCase, result: dict[str, Any]) -> ComponentCheckResult:
    if case.expected_final_workflow_status is None:
        return ComponentCheckResult(name="final_state", applicable=False, passed=True)
    actual = result.get("workflow_status")
    passed = actual == case.expected_final_workflow_status
    detail = "" if passed else f"expected workflow_status={case.expected_final_workflow_status!r}, got {actual!r}"
    return ComponentCheckResult(name="final_state", applicable=True, passed=passed, detail=detail)


def check_mutation(
    case: EndToEndEvalCase,
    orders_before: dict[str, dict[str, Any]],
    order_context_after: dict[str, Any] | None,
    action_result: dict[str, Any] | None,
) -> ComponentCheckResult:
    """Compare each customer order before/after against `case.expected_mutation`.

    `orders_before` maps order_id -> JSON-friendly order dict, captured from
    the fresh store BEFORE the graph ran. Every order not named by
    `expected_mutation` (or every order at all, when no mutation is
    expected) must be byte-identical after.
    """
    if order_context_after is None:
        return ComponentCheckResult(
            name="mutation", applicable=True, passed=False, detail="order_context missing from final state."
        )
    after_by_id = {order["order_id"]: order for order in order_context_after.get("orders", [])}

    expected = case.expected_mutation
    target_id = expected.order_id if expected is not None else None
    problems: list[str] = []

    for order_id, before in orders_before.items():
        after = after_by_id.get(order_id)
        if after is None:
            problems.append(f"order {order_id} missing from final order_context")
            continue

        if order_id != target_id:
            if before != after:
                problems.append(f"unrelated order {order_id} changed unexpectedly")
            continue

        expected_status = expected.status if expected.status is not None else before.get("status")
        if after.get("status") != expected_status:
            problems.append(f"order {order_id}: expected status={expected_status!r}, got {after.get('status')!r}")

        expected_payment_status = (
            expected.payment_status if expected.payment_status is not None else before.get("payment_status")
        )
        if after.get("payment_status") != expected_payment_status:
            problems.append(
                f"order {order_id}: expected payment_status={expected_payment_status!r}, "
                f"got {after.get('payment_status')!r}"
            )

        if expected.shipping_address_contains is not None:
            address = after.get("shipping_address") or ""
            if expected.shipping_address_contains.lower() not in address.lower():
                problems.append(
                    f"order {order_id}: shipping_address does not contain "
                    f"{expected.shipping_address_contains!r} (got {address!r})"
                )
        elif after.get("shipping_address") != before.get("shipping_address"):
            problems.append(f"order {order_id}: shipping_address changed unexpectedly")

    if expected is not None and expected.reference_id is not None:
        actual_reference_id = (action_result or {}).get("reference_id")
        if actual_reference_id != expected.reference_id:
            problems.append(
                f"expected action_result.reference_id={expected.reference_id!r}, got {actual_reference_id!r}"
            )

    return ComponentCheckResult(name="mutation", applicable=True, passed=not problems, detail="; ".join(problems))


def check_response(case: EndToEndEvalCase, final_response: str | None) -> ComponentCheckResult:
    """Deterministic substring/fact checks only - never an LLM-as-a-judge.

    These checks are transparent but cannot detect every possible
    hallucination or subtle semantic error; they validate the concrete
    expected facts a case declares, nothing more.
    """
    if final_response is None or not final_response.strip():
        return ComponentCheckResult(name="response", applicable=True, passed=False, detail="final_response is empty or missing.")

    missing_groups = contains_required_fact_groups(final_response, case.response_required_fact_groups)
    found_forbidden = contains_forbidden_facts(final_response, case.response_forbidden_facts)

    problems: list[str] = []
    if missing_groups:
        problems.append(f"missing required fact groups: {missing_groups!r}")
    if found_forbidden:
        problems.append(f"contains forbidden facts: {found_forbidden!r}")

    return ComponentCheckResult(name="response", applicable=True, passed=not problems, detail="; ".join(problems))
