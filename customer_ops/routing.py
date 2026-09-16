"""Deterministic case routing for the Mercora Customer Operations Agent.

Decides which workflow branch follows policy evaluation, from structured
state only (`intent`, `order_resolution`, `policy_assessment`) - never from
an LLM call. Policy answers "what is allowed?"; routing answers "what
workflow path follows from that?" - the two concerns stay deliberately
separate.
"""

from __future__ import annotations

from customer_ops.order_resolution import OrderResolution
from customer_ops.policies import PolicyAssessment
from customer_ops.state import CaseRoute, Intent

# Every `PolicyOutcome` must appear here exactly once. `information_only`
# (e.g. order_status) and `not_applicable` (intent "other") share the
# `information` route rather than a separate `no_action` route: both are
# non-operational, terminal outcomes, and the "other" case never has a
# selected order for a dedicated node to reference anyway.
_ROUTE_BY_OUTCOME: dict[str, CaseRoute] = {
    "needs_clarification": "clarification",
    "information_only": "information",
    "not_applicable": "information",
    "eligible": "action",
    "review_required": "approval",
    "blocked": "blocked",
}

# The `requires_human_approval` value every outcome must carry, given a
# correctly behaving policy engine. Used only to detect corrupted state.
_EXPECTED_APPROVAL_BY_OUTCOME: dict[str, bool] = {
    "needs_clarification": False,
    "information_only": False,
    "not_applicable": False,
    "eligible": False,
    "review_required": True,
    "blocked": False,
}

# Outcomes that can only be produced from a "selected" order resolution.
_SELECTED_ORDER_OUTCOMES = frozenset({"information_only", "eligible", "blocked", "review_required"})


class RoutingError(Exception):
    """Raised when policy/order-resolution state is internally inconsistent.

    Reserved for combinations that should never occur given a correctly
    behaving policy engine (e.g. outcome='eligible' paired with
    requires_human_approval=True) - never used to paper over such
    inconsistencies by silently choosing a plausible-looking route.
    """


def determine_case_route(
    intent: Intent,
    order_resolution: OrderResolution,
    policy_assessment: PolicyAssessment,
) -> CaseRoute:
    """Determine the next workflow branch from policy evaluation results.

    The branch is derived from `policy_assessment.outcome`; `intent` and
    `order_resolution` are cross-checked for internal consistency (e.g. an
    "eligible" outcome must correspond to a "selected" order resolution) so
    that corrupted state raises `RoutingError` instead of silently landing
    on a plausible-looking branch.
    """
    outcome = policy_assessment.outcome

    if outcome not in _ROUTE_BY_OUTCOME:
        raise RoutingError(f"No route is defined for policy outcome {outcome!r}.")

    expected_approval = _EXPECTED_APPROVAL_BY_OUTCOME[outcome]
    if policy_assessment.requires_human_approval != expected_approval:
        raise RoutingError(
            f"Policy outcome {outcome!r} is inconsistent with "
            f"requires_human_approval={policy_assessment.requires_human_approval!r} "
            f"(expected {expected_approval!r})."
        )

    if outcome == "needs_clarification" and order_resolution.status != "needs_clarification":
        raise RoutingError(
            "Policy outcome 'needs_clarification' is inconsistent with "
            f"order_resolution.status={order_resolution.status!r}."
        )

    if outcome in _SELECTED_ORDER_OUTCOMES and order_resolution.status != "selected":
        raise RoutingError(
            f"Policy outcome {outcome!r} requires a selected order, but "
            f"order_resolution.status={order_resolution.status!r}."
        )

    if outcome == "not_applicable" and intent != "other":
        raise RoutingError(
            f"Policy outcome 'not_applicable' is only valid for intent 'other', got {intent!r}."
        )

    return _ROUTE_BY_OUTCOME[outcome]
