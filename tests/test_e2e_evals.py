"""Tests for the end-to-end evaluation framework itself.

Fully offline: every graph-driving test injects a fake classifier, fake
action-input extractor, and fake response generator (mirroring
`tests/test_graph.py`'s doubles) - `evals.e2e_runner`/`evals.e2e_checks` are
never exercised against the real OpenAI API here. `evals.e2e_cases.E2E_CASES`
itself is also inspected directly (static data, no network).
"""

import pytest

from customer_ops.classifier import ClassificationDecision
from customer_ops.action_inputs import AddressExtraction
from customer_ops.response_generator import CustomerResponse
from evals.e2e_cases import E2E_CASES, EndToEndEvalCase, ExpectedMutation, ExpectedProposedAction
from evals.e2e_checks import (
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
    contains_forbidden_facts,
    contains_required_fact_groups,
    normalize_text,
)
from evals.e2e_runner import run_e2e_evals, run_single_case


# --- Fake dependencies (offline) ----------------------------------------------------


class FakeClassifier:
    """Deterministic `RequestClassifier` double keyed by exact message text."""

    def __init__(self, mapping: dict[str, tuple[str, str]], default=("other", "low")):
        self.mapping = mapping
        self.default = default

    def classify(self, customer_message: str) -> ClassificationDecision:
        intent, urgency = self.mapping.get(customer_message, self.default)
        return ClassificationDecision(intent=intent, urgency=urgency)


class RaisingClassifier:
    """Classifier double that always raises, to test technical-failure handling."""

    def classify(self, customer_message: str):
        raise RuntimeError("simulated classifier failure")


class FakeExtractor:
    def __init__(self, address: str | None = None):
        self.address = address

    def extract_new_shipping_address(self, customer_message: str) -> AddressExtraction:
        return AddressExtraction(new_shipping_address=self.address)


class FakeGenerator:
    def __init__(self, message: str = "FAKE_RESPONSE"):
        self.message = message

    def generate(self, response_context) -> CustomerResponse:
        return CustomerResponse(message=self.message)


# --- E2E_CASES benchmark composition -------------------------------------------------


def test_benchmark_contains_unique_case_ids():
    ids = [case.case_id for case in E2E_CASES]
    assert len(ids) == len(set(ids))


def test_benchmark_contains_all_intended_intents():
    intents = {case.expected_intent for case in E2E_CASES if case.expected_intent}
    assert intents == {
        "order_status",
        "refund_request",
        "cancel_order",
        "address_change",
        "billing_issue",
        "product_issue",
        "other",
    }


def test_benchmark_contains_spanish_and_english_cases():
    messages = [case.customer_message for case in E2E_CASES]
    assert any("pedido" in msg.lower() or "¿" in msg for msg in messages)
    assert any("please" in msg.lower() for msg in messages)


def test_benchmark_contains_approval_and_rejection_cases():
    decisions = {case.human_decision for case in E2E_CASES if case.human_decision is not None}
    assert decisions == {"approved", "rejected"}


def test_benchmark_contains_safe_mutation_blocked_clarification_information_routes():
    routes = {case.expected_route for case in E2E_CASES if case.expected_route}
    assert {"action", "blocked", "clarification", "information", "approval"}.issubset(routes)


# --- Component checks (pure, synthetic data) -----------------------------------------


def base_case(**overrides) -> EndToEndEvalCase:
    fields = dict(case_id="synthetic", description="synthetic case", customer_id="cust-001", customer_message="hola")
    fields.update(overrides)
    return EndToEndEvalCase(**fields)


def test_exact_intent_check():
    case = base_case(expected_intent="order_status")
    assert check_intent(case, {"intent": "order_status"}).passed is True
    assert check_intent(case, {"intent": "cancel_order"}).passed is False
    assert check_intent(base_case(), {"intent": "anything"}).applicable is False


def test_exact_urgency_check():
    case = base_case(expected_urgency="high")
    assert check_urgency(case, {"urgency": "high"}).passed is True
    assert check_urgency(case, {"urgency": "low"}).passed is False


def test_order_resolution_check():
    case = base_case(expected_order_resolution_status="selected", expected_selected_order_id="order-1001")
    good = {"order_resolution": {"status": "selected"}, "selected_order_id": "order-1001"}
    bad = {"order_resolution": {"status": "needs_clarification"}, "selected_order_id": None}
    assert check_order_resolution(case, good).passed is True
    assert check_order_resolution(case, bad).passed is False


def test_policy_check():
    case = base_case(
        expected_policy_outcome="eligible", expected_policy_code="CANCEL_ALLOWED", expected_requires_human_approval=False
    )
    good = {"policy_assessment": {"outcome": "eligible", "policy_code": "CANCEL_ALLOWED", "requires_human_approval": False}}
    bad = {"policy_assessment": {"outcome": "blocked", "policy_code": "CANCEL_BLOCKED_STATUS", "requires_human_approval": False}}
    assert check_policy(case, good).passed is True
    assert check_policy(case, bad).passed is False


def test_route_check():
    case = base_case(expected_route="blocked")
    assert check_route(case, {"route": "blocked"}).passed is True
    assert check_route(case, {"route": "information"}).passed is False


def test_action_check():
    case = base_case(
        expected_proposed_action=ExpectedProposedAction(
            action_type="cancel_order", order_id="order-1001", requires_human_approval=False
        )
    )
    good = {"proposed_action": {"action_type": "cancel_order", "order_id": "order-1001", "requires_human_approval": False}}
    bad = {"proposed_action": {"action_type": "change_address", "order_id": "order-1001", "requires_human_approval": False}}
    assert check_proposed_action(case, good).passed is True
    assert check_proposed_action(case, good).applicable is True
    assert check_proposed_action(case, bad).passed is False

    no_action_case = base_case(expected_proposed_action=None)
    result = check_proposed_action(no_action_case, {})
    assert result.applicable is False
    assert result.passed is True
    unexpected = check_proposed_action(no_action_case, {"proposed_action": {"action_type": "cancel_order"}})
    assert unexpected.passed is False


def test_pre_approval_interrupt_check():
    case = base_case(human_decision="approved")
    result = check_approval_behavior(
        case, interrupted=True, interrupt_payload={"action_type": "issue_refund"}, pre_approval_result={"route": "approval"}, post_result={"human_decision": "approved", "action_result": {"success": True}}
    )
    assert result.passed is True

    not_interrupted = check_approval_behavior(
        case, interrupted=False, interrupt_payload=None, pre_approval_result=None, post_result={}
    )
    assert not_interrupted.passed is False


def test_approved_mutation_check():
    case = base_case(expected_mutation=ExpectedMutation(order_id="order-1001", status="cancelled"))
    orders_before = {"order-1001": {"order_id": "order-1001", "status": "pending", "payment_status": "paid", "shipping_address": "addr"}}
    order_context_after = {"orders": [{"order_id": "order-1001", "status": "cancelled", "payment_status": "paid", "shipping_address": "addr"}]}
    result = check_mutation(case, orders_before, order_context_after, None)
    assert result.passed is True


def test_rejected_no_mutation_check():
    case = base_case(expected_mutation=None)
    orders_before = {"order-1001": {"order_id": "order-1001", "status": "delivered", "payment_status": "paid", "shipping_address": "addr"}}
    order_context_after = {"orders": [{"order_id": "order-1001", "status": "delivered", "payment_status": "paid", "shipping_address": "addr"}]}
    assert check_mutation(case, orders_before, order_context_after, None).passed is True

    mutated_unexpectedly = {"orders": [{"order_id": "order-1001", "status": "cancelled", "payment_status": "paid", "shipping_address": "addr"}]}
    assert check_mutation(case, orders_before, mutated_unexpectedly, None).passed is False


def test_unrelated_order_changing_fails_mutation_check():
    case = base_case(expected_mutation=ExpectedMutation(order_id="order-1001", status="cancelled"))
    orders_before = {
        "order-1001": {"order_id": "order-1001", "status": "pending", "payment_status": "paid", "shipping_address": "a"},
        "order-1002": {"order_id": "order-1002", "status": "shipped", "payment_status": "paid", "shipping_address": "b"},
    }
    order_context_after = {
        "orders": [
            {"order_id": "order-1001", "status": "cancelled", "payment_status": "paid", "shipping_address": "a"},
            {"order_id": "order-1002", "status": "delivered", "payment_status": "paid", "shipping_address": "b"},
        ]
    }
    result = check_mutation(case, orders_before, order_context_after, None)
    assert result.passed is False
    assert "order-1002" in result.detail


def test_final_state_check():
    case = base_case(expected_final_workflow_status="completed")
    assert check_final_state(case, {"workflow_status": "completed"}).passed is True
    assert check_final_state(case, {"workflow_status": "blocked"}).passed is False


# --- Text normalization / fact-group matching ----------------------------------------


def test_normalization_strips_accents_and_case():
    assert normalize_text("CANCELADO") == normalize_text("cancelado")
    assert normalize_text("dirección") == normalize_text("direccion")
    assert normalize_text("  múltiples   espacios ") == "multiples espacios"


def test_required_fact_groups_support_alternatives():
    text = "Tu pedido fue enviado hoy."
    missing = contains_required_fact_groups(text, (("enviado", "shipped"),))
    assert missing == []


def test_missing_required_fact_fails():
    text = "Tu pedido esta pendiente."
    missing = contains_required_fact_groups(text, (("enviado", "shipped"),))
    assert missing == [("enviado", "shipped")]


def test_forbidden_fact_fails():
    text = "Tu pedido fue cancelado exitosamente."
    found = contains_forbidden_facts(text, ("cancelado", "cancelled"))
    assert found == ["cancelado"]


def test_empty_final_response_fails_where_required():
    case = base_case()
    assert check_response(case, None).passed is False
    assert check_response(case, "   ").passed is False
    assert check_response(case, "non-empty response").passed is True


# --- Regression: corrected required-fact-group phrase expectations -------------------
#
# The first real benchmark run produced valid responses whose Spanish preterite
# ("cambió"/"investigó") normalized differently than the original required
# fact-group stems expected. These tests pin the corrected expectations in
# `evals.e2e_cases.E2E_CASES` directly against the actual response text from
# that run, and confirm an obviously negative response still fails.


def _case_by_id(case_id: str) -> EndToEndEvalCase:
    return next(case for case in E2E_CASES if case.case_id == case_id)


def test_safe_address_change_cases_accept_actual_valid_spanish_responses():
    with_address = _case_by_id("safe-address-change-with-address")
    urgent = _case_by_id("safe-address-change-urgent")

    assert check_response(
        with_address,
        "La dirección de envío de order-1010 se cambió correctamente a "
        "Avenida Central 456, Cordoba, Argentina.",
    ).passed is True
    assert check_response(
        urgent,
        "La dirección de order-1008 se cambió correctamente a Calle Nueva 789, Cordoba.",
    ).passed is True


def test_safe_address_change_cases_reject_negative_response_despite_containing_cambiar_verb():
    with_address = _case_by_id("safe-address-change-with-address")
    urgent = _case_by_id("safe-address-change-urgent")
    negative_with_address = "No pudimos cambiar la direccion de order-1010 en este momento, intenta mas tarde."
    negative_urgent = "No pudimos cambiar la direccion de order-1008 en este momento, intenta mas tarde."

    assert check_response(with_address, negative_with_address).passed is False
    assert check_response(urgent, negative_urgent).passed is False


def test_billing_investigation_failed_payment_accepts_actual_valid_spanish_response():
    case = _case_by_id("billing-investigation-failed-payment")
    actual = (
        "Se investigó el problema de pago de order-1007 correctamente. "
        "La orden figura como cancelada."
    )
    assert check_response(case, actual).passed is True


def test_billing_investigation_failed_payment_rejects_unsuccessful_investigation_response():
    case = _case_by_id("billing-investigation-failed-payment")
    negative = "No se pudo investigar el problema de pago de order-1007 en este momento."
    assert check_response(case, negative).passed is False


# --- Regression: forbidden-fact false-success phrases (not bare action roots) --------
#
# The second real benchmark run showed the bare-root forbidden facts
# ("canceled", "investigat", ...) incorrectly rejected legitimate NEGATED
# statements ("cannot be canceled", "was not investigated") merely because
# they contain the action word. These tests pin the corrected forbidden-fact
# expectations in `evals.e2e_cases.E2E_CASES`: the actual valid (negative/
# blocked/rejected) response must still pass, and an explicit false claim
# that the action succeeded must still fail.


def test_blocked_cancel_shipped_accepts_actual_valid_negated_response():
    case = _case_by_id("blocked-cancel-shipped")
    actual = "Order order-1001 cannot be canceled because it has already shipped."
    assert check_response(case, actual).passed is True


def test_blocked_cancel_shipped_rejects_false_success_claim():
    case = _case_by_id("blocked-cancel-shipped")
    false_success = "Order order-1001 has been canceled successfully."
    assert check_response(case, false_success).passed is False


def test_billing_investigation_rejected_accepts_actual_valid_non_execution_response():
    case = _case_by_id("billing-investigation-rejected")
    actual = (
        "Your billing issue for order-1009 was not investigated because the "
        "proposed operation was not approved."
    )
    assert check_response(case, actual).passed is True


def test_billing_investigation_rejected_rejects_false_success_claim():
    case = _case_by_id("billing-investigation-rejected")
    false_success = (
        "Your billing issue for order-1009 was investigated successfully and "
        "the investigation is now complete."
    )
    assert check_response(case, false_success).passed is False


def test_product_investigation_approved_accepts_actual_valid_spanish_response():
    # `investigate_product_issue` creates/opens an investigation record - it
    # does not represent the underlying product issue being resolved, so the
    # required phrasing is creation/initiation language, not completion
    # language. This is the actual response from the third real benchmark run.
    case = _case_by_id("product-investigation-approved")
    actual = (
        "Hemos iniciado la investigación del producto dañado de la orden "
        "order-1006. Referencia: product-investigation-order-1006."
    )
    assert check_response(case, actual).passed is True


def test_product_investigation_approved_accepts_valid_english_equivalent():
    case = _case_by_id("product-investigation-approved")
    actual = "A product investigation was opened for order-1006."
    assert check_response(case, actual).passed is True


def test_product_investigation_approved_rejects_non_performed_investigation_response():
    case = _case_by_id("product-investigation-approved")
    negative = "No se investigó el problema del producto de la orden order-1006 en este momento."
    assert check_response(case, negative).passed is False


def test_product_investigation_approved_rejects_spanish_negated_initiation_claim():
    # Spanish negation ("no se inicio...") prepends directly onto a
    # reflexive-verb phrase - this pins that the required alternatives were
    # chosen to avoid being a substring of the negated sentence.
    case = _case_by_id("product-investigation-approved")
    negative = "No se inició la investigación para order-1006."
    assert check_response(case, negative).passed is False


def test_product_investigation_approved_rejects_english_negated_initiation_claim():
    case = _case_by_id("product-investigation-approved")
    negative = "The product investigation for order-1006 was not opened."
    assert check_response(case, negative).passed is False


def test_product_investigation_approved_rejects_resolution_claim_without_creation_claim():
    # A response claiming the underlying issue was resolved, without ever
    # stating an investigation was created/opened, must not pass merely
    # because it sounds positive.
    case = _case_by_id("product-investigation-approved")
    resolved_only = "The issue with order-1006 has been resolved to your satisfaction."
    assert check_response(case, resolved_only).passed is False


def test_product_investigation_rejected_accepts_actual_valid_non_execution_response():
    case = _case_by_id("product-investigation-rejected")
    actual = (
        "The proposed investigation for the defective item in order-1008 was "
        "not performed because it was not approved."
    )
    assert check_response(case, actual).passed is True


def test_product_investigation_rejected_rejects_false_success_claim():
    case = _case_by_id("product-investigation-rejected")
    false_success = (
        "Investigation completed for order-1008: the investigation was "
        "created and resolved successfully."
    )
    assert check_response(case, false_success).passed is False


# --- Runner integration (offline, fake dependencies) ---------------------------------


def test_overall_case_pass_requires_all_applicable_components():
    case = base_case(
        case_id="synthetic-info",
        customer_id="cust-001",
        customer_message="Hola, donde esta mi pedido order-1001?",
        expected_intent="order_status",
        expected_urgency="low",
        expected_route="information",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1001"),
    )
    classifier = FakeClassifier({"Hola, donde esta mi pedido order-1001?": ("order_status", "low")})
    result = run_single_case(
        case, classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert result.passed is True
    assert all(component.passed for component in result.component_results if component.applicable)

    mismatched_case = case.model_copy(update={"case_id": "synthetic-info-mismatch", "expected_urgency": "high"})
    mismatched_result = run_single_case(
        mismatched_case, classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert mismatched_result.passed is False
    assert any("urgency" in message for message in mismatched_result.failure_messages)


def test_one_failed_case_does_not_abort_summary_calculation():
    passing = base_case(
        case_id="passing-case",
        customer_id="cust-001",
        customer_message="passing message",
        expected_intent="order_status",
        expected_final_workflow_status="completed",
    )
    failing = base_case(
        case_id="failing-case",
        customer_id="cust-001",
        customer_message="failing message",
        expected_intent="cancel_order",  # deliberately wrong vs. the fake classifier below
        expected_final_workflow_status="completed",
    )
    classifier = FakeClassifier(
        {"passing message": ("order_status", "low"), "failing message": ("order_status", "low")}
    )
    summary = run_e2e_evals(
        [passing, failing], classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert summary.total_cases == 2
    assert summary.overall_pass_count == 1
    result_by_id = {result.case_id: result for result in summary.results}
    assert result_by_id["passing-case"].passed is True
    assert result_by_id["failing-case"].passed is False


def test_technical_exception_during_a_case_is_captured_not_raised():
    case = base_case(case_id="raises", customer_id="cust-001", customer_message="hola")
    result = run_single_case(
        case, classifier=RaisingClassifier(), action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert result.passed is False
    assert any("RuntimeError" in message for message in result.failure_messages)


def test_applicability_aware_metric_denominators():
    with_intent = base_case(case_id="with-intent", customer_message="a", expected_intent="order_status")
    without_intent = base_case(case_id="without-intent", customer_message="b", expected_intent=None)
    classifier = FakeClassifier({"a": ("order_status", "low"), "b": ("order_status", "low")})
    summary = run_e2e_evals(
        [with_intent, without_intent],
        classifier=classifier,
        action_input_extractor=FakeExtractor(),
        response_generator=FakeGenerator(),
    )
    intent_metric = next(m for m in summary.metric_accuracies if m.name == "intent")
    assert intent_metric.applicable_count == 1
    assert intent_metric.passed_count == 1


def test_approved_resume_flow_offline():
    """Full interrupt -> resume flow, exercised with fakes only."""
    case = base_case(
        case_id="refund-approved-offline",
        customer_id="cust-001",
        customer_message="refund order-1002 please",
        expected_intent="refund_request",
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="issue_refund", order_id="order-1002", requires_human_approval=True
        ),
        human_decision="approved",
        expected_mutation=ExpectedMutation(order_id="order-1002", payment_status="refunded"),
        expected_final_workflow_status="completed",
    )
    classifier = FakeClassifier({"refund order-1002 please": ("refund_request", "medium")})
    result = run_single_case(
        case, classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert result.passed, result.failure_messages
    assert result.actual_final_status == "completed"


def test_rejected_resume_flow_offline():
    case = base_case(
        case_id="refund-rejected-offline",
        customer_id="cust-001",
        customer_message="refund order-1002 please",
        expected_intent="refund_request",
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="issue_refund", order_id="order-1002", requires_human_approval=True
        ),
        human_decision="rejected",
        expected_mutation=None,
        expected_final_workflow_status="completed",
    )
    classifier = FakeClassifier({"refund order-1002 please": ("refund_request", "medium")})
    result = run_single_case(
        case, classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert result.passed, result.failure_messages


# --- Fresh-case isolation --------------------------------------------------------------


def test_fresh_case_isolation_no_mutation_leakage():
    """A mutation in one case must never be visible to a later case."""
    mutating_case = base_case(
        case_id="isolation-mutate",
        customer_id="cust-002",
        customer_message="cancel order-1005",
        expected_intent="cancel_order",
        expected_proposed_action=ExpectedProposedAction(
            action_type="cancel_order", order_id="order-1005", requires_human_approval=False
        ),
        expected_mutation=ExpectedMutation(order_id="order-1005", status="cancelled"),
    )
    observing_case = base_case(
        case_id="isolation-observe",
        customer_id="cust-002",
        customer_message="where is order-1005",
        expected_intent="order_status",
        expected_selected_order_id="order-1005",
        expected_order_resolution_status="selected",
        expected_mutation=None,  # order-1005 must still be "pending" - its true fixture baseline
    )
    classifier = FakeClassifier(
        {"cancel order-1005": ("cancel_order", "medium"), "where is order-1005": ("order_status", "low")}
    )

    summary = run_e2e_evals(
        [mutating_case, observing_case],
        classifier=classifier,
        action_input_extractor=FakeExtractor(),
        response_generator=FakeGenerator(),
    )
    result_by_id = {result.case_id: result for result in summary.results}
    assert result_by_id["isolation-mutate"].passed, result_by_id["isolation-mutate"].failure_messages
    assert result_by_id["isolation-observe"].passed, result_by_id["isolation-observe"].failure_messages


def test_fresh_case_isolation_is_order_independent():
    """Running the same two cases in reversed order must give the same results."""
    mutating_case = base_case(
        case_id="isolation-mutate-2",
        customer_id="cust-002",
        customer_message="cancel order-1005 now",
        expected_intent="cancel_order",
        expected_proposed_action=ExpectedProposedAction(
            action_type="cancel_order", order_id="order-1005", requires_human_approval=False
        ),
        expected_mutation=ExpectedMutation(order_id="order-1005", status="cancelled"),
    )
    observing_case = base_case(
        case_id="isolation-observe-2",
        customer_id="cust-002",
        customer_message="where is order-1005 now",
        expected_intent="order_status",
        expected_selected_order_id="order-1005",
        expected_order_resolution_status="selected",
        expected_mutation=None,
    )
    classifier = FakeClassifier(
        {"cancel order-1005 now": ("cancel_order", "medium"), "where is order-1005 now": ("order_status", "low")}
    )

    summary = run_e2e_evals(
        [observing_case, mutating_case],  # reversed order vs. the previous test
        classifier=classifier,
        action_input_extractor=FakeExtractor(),
        response_generator=FakeGenerator(),
    )
    assert all(result.passed for result in summary.results), [r.failure_messages for r in summary.results]


# --- Result/summary serialization -----------------------------------------------------


def test_result_and_summary_are_json_serializable():
    import json

    case = base_case(case_id="serialization-check", customer_id="cust-001", customer_message="hola", expected_intent="order_status")
    classifier = FakeClassifier({"hola": ("order_status", "low")})
    summary = run_e2e_evals(
        [case], classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    json.dumps(summary.model_dump(mode="json"))


# --- Offline guarantee -----------------------------------------------------------------


def test_e2e_eval_framework_makes_no_network_calls(no_network):
    case = base_case(case_id="offline-check", customer_id="cust-001", customer_message="hola", expected_intent="order_status")
    classifier = FakeClassifier({"hola": ("order_status", "low")})
    summary = run_e2e_evals(
        [case], classifier=classifier, action_input_extractor=FakeExtractor(), response_generator=FakeGenerator()
    )
    assert summary.total_cases == 1
