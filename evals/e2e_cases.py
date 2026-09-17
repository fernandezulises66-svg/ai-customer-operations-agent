"""The curated end-to-end evaluation benchmark for the Mercora workflow.

`EndToEndEvalCase` is a declarative, serializable-friendly contract - no
callback functions, no hidden logic. Every case is grounded in the actual
synthetic fixtures (`data/customers.json`, `data/orders.json`): real
customer IDs, real order IDs, and order statuses read directly from those
files, never assumed. `E2E_CASES` is the 20-case curated benchmark; it
collectively exercises every supported intent, both languages, every
routing branch, the full approval interrupt/resume flow (both decisions),
and the read-only/safe/sensitive mutation boundaries - see
`evals/run_e2e_evals.py` for how it is executed and `README.md` for the
benchmark's documented scope and limitations.
"""

from __future__ import annotations

from pydantic import BaseModel

from customer_ops.models import OrderStatus, PaymentStatus
from customer_ops.state import (
    ActionType,
    CaseRoute,
    HumanDecision,
    Intent,
    OrderResolutionStatus,
    PolicyCode,
    PolicyOutcome,
    Urgency,
    WorkflowStatus,
)


class ExpectedProposedAction(BaseModel):
    """The `ProposedAction` a case expects to find in final state, if any."""

    action_type: ActionType
    order_id: str
    requires_human_approval: bool


class ExpectedMutation(BaseModel):
    """What should change (or stay unchanged) in the operational store.

    Only `order_id` is required; every other field defaults to "must equal
    its pre-case value" when left unset, so a case only needs to name the
    field(s) it actually expects to change. `reference_id` is checked
    against `action_result.reference_id` (investigations never change an
    order's own fields).
    """

    order_id: str
    status: OrderStatus | None = None
    payment_status: PaymentStatus | None = None
    shipping_address_contains: str | None = None
    reference_id: str | None = None


class EndToEndEvalCase(BaseModel):
    """One curated end-to-end benchmark case.

    Every `expected_*` field is optional; a `None` value means that
    dimension is not being checked for this case (see
    `evals/e2e_checks.py` for how "not applicable" affects per-metric
    accuracy vs. overall case pass). `response_required_fact_groups` is a
    tuple of alternative-phrasing tuples - at least one alternative in
    each group must appear in the final response.
    """

    case_id: str
    description: str
    customer_id: str
    customer_message: str

    expected_intent: Intent | None = None
    expected_urgency: Urgency | None = None

    expected_order_resolution_status: OrderResolutionStatus | None = None
    expected_selected_order_id: str | None = None

    expected_policy_outcome: PolicyOutcome | None = None
    expected_policy_code: PolicyCode | None = None
    expected_requires_human_approval: bool | None = None

    expected_route: CaseRoute | None = None

    expected_proposed_action: ExpectedProposedAction | None = None

    # Only for approval-required cases: the decision the runner supplies via
    # Command(resume={"decision": ...}) on the SAME thread_id after the
    # graph interrupts. None means this case is not expected to interrupt.
    human_decision: HumanDecision | None = None

    expected_final_workflow_status: WorkflowStatus | None = None

    # None means "no order field should change anywhere for this customer".
    expected_mutation: ExpectedMutation | None = None

    response_required_fact_groups: tuple[tuple[str, ...], ...] = ()
    response_forbidden_facts: tuple[str, ...] = ()


# --- The curated benchmark ----------------------------------------------------------
#
# Grounded in data/customers.json / data/orders.json:
#   cust-001 Mariana Torres  (active,    premium)  -> order-1001 shipped/paid,
#                                                      order-1002 delivered/paid,
#                                                      order-1003 cancelled/refunded
#   cust-002 Diego Fernandez (active,    standard) -> order-1004 processing/paid,
#                                                      order-1005 pending/pending
#   cust-003 Lucia Gomez     (active,    standard) -> order-1006 delivered/paid,
#                                                      order-1007 cancelled/failed
#   cust-004 Carlos Ruiz     (suspended, standard) -> order-1008 pending/pending
#   cust-005 Ana Beltran     (active,    premium)  -> no orders
#   cust-006 Javier Morales  (active,    standard) -> order-1009 shipped/paid,
#                                                      order-1010 processing/paid

E2E_CASES: tuple[EndToEndEvalCase, ...] = (
    # --- Information (2) --------------------------------------------------------
    EndToEndEvalCase(
        case_id="info-es-order-status-shipped",
        description="Spanish order-status request with an explicit shipped order.",
        customer_id="cust-001",
        customer_message="Hola, ¿dónde está mi pedido order-1001?",
        expected_intent="order_status",
        expected_urgency="low",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1001",
        expected_policy_outcome="information_only",
        expected_policy_code="ORDER_STATUS_INFO",
        expected_requires_human_approval=False,
        expected_route="information",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1001"),
        response_required_fact_groups=(
            ("order-1001",),
            ("enviado", "shipped", "en camino", "envio", "en transito"),
        ),
    ),
    EndToEndEvalCase(
        case_id="info-en-order-status-shipped",
        description="English order-status request with an explicit shipped order.",
        customer_id="cust-006",
        customer_message="Can you tell me the status of order-1009?",
        expected_intent="order_status",
        expected_urgency="low",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1009",
        expected_policy_outcome="information_only",
        expected_policy_code="ORDER_STATUS_INFO",
        expected_requires_human_approval=False,
        expected_route="information",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1009"),
        response_required_fact_groups=(
            ("order-1009",),
            ("shipped", "on its way", "in transit"),
        ),
    ),
    # --- Clarification (4) -------------------------------------------------------
    EndToEndEvalCase(
        case_id="clarify-ambiguous-multi-order",
        description="Order-specific request with no order ID for a customer with multiple orders.",
        customer_id="cust-001",
        customer_message="Quiero cancelar mi pedido.",
        expected_intent="cancel_order",
        expected_urgency="medium",
        expected_order_resolution_status="needs_clarification",
        expected_selected_order_id=None,
        expected_policy_outcome="needs_clarification",
        expected_policy_code="ORDER_REQUIRED",
        expected_requires_human_approval=False,
        expected_route="clarification",
        expected_final_workflow_status="completed",
        expected_mutation=None,
        response_required_fact_groups=(("pedido", "order"),),
    ),
    EndToEndEvalCase(
        case_id="clarify-unknown-order-id",
        description="Explicit order ID that does not exist anywhere in the fixtures.",
        customer_id="cust-002",
        customer_message="Where is my order order-9999?",
        expected_intent="order_status",
        expected_urgency="low",
        expected_order_resolution_status="needs_clarification",
        expected_selected_order_id=None,
        expected_policy_outcome="needs_clarification",
        expected_policy_code="ORDER_REQUIRED",
        expected_requires_human_approval=False,
        expected_route="clarification",
        expected_final_workflow_status="completed",
        expected_mutation=None,
        response_required_fact_groups=(("order", "pedido"),),
    ),
    EndToEndEvalCase(
        case_id="clarify-zero-orders",
        description="Customer with zero orders asks about an order.",
        customer_id="cust-005",
        customer_message="Where is my order?",
        expected_intent="order_status",
        expected_urgency="low",
        expected_order_resolution_status="needs_clarification",
        expected_selected_order_id=None,
        expected_policy_outcome="needs_clarification",
        expected_policy_code="ORDER_REQUIRED",
        expected_requires_human_approval=False,
        expected_route="clarification",
        expected_final_workflow_status="completed",
        expected_mutation=None,
        response_required_fact_groups=(("order", "pedido"),),
    ),
    EndToEndEvalCase(
        case_id="clarify-address-change-missing-address",
        description=(
            "Eligible address-change intent with no replacement address supplied - "
            "order resolves first, clarification happens at action-input stage, and "
            "the ProposedAction from that stage is retained for traceability."
        ),
        customer_id="cust-002",
        customer_message="Quiero cambiar la dirección de order-1004.",
        expected_intent="address_change",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1004",
        expected_policy_outcome="eligible",
        expected_policy_code="ADDRESS_CHANGE_ALLOWED",
        expected_requires_human_approval=False,
        expected_route="clarification",
        expected_proposed_action=ExpectedProposedAction(
            action_type="change_address", order_id="order-1004", requires_human_approval=False
        ),
        expected_final_workflow_status="completed",
        expected_mutation=None,
        response_required_fact_groups=(("direccion", "address"),),
    ),
    # --- Safe actions (3) --------------------------------------------------------
    EndToEndEvalCase(
        case_id="safe-cancel-pending",
        description="Eligible cancellation on a pending order.",
        customer_id="cust-002",
        customer_message="Please cancel order-1005.",
        expected_intent="cancel_order",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1005",
        expected_policy_outcome="eligible",
        expected_policy_code="CANCEL_ALLOWED",
        expected_requires_human_approval=False,
        expected_route="action",
        expected_proposed_action=ExpectedProposedAction(
            action_type="cancel_order", order_id="order-1005", requires_human_approval=False
        ),
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1005", status="cancelled"),
        response_required_fact_groups=(
            ("order-1005",),
            ("cancelad", "cancelled", "canceled"),
        ),
    ),
    EndToEndEvalCase(
        case_id="safe-address-change-with-address",
        description="Eligible address change with an explicit replacement address.",
        customer_id="cust-006",
        customer_message=(
            "Por favor cambia la direccion de envio de order-1010 a "
            "Avenida Central 456, Cordoba, Argentina."
        ),
        expected_intent="address_change",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1010",
        expected_policy_outcome="eligible",
        expected_policy_code="ADDRESS_CHANGE_ALLOWED",
        expected_requires_human_approval=False,
        expected_route="action",
        expected_proposed_action=ExpectedProposedAction(
            action_type="change_address", order_id="order-1010", requires_human_approval=False
        ),
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1010", shipping_address_contains="Avenida Central 456"),
        response_required_fact_groups=(
            ("order-1010",),
            (
                "actualiz",
                "cambiad",
                "se cambio correctamente",
                "direccion actualizada",
                "direccion cambiada",
                "updated successfully",
                "changed successfully",
            ),
        ),
    ),
    EndToEndEvalCase(
        case_id="safe-address-change-urgent",
        description=(
            "Explicitly urgent address change with clear, concrete time pressure "
            "(moving before delivery) rather than merely emotional wording."
        ),
        customer_id="cust-004",
        customer_message=(
            "Necesito cambiar la direccion de mi pedido order-1008 urgentemente a "
            "Calle Nueva 789, Cordoba, porque me mudo manana y el paquete no debe "
            "llegar a mi direccion anterior."
        ),
        expected_intent="address_change",
        expected_urgency="high",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1008",
        expected_policy_outcome="eligible",
        expected_policy_code="ADDRESS_CHANGE_ALLOWED",
        expected_requires_human_approval=False,
        expected_route="action",
        expected_proposed_action=ExpectedProposedAction(
            action_type="change_address", order_id="order-1008", requires_human_approval=False
        ),
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1008", shipping_address_contains="Calle Nueva 789"),
        response_required_fact_groups=(
            ("order-1008",),
            (
                "actualiz",
                "cambiad",
                "se cambio correctamente",
                "direccion actualizada",
                "direccion cambiada",
                "updated successfully",
                "changed successfully",
            ),
        ),
    ),
    # --- Blocked (3) ---------------------------------------------------------------
    EndToEndEvalCase(
        case_id="blocked-cancel-shipped",
        description="Cancellation request on an already-shipped order.",
        customer_id="cust-001",
        customer_message="Please cancel order-1001.",
        expected_intent="cancel_order",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1001",
        expected_policy_outcome="blocked",
        expected_policy_code="CANCEL_BLOCKED_STATUS",
        expected_requires_human_approval=False,
        expected_route="blocked",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1001"),
        # Forbid false-success claims only - "cancelado"/"cancelled"/"canceled"
        # alone would also match the legitimate negated response ("cannot be
        # canceled"/"no se puede cancelar"), so these are specific completed-
        # action phrases instead of the bare verb/adjective root.
        response_forbidden_facts=(
            "was canceled",
            "was cancelled",
            "has been canceled",
            "has been cancelled",
            "cancelado exitosamente",
            "se cancelo correctamente",
            "cancelado correctamente",
        ),
    ),
    EndToEndEvalCase(
        case_id="blocked-address-change-delivered",
        description="Address-change request on an already-delivered order.",
        customer_id="cust-003",
        customer_message="Quiero cambiar la direccion de order-1006.",
        expected_intent="address_change",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1006",
        expected_policy_outcome="blocked",
        expected_policy_code="ADDRESS_CHANGE_BLOCKED_STATUS",
        expected_requires_human_approval=False,
        expected_route="blocked",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1006"),
        response_forbidden_facts=("actualiz", "updated"),
    ),
    EndToEndEvalCase(
        case_id="blocked-refund-already-refunded",
        description="Refund request for an order that is already refunded.",
        customer_id="cust-001",
        customer_message="Quiero que me devuelvan el dinero de order-1003.",
        expected_intent="refund_request",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1003",
        expected_policy_outcome="blocked",
        expected_policy_code="ALREADY_REFUNDED",
        expected_requires_human_approval=False,
        expected_route="blocked",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1003"),
        response_required_fact_groups=(("reembolsad", "refunded", "ya"),),
    ),
    # --- Approval / refund (2) ------------------------------------------------------
    EndToEndEvalCase(
        case_id="approval-refund-approved",
        description="Refund request on a delivered/paid order, approved by the reviewer.",
        customer_id="cust-001",
        customer_message="I would like a refund for order-1002.",
        expected_intent="refund_request",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1002",
        expected_policy_outcome="review_required",
        expected_policy_code="REFUND_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="issue_refund", order_id="order-1002", requires_human_approval=True
        ),
        human_decision="approved",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1002", payment_status="refunded"),
        response_required_fact_groups=(
            ("order-1002",),
            ("reembols", "refund"),
        ),
    ),
    EndToEndEvalCase(
        case_id="approval-refund-rejected",
        description="Refund request on a delivered/paid order, rejected by the reviewer.",
        customer_id="cust-003",
        customer_message="Quiero un reembolso para mi pedido order-1006.",
        expected_intent="refund_request",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1006",
        expected_policy_outcome="review_required",
        expected_policy_code="REFUND_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="issue_refund", order_id="order-1006", requires_human_approval=True
        ),
        human_decision="rejected",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1006"),
        response_forbidden_facts=("refunded", "reembolsado", "reembolsada"),
    ),
    # --- Billing investigations (2) --------------------------------------------------
    EndToEndEvalCase(
        case_id="billing-investigation-approved",
        description="Billing issue (duplicate charge), approved by the reviewer.",
        customer_id="cust-002",
        customer_message="Me cobraron dos veces por order-1004.",
        expected_intent="billing_issue",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1004",
        expected_policy_outcome="review_required",
        expected_policy_code="BILLING_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="investigate_billing", order_id="order-1004", requires_human_approval=True
        ),
        human_decision="approved",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1004", reference_id="billing-investigation-order-1004"),
        response_required_fact_groups=(
            ("order-1004",),
            ("investigat", "revis", "review"),
        ),
    ),
    EndToEndEvalCase(
        case_id="billing-investigation-rejected",
        description="Billing issue (incorrect charge), rejected by the reviewer.",
        customer_id="cust-006",
        customer_message="I was charged incorrectly for order-1009.",
        expected_intent="billing_issue",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1009",
        expected_policy_outcome="review_required",
        expected_policy_code="BILLING_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="investigate_billing", order_id="order-1009", requires_human_approval=True
        ),
        human_decision="rejected",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1009"),
        # Forbid false-success claims only - the bare root "investigat" also
        # matches the legitimate rejected response ("was not investigated"),
        # so these are specific completed-investigation phrases instead.
        response_forbidden_facts=(
            "investigated successfully",
            "investigation completed",
            "investigation created",
            "se investigo correctamente",
            "investigacion completada",
            "se creo una investigacion",
        ),
    ),
    # --- Product investigations (2) --------------------------------------------------
    EndToEndEvalCase(
        case_id="product-investigation-approved",
        description="Product issue (damaged item), approved by the reviewer.",
        customer_id="cust-003",
        customer_message="El producto de order-1006 llego danado.",
        expected_intent="product_issue",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1006",
        expected_policy_outcome="review_required",
        expected_policy_code="PRODUCT_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="investigate_product_issue", order_id="order-1006", requires_human_approval=True
        ),
        human_decision="approved",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1006", reference_id="product-investigation-order-1006"),
        # `investigate_product_issue` creates/opens an investigation record -
        # it does not represent the underlying product issue being resolved.
        # These alternatives require a specific creation/initiation/
        # registration claim (never a bare root like "investig"/"inici", and
        # never completion/resolution wording, which would overstate what
        # this simulated action does). The Spanish alternatives deliberately
        # use noun-before-adjective order ("investigacion iniciada", not
        # "se inicio la investigacion") because Spanish negation ("no se
        # inicio...") prepends directly onto a reflexive-verb phrase and
        # would otherwise still contain it as a substring.
        response_required_fact_groups=(
            ("order-1006",),
            (
                "hemos iniciado la investigacion",
                "investigacion iniciada",
                "investigacion abierta",
                "investigacion registrada",
                "investigation created",
                "investigation opened",
                "investigation started",
                "investigation was opened",
                "opened an investigation",
                "created an investigation",
            ),
        ),
    ),
    EndToEndEvalCase(
        case_id="product-investigation-rejected",
        description="Product issue (defective item), rejected by the reviewer.",
        customer_id="cust-004",
        customer_message="The item in order-1008 arrived defective.",
        expected_intent="product_issue",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1008",
        expected_policy_outcome="review_required",
        expected_policy_code="PRODUCT_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="investigate_product_issue", order_id="order-1008", requires_human_approval=True
        ),
        human_decision="rejected",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1008"),
        # Forbid false-success claims only - the bare root "investigat" also
        # matches the legitimate rejected response ("investigation ... was
        # not performed"), so these are specific completed-investigation
        # phrases instead.
        response_forbidden_facts=(
            "investigated successfully",
            "investigation completed",
            "investigation created",
            "se investigo correctamente",
            "investigacion completada",
            "se creo una investigacion",
        ),
    ),
    # --- Other (1) -------------------------------------------------------------------
    EndToEndEvalCase(
        case_id="other-unsupported-request",
        description="A genuinely unsupported request that maps to intent 'other'.",
        customer_id="cust-001",
        customer_message="Hola, ¿venden tarjetas de regalo?",
        expected_intent="other",
        expected_urgency="low",
        expected_order_resolution_status="not_required",
        expected_selected_order_id=None,
        expected_policy_outcome="not_applicable",
        expected_policy_code="NOT_APPLICABLE",
        expected_requires_human_approval=False,
        expected_route="information",
        expected_final_workflow_status="completed",
        expected_mutation=None,
        response_forbidden_facts=("completado", "resuelto", "processed", "completed", "resolved"),
    ),
    # --- Additional coverage gap: failed-payment billing issue (1) -------------------
    EndToEndEvalCase(
        case_id="billing-investigation-failed-payment",
        description="Billing issue on an order whose payment already failed, approved by the reviewer.",
        customer_id="cust-003",
        customer_message="Mi pago para order-1007 fallo y no se por que.",
        expected_intent="billing_issue",
        expected_urgency="medium",
        expected_order_resolution_status="selected",
        expected_selected_order_id="order-1007",
        expected_policy_outcome="review_required",
        expected_policy_code="BILLING_REVIEW_REQUIRED",
        expected_requires_human_approval=True,
        expected_route="approval",
        expected_proposed_action=ExpectedProposedAction(
            action_type="investigate_billing", order_id="order-1007", requires_human_approval=True
        ),
        human_decision="approved",
        expected_final_workflow_status="completed",
        expected_mutation=ExpectedMutation(order_id="order-1007", reference_id="billing-investigation-order-1007"),
        response_required_fact_groups=(
            ("order-1007",),
            ("investig", "revis", "review"),
            ("correctamente", "resuelto", "successfully", "completed"),
        ),
    ),
)
