"""Streamlit portfolio demo for the Mercora AI Customer Operations Agent.

Run with:

    streamlit run streamlit_app.py

Renders the initial page without constructing any OpenAI-backed dependency
or making any network call - the workflow only runs once the user submits
a case. Real OpenAI API calls (and their cost) happen only then.

Architecture note (see `customer_ops/demo_runtime.py`): the mutable
workflow runtime (graph, `InMemorySaver`, `InMemoryCustomerActionStore`) is
held ONLY in `st.session_state`, never behind a global resource-caching
decorator or any other shared cache. Each browser session owns its own
runtime, and each new case gets a completely fresh one, so simulated
business-state mutations from one user/case are never visible to another.
"""

from __future__ import annotations

import os
import uuid

import openai
import streamlit as st

from customer_ops.action_inputs import ActionInputError
from customer_ops.classifier import ClassificationError
from customer_ops.demo_runtime import (
    DemoRuntime,
    DemoRuntimeError,
    create_demo_runtime,
    mark_error,
    resume_demo_case,
    start_demo_case,
)
from customer_ops.response_generator import ResponseGenerationError
from tools.customer_data import load_customer_records

st.set_page_config(page_title="Mercora — AI Customer Operations Agent", page_icon="🤖", layout="centered")


# --- OpenAI configuration (secrets -> env, never the reverse; never logged) -----------


def _ensure_openai_config_from_secrets() -> None:
    """Copy OPENAI_API_KEY/OPENAI_MODEL from `st.secrets` into the environment.

    Only fills in a variable that is NOT already set locally (via `.env`),
    and only if Streamlit secrets are configured at all - safe to call on
    every render, including in local development with no `secrets.toml`.
    Never prints, renders, or stores the value anywhere else.
    """
    for name in ("OPENAI_API_KEY", "OPENAI_MODEL"):
        if os.environ.get(name):
            continue
        try:
            value = st.secrets[name]
        except Exception:
            continue
        if value:
            os.environ[name] = value


_ensure_openai_config_from_secrets()


# --- Read-only fixture data (safe to cache - never the mutable runtime) ---------------


@st.cache_data
def _load_customers() -> list[dict]:
    """Load the synthetic customer fixture for the selector, cached.

    Immutable presentation data only - never the mutable graph/store, which
    must remain session-scoped (see module docstring).
    """
    return [c.model_dump(mode="json") for c in load_customer_records()]


_CUSTOMERS = _load_customers()
_CUSTOMER_IDS = [c["customer_id"] for c in _CUSTOMERS]
_CUSTOMER_NAMES = {c["customer_id"]: c["name"] for c in _CUSTOMERS}


def _format_customer(customer_id: str) -> str:
    return f"{customer_id} — {_CUSTOMER_NAMES.get(customer_id, '?')}"


# --- Example scenarios (UI convenience only - never referenced by production logic) ---

_EXAMPLE_SCENARIOS: tuple[tuple[str, str, str], ...] = (
    ("Consulta de estado de pedido", "cust-001", "Hola, ¿dónde está mi pedido order-1001?"),
    ("Cancelación segura (pedido pendiente)", "cust-002", "Please cancel order-1005."),
    (
        "Cambio de dirección con dirección nueva",
        "cust-006",
        "Por favor cambia la direccion de envio de order-1010 a Avenida Central 456, Cordoba, Argentina.",
    ),
    ("Reembolso (requiere aprobación humana)", "cust-001", "I would like a refund for order-1002."),
    ("Problema de facturación (requiere aprobación humana)", "cust-002", "Me cobraron dos veces por order-1004."),
    ("Problema de producto (requiere aprobación humana)", "cust-003", "El producto de order-1006 llego danado."),
    ("Cancelación bloqueada (pedido ya enviado)", "cust-001", "Please cancel order-1001."),
)


def _load_example(customer_id: str, message: str) -> None:
    """Populate the form widgets via session_state, safely, BEFORE they render.

    Bound as an `on_click` callback - callbacks run before the next script
    execution creates the widgets, so this never triggers Streamlit's
    "cannot modify a widget after it has been instantiated" error. Examples
    only populate customer_id + message; they never alter workflow behavior.
    """
    st.session_state["customer_select"] = customer_id
    st.session_state["message_input"] = message


def _start_new_case() -> None:
    """Discard the current DemoRuntime and clear the case-input fields.

    Also bound as an `on_click` callback for the same widget-safety reason.
    A pending approval, if any, is simply abandoned - acceptable because
    all simulated state here is intentionally non-durable.
    """
    st.session_state["demo_runtime"] = None
    st.session_state["message_input"] = ""


# Allowlisted, user-safe messages. Never derived from `str(exc)` - the raw
# exception text (which may embed provider/request details, configuration
# values, or other internals) is never rendered to the UI. Categorization
# is by exception TYPE only, never by inspecting message/secret content.
_MISSING_CONFIG_MESSAGE = "Falta configurar el acceso a la API de OpenAI. Configura OPENAI_API_KEY e intenta nuevamente."
_AI_SERVICE_FAILURE_MESSAGE = "El servicio de IA no pudo procesar este caso. Inicia un caso nuevo e intenta nuevamente."
_RUNTIME_STATE_MESSAGE = "Este caso de demo ya no está en un estado que se pueda continuar. Inicia un caso nuevo."
_GENERIC_FAILURE_MESSAGE = "El caso no se pudo completar de forma segura. Inicia un caso nuevo e intenta nuevamente."


def _safe_error_message(exc: Exception) -> str:
    """Map an exception to a concise, allowlisted, user-safe message.

    Never renders `str(exc)`, a traceback, or any other exception internal.
    Every branch here is an `isinstance` check on the exception's TYPE - see
    CLAUDE.md's "Streamlit demo" section on never leaking implementation,
    provider, or configuration details to the UI.

    - `DemoRuntimeError`: the demo runtime was used out of order (e.g. a
      resume with no pending approval) - not an OpenAI/config problem.
    - `ClassificationError` / `ActionInputError` / `ResponseGenerationError`:
      one of the three OpenAI-backed workflow steps failed at call time -
      these already wrap the underlying `openai.OpenAIError`/technical
      failure (see `customer_ops/classifier.py` et al.), so by the time an
      exception of this type reaches here it is already a domain-level
      "the AI call failed" signal, not a raw provider error.
    - `openai.OpenAIError` reaching here unwrapped can only originate from
      constructing the OpenAI client itself (e.g. no API key configured) -
      every actual API *call* site in this codebase already catches
      `OpenAIError` and re-raises one of the three types above.
    - Anything else: a generic, non-specific failure message.
    """
    if isinstance(exc, DemoRuntimeError):
        return _RUNTIME_STATE_MESSAGE
    if isinstance(exc, (ClassificationError, ActionInputError, ResponseGenerationError)):
        return _AI_SERVICE_FAILURE_MESSAGE
    if isinstance(exc, openai.OpenAIError):
        return _MISSING_CONFIG_MESSAGE
    return _GENERIC_FAILURE_MESSAGE


# --- Header -----------------------------------------------------------------------------

st.title("Mercora — AI Customer Operations Agent")
st.caption(
    "Workflow de operaciones de cliente con estado, construido sobre LangGraph, "
    "con acciones simuladas y aprobación humana."
)
st.info(
    "🧪 Esto es una demo de portafolio. Mercora, los clientes y los pedidos son "
    "ficticios/sintéticos. Ninguna acción mueve dinero real ni contacta sistemas "
    "externos reales.",
    icon="🧪",
)


# --- Sidebar: examples, architecture, limitations ----------------------------------------

with st.sidebar:
    st.header("Ejemplos")
    st.caption("Un clic completa el cliente y el mensaje - no ejecuta el caso automáticamente.")
    for label, customer_id, message in _EXAMPLE_SCENARIOS:
        st.button(label, key=f"example_{label}", on_click=_load_example, args=(customer_id, message))

    st.header("Arquitectura")
    st.markdown(
        "- **LangGraph** orquesta el workflow con estado y enrutamiento condicional.\n"
        "- **OpenAI** clasifica intención/urgencia, extrae direcciones y redacta la "
        "respuesta final - nunca decide política, enrutamiento ni aprobación.\n"
        "- Las acciones sensibles se pausan en un `interrupt()` real de LangGraph "
        "hasta que un humano aprueba o rechaza.\n"
        "- Cada caso corre sobre un `InMemorySaver` y un almacén simulado en "
        "memoria propios de esta sesión."
    )

    st.header("Limitaciones de la sesión")
    st.markdown(
        "- El estado activo del grafo vive en esta sesión de Streamlit, no en un "
        "servidor durable.\n"
        "- `InMemorySaver` y el almacén simulado no son persistentes.\n"
        "- Recargar el navegador o reiniciar el servidor pierde el caso activo.\n"
        "- **Nuevo caso** siempre arranca desde los datos sintéticos originales."
    )


# --- Session-scoped runtime (NEVER a global/cached singleton) ---------------------------

if "demo_runtime" not in st.session_state:
    st.session_state["demo_runtime"] = None

runtime: DemoRuntime | None = st.session_state["demo_runtime"]
is_awaiting_approval = runtime is not None and runtime.phase == "awaiting_approval"


# --- Input area ---------------------------------------------------------------------------

st.subheader("Nuevo caso")

with st.form("case_form", clear_on_submit=False):
    customer_id = st.selectbox(
        "Cliente (sintético)",
        options=_CUSTOMER_IDS,
        format_func=_format_customer,
        key="customer_select",
    )
    message = st.text_area("Mensaje del cliente", key="message_input", height=110)
    submitted = st.form_submit_button(
        "▶️ Ejecutar caso",
        disabled=is_awaiting_approval,
        type="primary",
    )

if is_awaiting_approval:
    st.caption("Hay una aprobación pendiente. Resuélvela abajo o inicia un caso nuevo.")

st.button("🆕 Nuevo caso", on_click=_start_new_case, key="new_case_button")

if submitted and not is_awaiting_approval:
    if not message or not message.strip():
        st.error("Escribe un mensaje del cliente antes de ejecutar el caso.")
    else:
        try:
            new_runtime = create_demo_runtime()
            start_demo_case(
                new_runtime,
                request_id=f"ui-{uuid.uuid4()}",
                customer_id=customer_id,
                customer_message=message.strip(),
            )
        except Exception as exc:  # noqa: BLE001 - shown to the user, never re-raised
            st.error(f"No se pudo ejecutar el caso. {_safe_error_message(exc)}")
        else:
            st.session_state["demo_runtime"] = new_runtime
            st.rerun()


# --- Workflow result: approval card, or final response, or error ------------------------

runtime = st.session_state["demo_runtime"]

if runtime is not None:
    st.divider()
    st.subheader("Resultado del caso")

    if runtime.phase == "error":
        st.error(f"Ocurrió un error durante el workflow: {runtime.error_message}")
        st.caption("Inicia un caso nuevo para volver a intentarlo - no se reintenta automáticamente.")

    elif runtime.phase == "awaiting_approval":
        payload = runtime.interrupt_payload or {}
        with st.container(border=True):
            st.markdown("### 🔒 Se requiere aprobación humana")
            st.caption("Acción simulada - ningún dinero ni sistema real se ve afectado.")
            if payload.get("action_type"):
                st.write(f"**Acción:** {payload['action_type']}")
            if payload.get("order_id"):
                st.write(f"**Pedido:** {payload['order_id']}")
            if payload.get("amount") is not None:
                st.write(f"**Monto:** {payload['amount']}")
            if payload.get("currency"):
                st.write(f"**Moneda:** {payload['currency']}")
            if payload.get("message"):
                st.write(f"**Mensaje:** {payload['message']}")

            col_approve, col_reject = st.columns(2)
            approve_clicked = col_approve.button("✅ Aprobar", type="primary", key="approve_button")
            reject_clicked = col_reject.button("❌ Rechazar", key="reject_button")

        if approve_clicked or reject_clicked:
            decision = "approved" if approve_clicked else "rejected"
            try:
                resume_demo_case(runtime, decision=decision)
            except Exception as exc:  # noqa: BLE001 - shown to the user, never re-raised
                mark_error(runtime, _safe_error_message(exc))
            st.session_state["demo_runtime"] = runtime
            st.rerun()

    elif runtime.phase == "completed":
        final_response = (runtime.latest_state or {}).get("final_response")
        if final_response:
            st.success(final_response)
        else:
            st.warning("El caso terminó sin una respuesta final visible.")

    # --- Workflow details ---------------------------------------------------------------

    state = runtime.latest_state or {}
    with st.expander("Detalles del workflow"):
        policy_assessment = state.get("policy_assessment") or {}
        proposed_action = state.get("proposed_action") or {}
        order_resolution = state.get("order_resolution") or {}
        rows = [
            ("Intención", state.get("intent")),
            ("Urgencia", state.get("urgency")),
            ("Pedido seleccionado", state.get("selected_order_id")),
            ("Resolución de pedido", order_resolution.get("status")),
            ("Resultado de política", policy_assessment.get("outcome")),
            ("Motivo de política", policy_assessment.get("reason")),
            ("Ruta", state.get("route")),
            ("Acción propuesta", proposed_action.get("action_type")),
            ("Decisión humana", state.get("human_decision")),
            ("Estado del workflow", state.get("workflow_status")),
        ]
        visible_rows = [{"Campo": label, "Valor": str(value)} for label, value in rows if value]
        if visible_rows:
            st.table(visible_rows)
        else:
            st.caption("Aún no hay detalles disponibles.")

    # --- Audit trail ----------------------------------------------------------------------

    with st.expander("Registro de auditoría"):
        audit_log = state.get("audit_log") or []
        if audit_log:
            st.table(
                [
                    {"Paso": event.get("step", ""), "Estado": event.get("status", ""), "Mensaje": event.get("message", "")}
                    for event in audit_log
                ]
            )
        else:
            st.caption("Sin eventos todavía.")
