"""UI-agnostic orchestration helpers for the Streamlit human-in-the-loop demo.

`DemoRuntime` bundles one browser session's own compiled graph, its
combined `InMemoryCustomerActionStore` (used for both reads and mutations,
so a single case never drifts between two independently-loaded copies),
its `InMemorySaver` (held inside the compiled graph), and its `thread_id`.
Every `create_demo_runtime()` call builds a completely fresh set of these -
nothing here is a module-level singleton or a `@st.cache_resource`-style
global, so unrelated browser sessions (and unrelated cases within one
session) never share or leak mutable state.

This module never imports Streamlit. It reuses the exact same real
LangGraph `interrupt()`/`Command(resume=...)` semantics already proven by
`customer_ops/graph.py` and exercised by `evals/e2e_runner.py` - the
interrupt-inspection logic below is reimplemented locally (not imported
from `evals/`) so the production demo never depends on evaluation code.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Literal

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command

from customer_ops.action_inputs import ActionInputExtractor
from customer_ops.classifier import RequestClassifier
from customer_ops.graph import build_customer_ops_graph
from customer_ops.response_generator import CustomerResponseGenerator
from tools.action_store import InMemoryCustomerActionStore

DemoPhase = Literal["idle", "awaiting_approval", "completed", "error"]


class DemoRuntimeError(Exception):
    """Raised when a demo-runtime operation is used out of order.

    E.g. resuming when no approval is currently pending. Never silently
    ignored - the caller (the Streamlit layer) must surface this rather
    than guess what the user meant.
    """


@dataclass
class DemoRuntime:
    """One browser session's own graph/store/thread - never shared.

    `graph` and `store` are live runtime objects, not `CustomerOpsState` -
    they are never expected to be JSON/Pydantic serializable. `thread_id`
    is generated once here and reused for every invoke/resume on this
    runtime; the graph itself never generates or derives it.
    """

    graph: CompiledStateGraph
    store: InMemoryCustomerActionStore
    thread_id: str
    phase: DemoPhase = "idle"
    latest_state: dict[str, Any] | None = None
    interrupt_payload: dict[str, Any] | None = None
    error_message: str | None = None

    @property
    def config(self) -> dict[str, Any]:
        return {"configurable": {"thread_id": self.thread_id}}


def create_demo_runtime(
    *,
    classifier: RequestClassifier | None = None,
    action_input_extractor: ActionInputExtractor | None = None,
    response_generator: CustomerResponseGenerator | None = None,
) -> DemoRuntime:
    """Build a brand-new, fully isolated demo runtime.

    Always constructs exactly one fresh `InMemoryCustomerActionStore.from_json()`
    and passes it as BOTH the read-only store and the mutating action store,
    a fresh `InMemorySaver()`, and a freshly compiled graph - so a case
    started from this runtime can never see a mutation from any other
    runtime, and reads/writes within it never diverge. `thread_id` is a
    fresh `demo-<uuid4>` (runtime correlation only, not deterministic
    business data - `uuid4()` is fine here).

    Passing fake `classifier`/`action_input_extractor`/`response_generator`
    keeps this fully offline for tests, exactly like
    `build_customer_ops_graph`'s own dependency injection; omitting them
    uses the real OpenAI-backed implementations.
    """
    store = InMemoryCustomerActionStore.from_json()
    graph = build_customer_ops_graph(
        classifier=classifier,
        store=store,
        action_input_extractor=action_input_extractor,
        action_store=store,
        checkpointer=InMemorySaver(),
        response_generator=response_generator,
    )
    return DemoRuntime(graph=graph, store=store, thread_id=f"demo-{uuid.uuid4()}")


def extract_interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    """Return the public interrupt payload from a graph result, or `None`.

    Mirrors the `"__interrupt__" in result` / `result["__interrupt__"][0].value`
    semantics already used by `evals/e2e_runner.py`, reimplemented locally
    so the production demo never imports evaluation code. The returned
    value is exactly the `ApprovalRequest.model_dump(mode="json")` payload
    `human_approval_node` passed to `interrupt()` - never the full
    interrupted `CustomerOpsState`.
    """
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value


def start_demo_case(
    runtime: DemoRuntime, *, request_id: str, customer_id: str, customer_message: str
) -> DemoRuntime:
    """Invoke the graph once for a brand-new case on `runtime`'s own thread_id.

    Mutates and returns `runtime` in place. `request_id` is generated by
    the caller (the UI/application layer) - this function never invents
    one. If the graph interrupts, stores ONLY the public interrupt payload
    (never the raw interrupted state) and sets `phase="awaiting_approval"` -
    no final response is generated and no sensitive mutation has occurred.
    If the graph completes without interrupting, sets `phase="completed"`.
    """
    result = runtime.graph.invoke(
        {"request_id": request_id, "customer_id": customer_id, "customer_message": customer_message},
        config=runtime.config,
    )
    runtime.latest_state = result
    payload = extract_interrupt_payload(result)
    if payload is not None:
        runtime.interrupt_payload = payload
        runtime.phase = "awaiting_approval"
    else:
        runtime.interrupt_payload = None
        runtime.phase = "completed"
    runtime.error_message = None
    return runtime


def resume_demo_case(runtime: DemoRuntime, *, decision: Literal["approved", "rejected"]) -> DemoRuntime:
    """Resume a pending approval on `runtime`'s SAME thread_id.

    Raises `DemoRuntimeError` if no approval is currently pending on this
    runtime - never silently no-ops and never invents a decision. Resumes
    via the real `Command(resume={"decision": decision})` mechanism on the
    exact same `thread_id` the interrupt was raised under; the execution
    happens exactly once, inside the graph itself, never emulated here.
    Clears `interrupt_payload` and sets `phase="completed"` afterward.
    """
    if runtime.phase != "awaiting_approval":
        raise DemoRuntimeError(f"Cannot resume: no approval is pending (phase={runtime.phase!r}).")

    result = runtime.graph.invoke(Command(resume={"decision": decision}), config=runtime.config)
    runtime.latest_state = result
    runtime.interrupt_payload = None
    runtime.phase = "completed"
    runtime.error_message = None
    return runtime


def mark_error(runtime: DemoRuntime, message: str) -> DemoRuntime:
    """Record a concise, user-safe error on `runtime` and set `phase="error"`.

    Used by the Streamlit layer around invoke/resume calls that may raise
    (e.g. a configuration error, a technical workflow exception). Never
    retries or replays an action on this runtime - the caller is expected
    to offer "New case" instead.
    """
    runtime.phase = "error"
    runtime.error_message = message
    return runtime
