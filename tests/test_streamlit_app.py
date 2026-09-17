"""Tests for the Streamlit portfolio demo's initial render and UI wiring.

Fully offline. Most tests here never click the case-submission button, so
the real OpenAI-backed `create_demo_runtime()` path is never exercised -
full workflow behavior (including HITL interrupt/resume) is covered
separately and fully offline in `tests/test_demo_runtime.py`. The
error-message tests below DO click submit, but with
`customer_ops.demo_runtime.create_demo_runtime` monkeypatched to raise
immediately (never constructing a real OpenAI client), so they stay fully
offline while still exercising the real rendered error path end to end.

`streamlit_app.py` itself is never imported directly as a plain Python
module in this file (only ever driven through `AppTest`): a direct
`import streamlit_app` executes its module body outside of Streamlit's
script-runner sandbox, which leaves internal widget/form tracking state
corrupted for every subsequent `AppTest` run in the same pytest process
(reproduced while developing this file - it makes later, unrelated
`AppTest` runs fail with `StreamlitInvalidFormCallbackError`). Monkeypatching
`customer_ops.demo_runtime.create_demo_runtime` (an ordinary package
import, not a script exec) has none of that risk, since `AppTest` re-reads
that name fresh from `customer_ops.demo_runtime` on every sandboxed run.

Note on network isolation: the shared `no_network` fixture (which patches
`socket.socket.connect` unconditionally) is NOT used here. On Windows,
`streamlit.testing.v1.AppTest` spins up an asyncio `ProactorEventLoop`,
which opens a purely-local loopback `socket.socketpair()` for its own
internal self-pipe - that call also goes through `socket.socket.connect`
and would be a false positive under the blanket fixture. Instead, the
"no OpenAI/network call" guarantee is verified precisely by patching
`openai.OpenAI.__init__` itself to raise if constructed.
"""

from pathlib import Path

import openai
import pytest
from streamlit.testing.v1 import AppTest

import customer_ops.classifier as classifier_module
import customer_ops.demo_runtime as demo_runtime_module


_APP_PATH = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


def _run_app() -> AppTest:
    at = AppTest.from_file(_APP_PATH)
    at.run(timeout=30)
    return at


def _submit_case(at: AppTest, *, customer_id: str = "cust-001", message: str = "hola") -> AppTest:
    at.selectbox(key="customer_select").select(customer_id).run(timeout=30)
    at.text_area(key="message_input").set_value(message).run(timeout=30)
    submit = next(b for b in at.button if "case_form" in b.key)
    submit.click().run(timeout=30)
    return at


# --- Initial render -----------------------------------------------------------------


def test_app_renders_without_exception():
    at = _run_app()
    assert not at.exception


def test_title_is_present():
    at = _run_app()
    titles = [t.value for t in at.title]
    assert any("Mercora" in title for title in titles)


def test_synthetic_simulation_notice_is_present():
    at = _run_app()
    info_text = " ".join(i.value for i in at.info)
    assert "sintéticos" in info_text or "sintetico" in info_text.lower()
    assert "ficticio" in info_text.lower() or "ficticios" in info_text.lower()


def test_customer_selector_exists():
    at = _run_app()
    assert at.selectbox(key="customer_select") is not None


def test_message_input_exists():
    at = _run_app()
    assert at.text_area(key="message_input") is not None


def test_run_case_submit_control_exists():
    at = _run_app()
    submit_buttons = [b for b in at.button if "case_form" in b.key]
    assert len(submit_buttons) == 1


def test_new_case_reset_control_is_present():
    at = _run_app()
    assert at.button(key="new_case_button") is not None


def test_architecture_and_persistence_disclaimer_is_visible():
    at = _run_app()
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "LangGraph" in sidebar_text
    assert "InMemorySaver" in sidebar_text
    assert "persistente" in sidebar_text or "no son persistentes" in sidebar_text


# --- Offline guarantees ---------------------------------------------------------------


def test_initial_render_does_not_require_openai_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    at = _run_app()
    assert not at.exception


def test_no_openai_call_occurs_merely_by_rendering_the_page(monkeypatch):
    def _fail_if_constructed(*args, **kwargs):
        raise AssertionError("OpenAI client must not be constructed during initial render.")

    monkeypatch.setattr("openai.OpenAI.__init__", _fail_if_constructed)
    at = _run_app()
    assert not at.exception


# --- Example scenarios (offline UI wiring only - no case execution) -------------------


def test_example_scenario_buttons_populate_form_without_running_a_case():
    at = _run_app()
    example_buttons = [b for b in at.button if b.key.startswith("example_")]
    assert len(example_buttons) >= 5  # order status, safe action, address change, approvals, blocked

    target = next(b for b in example_buttons if "Reembolso" in b.key)
    target.click().run(timeout=30)

    assert not at.exception
    assert at.selectbox(key="customer_select").value == "cust-001"
    assert "order-1002" in at.text_area(key="message_input").value
    # Loading an example must never itself create/advance a workflow runtime.
    assert at.session_state.get("demo_runtime") is None


def test_new_case_button_clears_message_input():
    at = _run_app()
    example_buttons = [b for b in at.button if b.key.startswith("example_")]
    example_buttons[0].click().run(timeout=30)
    assert at.text_area(key="message_input").value

    at.button(key="new_case_button").click().run(timeout=30)
    assert at.text_area(key="message_input").value == ""
    assert at.session_state.get("demo_runtime") is None


# --- Allowlisted, user-safe error messages (never raw exception text) ------------------


def test_missing_api_configuration_shows_a_useful_safe_message(monkeypatch):
    def _raise(**kwargs):
        raise openai.OpenAIError("The api_key client option must be set - env FOO=bar")

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("OPENAI_API_KEY" in e for e in errors)
    assert not any("api_key client option" in e for e in errors)
    assert not any("FOO=bar" in e for e in errors)


def test_ai_service_failure_shows_a_generic_ai_service_message(monkeypatch):
    # `customer_ops.classifier.ClassificationError` is looked up fresh here
    # (via the module, not a name bound at collection time) because
    # `tests/test_classifier.py` reloads that module elsewhere in the suite,
    # which replaces the class object - a stale bound reference would no
    # longer match `isinstance` checks against the reloaded class.
    def _raise(**kwargs):
        raise classifier_module.ClassificationError(
            "Request classification failed: connection reset by peer at 10.0.0.5:443"
        )

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("no pudo procesar" in e for e in errors)
    assert not any("connection reset" in e for e in errors)
    assert not any("10.0.0.5" in e for e in errors)


def test_generic_exception_shows_a_generic_safe_message(monkeypatch):
    def _raise(**kwargs):
        raise RuntimeError("DISTINCTIVE_INTERNAL_DETAIL_should_never_reach_the_ui")

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    assert not at.exception
    errors = [e.value for e in at.error]
    assert any("no se pudo completar" in e for e in errors)
    assert not any("DISTINCTIVE_INTERNAL_DETAIL_should_never_reach_the_ui" in e for e in errors)


def test_raw_exception_message_is_never_surfaced(monkeypatch):
    marker = "RAW_EXCEPTION_TEXT_MARKER_9f3a"

    def _raise(**kwargs):
        raise RuntimeError(marker)

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    page_text = " ".join(e.value for e in at.error) + " ".join(w.value for w in at.warning)
    assert marker not in page_text


def test_secret_looking_exception_content_is_never_rendered(monkeypatch):
    fake_secret = "sk-FAKESECRETVALUE1234567890abcdefABCDEF"

    def _raise(**kwargs):
        raise RuntimeError(f"OpenAI request failed with Authorization: Bearer {fake_secret}")

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    page_text = " ".join(e.value for e in at.error) + " ".join(w.value for w in at.warning)
    assert fake_secret not in page_text
    assert "Bearer" not in page_text


def test_construction_failure_does_not_create_a_runtime(monkeypatch):
    """Preserved behavior: a failure before a runtime exists leaves session_state empty."""

    def _raise(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(demo_runtime_module, "create_demo_runtime", _raise)
    at = _submit_case(_run_app())

    assert at.session_state.get("demo_runtime") is None


# --- No global mutable runtime cache (regression check) --------------------------------


def test_streamlit_app_does_not_cache_the_mutable_runtime_globally():
    source = Path(_APP_PATH).read_text(encoding="utf-8")
    assert "st.cache_resource" not in source
    assert "@cache_resource" not in source
