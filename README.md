# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution, policy evaluation, conditional routing, structured action
proposals, simulated safe-action execution, human-in-the-loop approval with
LangGraph checkpointing, final customer-facing response generation, a
curated end-to-end evaluation benchmark, and a Streamlit human-in-the-loop
demo (Iteration 10).** This is a portfolio project and is not production
software.

> Mercora is a fictional e-commerce company invented for this project. All
> customers, orders, policies, payments, addresses, and tickets referenced
> anywhere in this repository are synthetic and do not represent any real
> business or individual. All data in `data/` is fabricated for this project.

## Long-term goal

Build a portfolio-grade customer operations agent that receives a customer
request and orchestrates a multi-step workflow able to:

- classify the request (intent + urgency)
- retrieve customer information
- retrieve order information
- evaluate company policies
- decide whether the case can be answered directly
- propose operational actions
- execute safe, simulated actions
- pause for human approval before sensitive actions
- escalate cases when necessary
- generate a final customer-facing response
- expose a structured audit trail of the workflow

The project is intentionally distinct from an SQL data agent or a RAG
knowledge assistant. Its purpose is to demonstrate: LangGraph, stateful
workflows, conditional routing, tool usage, structured LLM outputs,
human-in-the-loop, checkpointing, safe action execution, evaluation, and
observability.

## Planned architecture

- **LangGraph** (`StateGraph`) orchestrates workflow state and conditional
  routing directly — no LangChain agents, `create_agent`, CrewAI, AutoGen, or
  n8n. Conditional edges (`add_conditional_edges`) dispatch on structured
  state, not an if/else buried inside a single node.
- The **OpenAI SDK** is called directly inside specific graph nodes that need
  LLM reasoning, using the Responses API's Structured Outputs mechanism
  rather than free-form text parsing; LangGraph orchestrates, it does not
  hide business logic. The LLM's responsibilities are deliberately narrow:
  classify intent/urgency, extract an explicit replacement address when
  needed, and phrase the final customer response - it never decides policy,
  routing, order selection, approval, or execution results; those are
  computed deterministically first and only handed to the model as already-
  validated facts to communicate.
- **Tools** (`tools/`) are narrow, independently testable interfaces used by
  graph nodes - the read-only `CustomerOperationsStore` and the mutating
  `CustomerActionStore`, kept as separate interfaces even where one class
  implements both.
- Sensitive/high-impact simulated actions require a **human-in-the-loop**
  approval step before execution: a real LangGraph `interrupt()` pauses the
  graph, and execution resumes only via `Command(resume=...)` carrying an
  explicit approved/rejected decision.
- The workflow state is checkpoint-friendly and produces a **structured audit
  trail** of observable events (not model reasoning).

## Current implementation (Iteration 8: Customer Response Generation)

Every completed (or paused) case now ends with a safe, customer-facing
response, grounded only in validated workflow state - completing the full
pipeline:

**Policy** (what is allowed) **-> Routing** (which workflow path follows)
**-> Proposed Action** (what operation is intended) **-> Action Input**
(what validated parameters it needs) **-> `interrupt()`** (sensitive actions
only - pause and wait for a human) **-> Executor** (calls exactly one store
method, only after approval) **-> Simulated Store** (the in-memory mutation)
**-> Final Response** (phrase the outcome - never decide it).

This separation is the project's core architecture point:

- **LLM responsibilities**: classify intent/urgency; extract an explicit
  replacement address only when `change_address` needs one; phrase the
  final customer-facing response (`customer_ops/classifier.py`,
  `customer_ops/action_inputs.py`, `customer_ops/response_generator.py`).
- **Deterministic responsibilities**: customer/order facts, order
  resolution, policy, routing, action-proposal mapping, action execution,
  and approval enforcement (everything else in `customer_ops/`). The model
  never decides any of these - they already exist as structured state by
  the time the response generator runs.

Implemented now:

- **`customer_ops/response_generator.py`**: `CustomerResponse` (`message`
  only - no reasoning, rationale, confidence, or citations) and
  `ResponseContext`, a narrow, JSON-friendly, PII-free subset of state
  (`customer_message`, `intent`, `route`, `workflow_status`,
  `selected_order_id`, `order_summary` - order ID/status/tracking number
  only, `policy_outcome`/`policy_code`/`policy_reason`, `proposed_action`,
  `action_input`, `human_decision`, `action_result`). Never the full
  `customer_context`/`order_context`, never `audit_log`, never checkpointer
  data. `build_response_context(...)` takes only already-extracted
  primitives (e.g. a single already-selected order, never the full order
  list), so unrelated data structurally cannot leak in.
- **`OpenAICustomerResponseGenerator`**: one OpenAI Responses API
  Structured Outputs call per response, mirroring
  `OpenAIRequestClassifier`/`OpenAIActionInputExtractor`. Instructed to use
  ONLY the supplied context - never invent dates, amounts, addresses,
  shipping estimates, compensation, policies, or support promises; never
  claim an action succeeded unless `action_result.success` is true; never
  claim a sensitive action happened unless `human_decision == "approved"`;
  never mention LangGraph/OpenAI/internal architecture or expose raw
  policy codes.
- **Language**: Spanish-first; replies in English only when the customer's
  own message is clearly English. Order IDs and action names are never
  translated. No language selector.
- **`final_response` node**: reached by every terminal branch
  (clarification, information, blocked, safe execution, approved execution,
  rejected). Builds the `ResponseContext`, calls
  `generator.generate(...)` exactly once, stores only the resulting message
  text in `final_response` (never the full `CustomerResponse` object), sets
  `workflow_status = "completed"`, and appends one generic audit event -
  the response text itself is never duplicated into `audit_log`.
- **HITL ordering preserved**: an approval-required case still pauses at
  `interrupt()` before ever reaching `final_response` - proven by tests
  showing the response generator is called zero times on the interrupted
  pass and exactly once total after resume, never twice.
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> resolve_order -> evaluate_policy -> (conditional) -> {clarification, information, propose_action -> prepare_action_input -> (conditional) -> {execute_safe_action, clarification}, prepare_approval -> prepare_approval_input -> human_approval -- interrupt() -- (resume) --> {execute_approved_action, approval_rejected}, blocked} -> final_response -> END`,
  built via `build_customer_ops_graph(classifier=None, store=None, action_input_extractor=None, action_store=None, checkpointer=None, response_generator=None)`.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts, models, the data store, the classifier,
  order resolution, policy evaluation, routing, action proposal, action
  inputs, the action store, the action executor, approval contracts, the
  response generator, and full graph branch/HITL/response coverage - all
  running with no network access and no API key.

**Important limitations, stated accurately**: `InMemorySaver` checkpoint
state survives separate invoke/resume calls only while this Python process
and this saver instance stay alive - it provides **no durable persistence
across a process restart**, and it is a different concern entirely from the
simulated business-data mutations in `tools/action_store.py` (also
process-local). No real money, orders, or customer systems are ever
touched. The JSON fixtures remain example source data, not persistent
business storage.

Planned later (not implemented yet):

- Deployment.
- Durable checkpoint/business-data persistence (a future improvement beyond
  `InMemorySaver`).

## End-to-End Evaluation

`evals/` contains a curated, rule-based end-to-end evaluation benchmark that
measures the workflow implemented above - it does not add any new business
logic, routes, or action types, and production behavior was not changed to
make it pass.

- **`evals/e2e_cases.py`**: `EndToEndEvalCase`, a declarative Pydantic
  contract (no callback functions) with every `expected_*` field optional,
  and `E2E_CASES`, 20 curated cases grounded strictly in the real
  `data/customers.json`/`data/orders.json` fixtures (real customer IDs, real
  order IDs, real order/payment statuses - never assumed). Coverage
  includes every intent, both Spanish and English, every routing branch
  (information, clarification, action, approval, blocked), a zero-order
  customer, an unknown/non-owned order reference, an already-refunded
  order, missing-address clarification, safe mutations (cancel, address
  change), sensitive mutations gated by human approval (refund, billing
  investigation, product investigation) with both an approved and a
  rejected case each, and an unsupported ("other") request.
- **`evals/e2e_checks.py`**: ten transparent, rule-based check functions
  (intent, urgency, order resolution, policy, route, proposed action,
  approval behavior, mutation, final state, response). Response checks use
  only deterministic normalized-text comparisons
  (`normalize_text`/`contains_required_fact_groups`/
  `contains_forbidden_facts`) - **this benchmark never uses an LLM as a
  judge**. These checks are transparent but cannot catch every possible
  hallucination; they only verify the specific facts a case declares.
  Every check reports both `applicable` (whether this case exercises that
  dimension) and `passed` (always the genuine correctness verdict for this
  case) - per-metric accuracy is applicability-aware (cases that don't
  exercise a dimension are excluded from that metric's denominator, never
  counted as a pass), while a case's overall pass/fail always reflects
  every check, applicable or not.
- **`evals/e2e_runner.py`**: runs each case against a **completely fresh**
  graph, `InMemorySaver`, and `InMemoryCustomerActionStore` - no mutation or
  state from one case is ever visible to another, and case order never
  affects results. Approval-designed cases genuinely trigger a real
  LangGraph `interrupt()` and resume via `Command(resume={"decision":
  ...})` on the same deterministic `eval-<case_id>` thread ID - the human
  approval gate is never bypassed. A per-case failure (a mismatch against
  expectations, or an unexpected exception) is captured and does not abort
  the rest of the benchmark run.
- **`evals/run_e2e_evals.py`**: the CLI entry point,
  `python -m evals.run_e2e_evals`. **This command makes real OpenAI API
  calls** (the real classifier, address extractor, and response generator)
  for all 20 cases and **consumes real API usage/cost**. It is never run by
  pytest and must be run manually, after reviewing the code. Because three
  of the ten workflow components are model-backed, results can vary
  slightly between runs; every other component (policy, routing, order
  resolution, action execution, approval enforcement) is fully
  deterministic and does not vary. Prints a `[PASS]`/`[FAIL]` line per case
  plus a summary with ten applicability-aware accuracy metrics and an
  overall pass rate, ending with a disclaimer that these metrics describe
  only this curated benchmark, not general model accuracy or production
  reliability.
- **Latest reviewed real-benchmark result** (`python -m evals.run_e2e_evals`,
  run manually and reviewed, not run by pytest or by this iteration):
  20 cases, 20/20 passed, all structured workflow metrics at 100%, response
  fact-check at 100%. **These results apply only to this curated benchmark
  and are not claims of general model accuracy or production reliability.**
  Results can vary across runs, since three of the ten workflow components
  are model-backed - see `evals/run_e2e_evals.py` for the same disclaimer
  printed after every run.

`tests/test_e2e_evals.py` tests this framework's own logic - fully
offline, using synthetic cases and fake classifier/extractor/response-
generator dependencies (never the real 20-case benchmark, never a real
OpenAI call). See [Tests](#tests) below for how to run the full pytest
suite, which remains 100% offline.

## Streamlit Demo

`streamlit_app.py` is a portfolio-facing, Spanish-first UI around the exact
workflow described above - it adds no new business logic, and does not
change policy/routing/action-execution behavior.

```powershell
.\.venv\Scripts\streamlit run streamlit_app.py
```

Loading the page never calls OpenAI or makes a network call - a real
OpenAI-backed run only starts when a case is actually submitted, and that
is where real API usage/cost occurs. `OPENAI_API_KEY` (and optionally
`OPENAI_MODEL`) can come from a local `.env`, or - for a future deployment -
from `st.secrets`; either way, the value is copied into the process
environment only, never logged, rendered, or stored in graph state.

What the demo exercises, using the real production workflow:

- a synthetic-customer selector and a free-text customer message, submitted
  through a form (`▶️ Ejecutar caso`) - nothing invokes the workflow on
  every keystroke;
- safe actions (cancel, address change) executing directly and showing the
  grounded `final_response` exactly as the graph produced it - the UI never
  regenerates or edits it;
- sensitive actions (refund, billing investigation, product investigation)
  pausing at a real LangGraph `interrupt()`, rendering only the public
  `ApprovalRequest` fields (action, order, amount, currency, message - never
  customer email/address or full state), with **Aprobar**/**Rechazar**
  buttons that resume the SAME `thread_id` via `Command(resume=...)` -
  execution happens exactly once, only after an explicit "approved"
  decision;
- a "Detalles del workflow" expander (intent, urgency, selected order,
  order resolution, policy outcome/reason, route, proposed action, human
  decision, workflow status) and a "Registro de auditoría" expander
  (the existing PII-free `audit_log`, rendered as-is);
- sidebar example scenarios (order status, safe cancellation, address
  change, refund/billing/product approval, blocked cancellation) that only
  populate the customer/message fields - they never alter workflow
  behavior and are not referenced anywhere in production logic;
- a **Nuevo caso** control that discards the active runtime and starts the
  next case from a completely fresh simulated store/checkpointer.

Session and persistence model (see `customer_ops/demo_runtime.py`): each
browser session owns its own graph, `InMemorySaver`, and
`InMemoryCustomerActionStore`, held only in `st.session_state` - never
behind a global cache, so unrelated sessions can never see or mutate each
other's simulated state. Each new case gets an equally fresh runtime, so
one demo case's mutations never affect the next. None of this is durable:
a browser reload or a server restart loses the active case and every
simulated mutation, by design - "New case" intentionally starts over from
the original synthetic fixtures, not from a saved snapshot.

## Tech stack

- Python 3.13+
- [LangGraph](https://github.com/langchain-ai/langgraph) `>=1.1,<2.0` for
  orchestration, including `interrupt()`/`Command(resume=...)` for
  human-in-the-loop and `InMemorySaver` (from `langgraph-checkpoint`,
  already a LangGraph dependency) for demo/development checkpointing
- [OpenAI SDK](https://github.com/openai/openai-python) `>=3.0,<4.0`, used via
  the Responses API structured-output path (`responses.parse`)
- [Pydantic](https://github.com/pydantic/pydantic) `>=2.0,<3.0`
- [python-dotenv](https://github.com/theskumar/python-dotenv) for local
  environment configuration
- [Streamlit](https://github.com/streamlit/streamlit) `>=1.63,<2.0` for the
  portfolio demo UI (`streamlit_app.py`), including
  `streamlit.testing.v1.AppTest` for offline UI tests
- [pytest](https://github.com/pytest-dev/pytest) for testing

## Project structure

```
ai-customer-operations-agent/
│
├── customer_ops/
│   ├── __init__.py
│   ├── state.py            # CustomerOpsState, AuditEvent, OrderContext, controlled vocabularies
│   ├── models.py           # CustomerRecord, OrderItem, OrderRecord (Pydantic data contracts)
│   ├── classifier.py       # ClassificationDecision, RequestClassifier, OpenAIRequestClassifier
│   ├── order_resolution.py # OrderResolution, resolve_order() - deterministic order selection
│   ├── policies.py         # PolicyAssessment, evaluate_policy() - deterministic Mercora rules
│   ├── routing.py          # determine_case_route() - deterministic branch selection
│   ├── action_proposal.py  # ProposedAction, propose_action() - structured action intent
│   ├── action_inputs.py    # ActionInputResult, prepare_action_input(), address extraction
│   ├── action_executor.py  # ActionResult, execute_action() - one simulated mutation call
│   ├── approval.py         # ApprovalRequest, HumanApprovalResponse - HITL contracts
│   ├── response_generator.py # CustomerResponse, ResponseContext, build_response_context()
│   ├── graph.py            # all graph nodes, build_customer_ops_graph()
│   └── demo_runtime.py     # DemoRuntime, create/start/resume_demo_case() - UI-agnostic
│
├── tools/
│   ├── __init__.py
│   ├── customer_data.py    # CustomerOperationsStore, JsonCustomerOperationsStore (read-only)
│   └── action_store.py     # CustomerActionStore, InMemoryCustomerActionStore (simulated mutation)
│
├── data/
│   ├── customers.json      # synthetic Mercora customers
│   └── orders.json         # synthetic Mercora orders
│
├── evals/
│   ├── __init__.py
│   ├── e2e_cases.py     # EndToEndEvalCase, E2E_CASES (the 20-case benchmark)
│   ├── e2e_checks.py    # rule-based ComponentCheckResult check functions
│   ├── e2e_runner.py    # run_single_case()/run_e2e_evals() - fresh graph per case
│   └── run_e2e_evals.py # CLI: python -m evals.run_e2e_evals (real OpenAI calls)
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_state.py
│   ├── test_models.py
│   ├── test_customer_data.py
│   ├── test_classifier.py
│   ├── test_order_resolution.py
│   ├── test_policies.py
│   ├── test_routing.py
│   ├── test_action_proposal.py
│   ├── test_action_inputs.py
│   ├── test_action_store.py
│   ├── test_action_executor.py
│   ├── test_approval.py
│   ├── test_response_generator.py
│   ├── test_graph.py
│   ├── test_e2e_evals.py
│   ├── test_demo_runtime.py
│   └── test_streamlit_app.py
│
├── streamlit_app.py      # Streamlit portfolio demo UI (see below)
├── app.py                # minimal placeholder entry point
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
└── .gitignore
```

## Setup

Requires Python 3.13+. Run from PowerShell in the project root.

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` to run the graph for
real (i.e. invoke it without injecting fakes). No key is required to run the
test suite — tests always inject a fake `RequestClassifier`, a fake/in-memory
`CustomerOperationsStore`, a fake `ActionInputExtractor`, and a fake
`CustomerResponseGenerator`, use a fresh `InMemorySaver` per test graph, and
make no network calls:

```powershell
Copy-Item .env.example .env
```

## Running the placeholder entry point

```powershell
.\.venv\Scripts\python app.py
```

## Tests

```powershell
.\.venv\Scripts\python -m pytest -q
```

This always runs fully offline - no `OPENAI_API_KEY` needed, no network
calls made, including for `tests/test_e2e_evals.py`, `tests/test_demo_runtime.py`,
and `tests/test_streamlit_app.py` (the last uses
`streamlit.testing.v1.AppTest` to render `streamlit_app.py` and exercise its
UI wiring without ever submitting a case, so the real OpenAI-backed path is
never triggered by pytest).

## Running the real end-to-end benchmark (optional, costs real API usage)

```powershell
.\.venv\Scripts\python -m evals.run_e2e_evals
```

This is separate from `pytest` and is never run automatically. It requires
a valid `OPENAI_API_KEY` in `.env` and makes real OpenAI API calls for all
20 cases in the benchmark. See [End-to-End Evaluation](#end-to-end-evaluation)
above for what it checks and for the latest reviewed result.
