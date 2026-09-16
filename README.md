# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution, policy evaluation, conditional routing, structured action
proposals, simulated safe-action execution, and human-in-the-loop approval
with LangGraph checkpointing (Iteration 7).** This is a portfolio project
and is not production software.

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
  hide business logic.
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

## Current implementation (Iteration 7: Human-in-the-Loop Approval and Checkpointing)

The full pipeline from a customer message to a simulated operational change
now exists, including a genuine pause-for-human-approval step:

**Policy** (what is allowed) **-> Routing** (which workflow path follows)
**-> Proposed Action** (what operation is intended) **-> Action Input**
(what validated parameters it needs) **-> `interrupt()`** (sensitive actions
only - pause and wait for a human) **-> Executor** (calls exactly one store
method, only after approval) **-> Simulated Store** (the in-memory mutation).

- **LLM**: intent + urgency interpretation, and - only for `change_address`
  - structured address extraction (`customer_ops/classifier.py`,
  `customer_ops/action_inputs.py`).
- **Deterministic data store**: customer/order facts (`tools/customer_data.py`).
- **Deterministic resolver / policy engine / router**: order selection,
  eligibility, and branch selection (`customer_ops/order_resolution.py`,
  `customer_ops/policies.py`, `customer_ops/routing.py`).
- **Structured action proposals and inputs**: what a future action would do
  and what validated parameters it needs, never invented
  (`customer_ops/action_proposal.py`, `customer_ops/action_inputs.py`).
- **Human-in-the-loop approval**: a real LangGraph `interrupt()` that pauses
  the graph and exposes a minimal, PII-free `ApprovalRequest`; the human
  decision comes only from external `Command(resume=...)` input - never
  from the LLM, policy, intent, or urgency (`customer_ops/approval.py`).
- **Simulated execution**: exactly one mutation call against an in-memory
  operational store - for safe actions immediately, for sensitive actions
  only after an explicit "approved" decision
  (`customer_ops/action_executor.py`, `tools/action_store.py`).

Implemented now:

- **`customer_ops/approval.py`**: `ApprovalRequest` (`request_id`,
  `action_type`, `order_id`, `message`, optional `amount`/`currency` for
  `issue_refund` derived from the validated order) and
  `HumanApprovalResponse` (`decision: "approved" | "rejected"`, using the
  existing `HumanDecision` vocabulary). Never includes customer email,
  shipping address, or a full customer/order record. A malformed resume
  payload raises `ApprovalError` - it is never silently interpreted.
- **`human_approval` node**: reads the already-prepared `ProposedAction` and
  `ActionInputResult`, builds the `ApprovalRequest`, then calls
  `interrupt(approval_request.model_dump(mode="json"))`. Because LangGraph
  re-executes an interrupted node from its beginning on every resume,
  everything before `interrupt()` here is a pure read/validation - no
  mutation, no network call, no audit-log append happens until after the
  human's decision comes back. On resume, it validates the decision and
  returns `Command(update=..., goto="execute_approved_action" |
  "approval_rejected")` - LangGraph-native resume routing.
- **`execute_approved_action` node**: only reached after an "approved"
  decision. Re-validates that the action genuinely required approval and
  that its input is ready, then calls the *same* `execute_action` used for
  safe actions, explicitly passing `human_approved=True` - the executor's
  own defense-in-depth check from Iteration 6 still applies and is never
  bypassed. Synchronizes `order_context` exactly like safe execution does.
- **`approval_rejected` node**: performs zero mutation; `action_result`
  stays absent; no automatic escalation.
- **Checkpointing**: `build_customer_ops_graph(..., checkpointer=None)`
  defaults to a fresh, process-local `InMemorySaver()` (never a module-level
  global). Once compiled, *every* invocation - interrupted or not - requires
  `config={"configurable": {"thread_id": ...}}`; the graph never generates
  or derives a thread_id, the caller owns thread identity.
- **Resume safety, proven by tests**: no store mutation happens before
  `interrupt()` or on node replay; an approved resume causes exactly one
  store mutation; resuming with an unrelated/unknown thread_id does not
  resume the pending approval (the fresh run hits `intake`'s own explicit
  validation instead); a malformed resume decision raises `ApprovalError`
  without mutating anything.
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> resolve_order -> evaluate_policy -> (conditional) -> {clarification, information, propose_action -> prepare_action_input -> (conditional) -> {execute_safe_action, clarification}, prepare_approval -> prepare_approval_input -> human_approval -- interrupt() -- (resume) --> {execute_approved_action, approval_rejected}, blocked} -> END`,
  built via `build_customer_ops_graph(classifier=None, store=None, action_input_extractor=None, action_store=None, checkpointer=None)`.
- Observable audit events for every stage, including `human_approval` (e.g.
  "Human reviewer approved the proposed simulated action.") and
  `approval_rejected` - never reviewer identity, timestamps, or hidden
  reasoning.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call, no
  interactive approval CLI yet).
- Unit tests for state contracts, models, the data store, the classifier,
  order resolution, policy evaluation, routing, action proposal, action
  inputs, the action store, the action executor, approval contracts, and
  full graph branch/HITL coverage - all running with no network access and
  no API key.

**Important limitations, stated accurately**: `InMemorySaver` checkpoint
state survives separate invoke/resume calls only while this Python process
and this saver instance stay alive - it provides **no durable persistence
across a process restart**, and it is a different concern entirely from the
simulated business-data mutations in `tools/action_store.py` (also
process-local). No real money, orders, or customer systems are ever
touched. The JSON fixtures remain example source data, not persistent
business storage.

Planned later (not implemented yet):

- Final customer-facing response generation.
- Evaluation harness (`evals/`).
- Streamlit UI.
- Deployment.
- Durable checkpoint/business-data persistence (a future improvement beyond
  `InMemorySaver`).

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
│   └── graph.py            # all graph nodes, build_customer_ops_graph()
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
│   └── __init__.py      # empty in Iteration 1
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
│   └── test_graph.py
│
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
`CustomerOperationsStore`, and a fake `ActionInputExtractor`, use a fresh
`InMemorySaver` per test graph, and make no network calls:

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
