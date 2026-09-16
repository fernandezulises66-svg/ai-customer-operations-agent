# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution, policy evaluation, conditional routing, structured action
proposals, and simulated safe-action execution (Iteration 6).** This is a
portfolio project and is not production software.

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
  approval step before execution. That interrupt/resume step does not exist
  yet (Iteration 6 prepares validated execution input for it but never
  executes an approval-required action); `issue_refund`,
  `investigate_billing`, and `investigate_product_issue` currently stop at
  `awaiting_approval`.
- The workflow state is checkpoint-friendly and produces a **structured audit
  trail** of observable events (not model reasoning).

## Current implementation (Iteration 6: Action Inputs and Simulated Execution)

The full pipeline from a customer message to a simulated operational change
now exists:

**Policy** (what is allowed) **-> Routing** (which workflow path follows)
**-> Proposed Action** (what operation is intended) **-> Action Input**
(what validated parameters it needs) **-> Executor** (calls exactly one
store method) **-> Simulated Store** (the in-memory mutation).

- **LLM**: intent + urgency interpretation, and - only for `change_address`
  - structured address extraction (`customer_ops/classifier.py`,
  `customer_ops/action_inputs.py`).
- **Deterministic data store**: customer/order facts (`tools/customer_data.py`).
- **Deterministic resolver**: which order a request refers to
  (`customer_ops/order_resolution.py`).
- **Deterministic policy engine**: operational eligibility
  (`customer_ops/policies.py`).
- **Deterministic router**: which workflow branch follows from that policy
  result (`customer_ops/routing.py`).
- **Structured action proposals**: what a future action *would* do
  (`customer_ops/action_proposal.py`).
- **Structured action inputs**: the validated parameters an action needs
  before it can run, never invented (`customer_ops/action_inputs.py`).
- **Simulated execution**: exactly one mutation call against an in-memory
  operational store, for safe actions only (`customer_ops/action_executor.py`,
  `tools/action_store.py`).

Implemented now:

- **Explicit action-input contracts** (`customer_ops/action_inputs.py`):
  `CancelOrderInput`, `ChangeAddressInput`, `RefundInput`,
  `InvestigationInput` - one Pydantic model per action type - and
  `ActionInputResult` (`ready`, `action_type`, `parameters`,
  `missing_fields`). `cancel_order`, `issue_refund`, and both investigation
  types build their input deterministically (just `order_id`); only
  `change_address` needs a customer-supplied value not already in hand.
- **Structured address extraction** for `change_address`
  (`OpenAIActionInputExtractor`, via `ActionInputExtractor`): one OpenAI
  Responses API Structured Outputs call that returns only the address the
  customer explicitly typed - never inferred, completed, or guessed. No
  address means `ActionInputResult(ready=False,
  missing_fields=["new_shipping_address"])`, never an invented address.
- **A simulated in-memory mutable operational layer**
  (`tools/action_store.py`): `InMemoryCustomerActionStore` implements both
  the read-only `CustomerOperationsStore` protocol and a new mutating
  `CustomerActionStore` protocol (`cancel_order`, `change_shipping_address`,
  `issue_full_refund`, `create_billing_investigation`,
  `create_product_investigation`) against the *same* in-memory records, so
  one graph invocation always sees a coherent snapshot rather than two
  independently-loaded copies that could drift apart. `data/customers.json`
  and `data/orders.json` are never written to - mutations live only in that
  store instance's memory for the life of the process.
- **Safe action execution** (`customer_ops/action_executor.py`,
  `execute_action`): converts a validated `ProposedAction` +
  `ActionInputResult` into exactly one `CustomerActionStore` call and
  returns a structured `ActionResult` (`action_type`, `success`,
  `order_id`, `message`, optional `reference_id`). Contract mismatches,
  not-ready input, unsupported action types, and store failures all raise
  `ActionExecutionError` - never a fake success.
- **Defense-in-depth approval gate at the executor boundary**:
  `execute_action(..., human_approved=...)` refuses to run `issue_refund`,
  `investigate_billing`, or `investigate_product_issue` unless
  `human_approved=True` is passed explicitly - enforced even though the
  graph in this iteration never calls the executor for those action types
  at all.
- **Graph changes**: the safe-action branch now runs
  `propose_action -> prepare_action_input -> (conditional) ->
  execute_safe_action` (ready) or back to `clarification` (missing input -
  nothing is guessed, and the proposal is kept in state for traceability).
  The approval branch runs `prepare_approval -> prepare_approval_input ->
  awaiting_approval` and still never executes.
- **State synchronization after mutation**: a successful safe execution
  re-reads the affected order from the same store and replaces only that
  order's entry in `order_context` - graph state never shows stale data,
  and no other order is touched.
- LangGraph workflow:
  `START -> intake -> classify_request -> load_context -> resolve_order -> evaluate_policy -> (conditional) -> {clarification, information, propose_action -> prepare_action_input -> (conditional) -> {execute_safe_action, clarification}, prepare_approval -> prepare_approval_input, blocked} -> END`,
  built via `build_customer_ops_graph(classifier=None, store=None, action_input_extractor=None, action_store=None)`.
- Observable audit events for every stage, including `action_input` and
  `action_execution` (e.g. "Additional action input is required:
  new_shipping_address.") - never the actual address text, hidden
  reasoning, or other customer/order PII.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts, models, the data store, the classifier,
  order resolution, policy evaluation, routing, action proposal, action
  inputs, the action store, the action executor, and full graph branch
  coverage - all running with no network access and no API key.

**Important limitations, stated accurately**: every mutation is simulated
and lives only in that process's memory - a restart resets it completely.
No real money, orders, or customer systems are ever touched. The JSON
fixtures remain example source data, not persistent business storage; this
is not the same thing as the LangGraph checkpointing that arrives in a
later iteration (that persists *workflow* state, not business data).

Planned later (not implemented yet):

- LangGraph human interrupt/resume (`interrupt()` / `Command(resume=...)`)
  and the actual approval workflow.
- Checkpointing / thread persistence.
- Final customer-facing response generation.
- Evaluation harness (`evals/`).
- Streamlit UI.
- Deployment.

## Tech stack

- Python 3.13+
- [LangGraph](https://github.com/langchain-ai/langgraph) `>=1.1,<2.0` for
  orchestration
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
`CustomerOperationsStore`, and a fake `ActionInputExtractor`, and make no
network calls:

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
