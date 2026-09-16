# AI Customer Operations Agent

**Status: intake + classification + context loading + deterministic order
resolution, policy evaluation, conditional routing, structured action
proposals, simulated safe-action execution, human-in-the-loop approval with
LangGraph checkpointing, and final customer-facing response generation
(Iteration 8).** This is a portfolio project and is not production
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
│   ├── response_generator.py # CustomerResponse, ResponseContext, build_response_context()
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
│   ├── test_response_generator.py
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
