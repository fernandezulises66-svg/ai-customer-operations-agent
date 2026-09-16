# AI Customer Operations Agent

**Status: foundation only (Iteration 1).** This is a portfolio project and is
not production software.

> Mercora is a fictional e-commerce company invented for this project. All
> customers, orders, policies, payments, addresses, and tickets referenced
> anywhere in this repository are synthetic and do not represent any real
> business or individual.

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
  n8n.
- The **OpenAI SDK** is called directly inside specific graph nodes that need
  LLM reasoning; LangGraph orchestrates, it does not hide business logic.
- **Tools** (`tools/`) are narrow, independently testable functions used by
  graph nodes.
- Sensitive/high-impact simulated actions require a **human-in-the-loop**
  approval step before execution.
- The workflow state is checkpoint-friendly and produces a **structured audit
  trail** of observable events (not model reasoning).

## Current implementation (Iteration 1: Project Foundation and LangGraph State)

Implemented now:

- Project scaffold (`customer_ops/`, `tools/`, `data/`, `evals/`, `tests/`).
- Typed workflow state (`customer_ops/state.py`): `CustomerOpsState`,
  `AuditEvent`, and controlled `Literal` vocabularies for intent, urgency,
  workflow status, and human decision.
- A single deterministic node, `intake_node`, that validates and normalizes
  the initial request (`request_id`, `customer_id`, `customer_message`),
  sets `workflow_status` to `"received"`, and appends one audit event. It
  makes no LLM calls and infers nothing.
- A minimal compiled LangGraph workflow: `START -> intake -> END`, built via
  `build_customer_ops_graph()`.
- A minimal `app.py` placeholder entry point (no CLI, no OpenAI call).
- Unit tests for state contracts and graph behavior.

Planned later (not implemented yet):

- LLM-based request classification (intent, urgency).
- Operational tools (customer lookup, order lookup, etc.).
- Policy evaluation logic.
- Conditional routing between workflow branches.
- Human-in-the-loop approval gate.
- Checkpointing / persistence.
- Simulated action execution.
- Evaluation harness (`evals/`).
- Streamlit UI.
- Deployment.

## Tech stack

- Python 3.13+
- [LangGraph](https://github.com/langchain-ai/langgraph) `>=1.1,<2.0` for
  orchestration
- [OpenAI SDK](https://github.com/openai/openai-python) `>=3.0,<4.0` (not yet
  called in this iteration)
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
│   ├── state.py        # CustomerOpsState, AuditEvent, controlled vocabularies
│   └── graph.py         # intake_node, build_customer_ops_graph()
│
├── tools/
│   └── __init__.py      # empty in Iteration 1
│
├── data/
│   └── .gitkeep
│
├── evals/
│   └── __init__.py      # empty in Iteration 1
│
├── tests/
│   ├── __init__.py
│   ├── test_state.py
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

Copy `.env.example` to `.env` if you plan to configure environment variables
later (no key is required for Iteration 1 — no OpenAI calls are made):

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
