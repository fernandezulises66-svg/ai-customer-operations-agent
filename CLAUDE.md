# CLAUDE.md

Guidance for Claude Code (and any future contributor) working in this repository.

## Project identity

- **AI Customer Operations Agent** — a portfolio project.
- Fictional e-commerce company: **Mercora**. All customers, orders, policies,
  payments, addresses, and tickets are synthetic. Never use or imply real
  company data.
- Purpose: demonstrate LangGraph, stateful workflows, conditional routing,
  tool usage, structured LLM outputs, human-in-the-loop, checkpointing, safe
  action execution, evaluation, and observability. Deliberately distinct from
  an SQL data agent or a RAG knowledge assistant.

## Architecture

- LangGraph (`StateGraph`) orchestrates workflow state and routing directly.
  Do not use LangChain agents, `create_agent`, CrewAI, AutoGen, or n8n.
- The OpenAI SDK is called directly inside specific graph nodes that need LLM
  reasoning. LangGraph orchestrates; it does not hide business logic.
- Business logic must be explicit and readable in node code, not buried in a
  framework abstraction.
- Tools (in `tools/`) must be narrow and independently testable.
- Sensitive/high-impact simulated actions require human approval before
  execution (human-in-the-loop gate in the graph).
- `CustomerOpsState` values must remain serialization/checkpoint friendly
  (str, int, float, bool, list, dict, or None). Never store Pydantic model
  instances, live clients, or other runtime objects directly in state -
  convert them (e.g. `model_dump(mode="json")`) before writing to state.

## Safety and behavior

- Never fabricate customer or order data.
- Never execute real financial or external actions — all operational actions
  are simulated.
- Actions must be validated before execution.
- Future high-impact actions require explicit human approval in the workflow.
- No hidden reasoning / chain-of-thought in outputs or audit logs. The audit
  trail records observable events (what happened), not model reasoning, and
  not full customer/order details (e.g. email, address, tracking number) -
  keep audit messages generic and observable, never a data dump.
- Model decisions should use structured outputs (e.g. Pydantic schemas via
  the OpenAI Responses API `text_format`), not free-form text parsing, where
  practical.
- A technical failure (API error, malformed/missing structured output) must
  never silently become a business decision. Raise a domain-specific
  exception instead of substituting a default value such as `intent =
  "other"`.
- Operational facts (customer identity, order status, totals, addresses,
  tracking numbers, purchased items, etc.) must come from deterministic
  data/tools, never from model invention - even when the model is only
  "filling in" a plausible-looking value.
- Read-only and mutating operational tools must stay clearly separated
  (e.g. `tools/customer_data.py` is read-only; simulated mutations belong in
  their own future module and require human approval).
- Business eligibility (whether an action is allowed, which order a request
  refers to) must be decided by deterministic policy code, never LLM
  improvisation. The LLM interprets intent/urgency only.
- Ambiguous operational targets (e.g. which order a message refers to) must
  not be guessed. Conservative deterministic logic should prefer an
  explicit "needs clarification" business outcome over a plausible-looking
  guess.
- A human-approval requirement is a policy *output* (e.g.
  `requires_human_approval` on a policy assessment); actually pausing for
  and acting on that approval is a separate, later workflow step.
- Technical/state-corruption errors (e.g. a referenced order absent from
  the customer's own scoped context) must raise a domain exception, never
  silently resolve to a valid-looking business outcome.
- Conditional routing must be driven by explicit structured state (e.g.
  intent, order resolution, policy assessment), never by asking the LLM
  which branch to take.
- Proposing an action and executing it are separate stages. A proposed
  action is a structured statement of intent only - never invent its
  execution payload (e.g. a new address, refund amount, or replacement
  item) before that data actually exists.
- Human-approval state must never be fabricated: `requires_human_approval`
  is a policy/action output, but a human decision (`human_decision`) must
  never be set except by an actual future approval step.
- Execution must use explicit, validated action inputs (e.g.
  `ActionInputResult`/one Pydantic model per action type) - never a raw or
  arbitrary dict payload.
- A missing action input (e.g. a replacement address the customer never
  supplied) must never be invented, autocompleted, or guessed - it becomes
  an explicit "not ready"/clarification-required outcome instead.
- Approval-required execution requires explicit approval at the executor
  boundary itself (e.g. `human_approved=True`), enforced even if upstream
  routing already prevents reaching it - defense in depth, not a single
  point of failure.
- Business-store mutation is simulated and in-memory only; it must never
  write to the synthetic JSON fixtures or any real external system.
  Mutation (`tools/action_store.py`) and fixture persistence
  (`data/*.json`, loaded by `tools/customer_data.py`) are separate
  concerns - fixtures are example source data, not mutable business
  storage.
- After a successful simulated mutation, graph state must be
  resynchronized (e.g. re-read the affected order and replace it in
  `order_context`) so state never shows stale data - and only the affected
  record changes, never an unrelated one.

## Language

- Code, comments, docstrings, commit messages, and README content: English.
- Customer-facing demo content may later be Spanish-first.
- Synthetic company data may be Spanish where appropriate (e.g. sample
  customer messages).

## Testing

- pytest must never make real OpenAI calls.
- Model clients must be injectable/mockable.
- Deterministic logic gets deterministic tests.
- External/model failures must eventually be handled explicitly (no silent
  swallowing of errors).

## Scope control

- Implement only the requirements of the current iteration.
- Do not silently build ahead into future iterations.
- Avoid unnecessary frameworks and speculative abstractions — prefer the
  simplest structure that satisfies the current iteration.

## Environment

- Target Python 3.13+.
- Development happens on Windows using PowerShell and VS Code.
- Dependencies are installed into a local `.venv`, never a global Python
  install.
