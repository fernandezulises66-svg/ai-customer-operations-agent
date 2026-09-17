"""Evaluation harnesses for the Customer Operations Agent.

`e2e_cases` / `e2e_checks` / `e2e_runner` implement the curated end-to-end
evaluation benchmark (see `run_e2e_evals.py` for the manual CLI entry point
and `README.md`'s "Evaluation" section for how to run and interpret it).
The benchmark's own tests, `tests/test_e2e_evals.py`, run fully offline
against fake workflow dependencies - only the CLI makes real OpenAI calls,
and only when run manually.
"""
