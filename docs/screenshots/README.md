# Screenshots

This directory holds real screenshots of the running Streamlit demo, taken
manually after deployment (or from a local `streamlit run streamlit_app.py`
session). No screenshot here is synthetic or generated - each one must come
from the actual app.

## Planned screenshots

1. **`information-response.png`** - a completed order-status flow: a
   synthetic customer's order-status question, answered directly with no
   approval interrupt, showing the grounded final response.

2. **`approval-pending.png`** - a sensitive action (refund, billing
   investigation, or product investigation) paused at the human-approval
   card. Must show the pending-approval UI with **no final response yet**
   and no mutation applied.

3. **`approval-completed.png`** - the same case immediately after clicking
   **Aprobar**, showing the resumed workflow's completed final response.

4. **`workflow-details.png`** - the "Detalles del workflow" and/or
   "Registro de auditoría" expanders open, showing the structured
   observability fields (intent, route, policy outcome, workflow status,
   audit steps).

## Rules for any screenshot added here

- No API keys, tokens, or other secrets visible anywhere in the image.
- No browser developer tools, terminal output, or local file paths.
- No personal information - only the synthetic Mercora customer/order data
  already built into this project's fixtures.
- Crop to the relevant app content; a full-desktop screenshot is not
  necessary.

## Status

No screenshots have been added yet. `README.md` intentionally does not
reference images from this directory until real screenshots exist here -
add the `![...](docs/screenshots/...)` references in a small follow-up
once they do, so the main README never links a missing image.
