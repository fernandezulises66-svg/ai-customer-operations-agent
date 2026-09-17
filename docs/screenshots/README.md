# Screenshots

This directory holds real screenshots of the running Streamlit demo, taken
manually after deployment (or from a local `streamlit run streamlit_app.py`
session). No screenshot here is synthetic or generated - each one must come
from the actual app.

## Current screenshots

All four planned screenshots have been captured from the public deployed
app and are embedded in the main `README.md`'s "Demo Screenshots" section:

1. **`information-response.png`** - a completed order-status flow: a
   synthetic customer's order-status question, answered directly with no
   approval interrupt, showing the grounded final response.

2. **`approval-pending.png`** - a sensitive action (refund) paused at the
   human-approval card, with **no final response yet** and no mutation
   applied.

3. **`approval-completed.png`** - the same case immediately after clicking
   **Aprobar**, showing the resumed workflow's completed final response.

4. **`workflow-details.png`** - the "Detalles del workflow" expander open,
   showing the structured observability fields (intent, route, policy
   outcome, workflow status, and more).

## Rules for any screenshot added here

- No API keys, tokens, or other secrets visible anywhere in the image.
- No browser developer tools, terminal output, or local file paths.
- No personal information - only the synthetic Mercora customer/order data
  already built into this project's fixtures.
- Crop to the relevant app content; a full-desktop screenshot is not
  necessary.

## Status

All four screenshots above exist and are referenced from the main
`README.md`. If a screenshot is ever replaced (e.g. after a UI change),
re-capture it from the real running app and re-check it against the rules
above before committing - never restore a placeholder or generated image.
