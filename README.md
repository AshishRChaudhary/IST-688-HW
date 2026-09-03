# IST 688 - Homework Application

A multipage Streamlit app for "Building Human-Centered AI Applications." The app is titled
"HW Manager"; each homework lives as a `.py` file in the `HW/` sub-directory, added as a page
via `st.navigation` in `streamlit_app.py`. New homeworks get added the same way: drop a
`HWX.py` into `HW/` and add it to the navigation list.

- **HW 1** (`HW/HW1.py`) - Document question answering: upload a `.txt`/`.pdf` file, ask a
  question about it, and get an answer from GPT (user supplies and validates their own
  API key via a text input).
- **HW 2** (`HW/HW2.py`, default page) - URL summarizer: enter a web page URL at the top of
  the page and get a summary, with sidebar options for summary type, output language, and
  which LLM to use (OpenAI or Anthropic), plus the 'use advanced model' checkbox. Keys come
  from `st.secrets` and are validated against the selected provider before any summary runs.

`HW1.py` also exists at the repo root: that is the standalone HW 1 app, deployed on its own
per the HW 1 assignment (`streamlit run HW1.py`). It is byte-identical to `HW/HW1.py` — if you
edit one, copy the change to the other.

Assignment instruction PDFs under `Instructions/` and the model-comparison write-ups are kept
locally and are gitignored (not part of the repo).

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Add your API keys to `.streamlit/secrets.toml`:

   ```
   OPENAI_API_KEY = "your-key-here"
   ANTHROPIC_API_KEY = "your-key-here"
   ```

3. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```

### Deploying to Streamlit Community Cloud

`.streamlit/secrets.toml` is gitignored and never pushed to GitHub. After deploying,
add the same keys under the app's **Settings > Secrets** in the Streamlit Community Cloud
dashboard — they're stored separately from the repo and injected into `st.secrets` at runtime.
