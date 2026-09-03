# ML Pipeline Agent

A multi-agent ML pipeline: five LLM agents each wrap one deterministic pipeline
stage (data profiling, feature engineering, hyperparameter tuning, training),
with a rule-based orchestrator that routes loop-backs between them. A FastAPI
service exposes it over HTTP, and a Streamlit app drives it end to end.

See `AGENT.md` for the architecture in depth.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

Create a `.env` in the project root:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+psycopg://postgres:<password>@localhost:5432/ml_pipeline
```

`DATABASE_URL` needs a reachable PostgreSQL database. Defaults live in
`core/config.py`.

## Running it

The backend and the UI are two separate processes, in two terminals.

**Terminal 1 — the API:**

```bash
uvicorn api.main:app --reload
```

Serves on http://127.0.0.1:8000. Interactive docs at `/docs`.

**Terminal 2 — the Streamlit test harness:**

```bash
streamlit run streamlit_app.py
```

Opens on http://localhost:8501. The API base URL is configurable in the
sidebar and defaults to `http://127.0.0.1:8000`; the sidebar shows whether the
backend is reachable. Start the API first — the UI is a pure HTTP client and
has nothing to talk to on its own.

### Using the app

- **New Run** — either fill in a structured spec (the `success_metric` options
  are driven by `task_type`, from `ProblemSpec`'s own validator sets) or
  describe the problem in plain English and let `RequirementAgent` extract it.
  Either mode can upload a `.csv`/`.parquet`, which is POSTed to `/uploads`
  first and the returned path used as `data_source`.
  If the agent answers with `needs_clarification`, the UI shows the question
  and missing fields and lets you revise — that is a normal response, not an
  error.
- **Live Run** — polls `GET /runs/{run_id}` every 2s while the run is active,
  and stops once it reaches `done` or `escalated`. Shows stage progress, each
  completed stage's real output, the full LLM trace with per-call cost, and the
  history log.
- **Run History** — lists past runs, newest first; open any one to see the same
  detail view.

## Test data

`sample_data/` holds datasets built to exercise every branch of the feature
stage (numeric, bool, low/high-cardinality categorical, datetime, missing
values, leakage). Regenerate with:

```bash
python sample_data/make_sample_data.py
```

Two things to know:

- **The target column must be numeric (0/1), not text.** `feature_stage.py`
  correlates features against `y.astype(float)`; a string target makes that
  raise, every correlation falls back to `0.0`, and every feature is dropped as
  low-signal — leaving a zero-column matrix that training cannot fit.
- **Use `.parquet` if you want the datetime path exercised.** A CSV round-trip
  turns `datetime64` back into a string, after which the column is treated as
  high-cardinality text and dropped. Parquet preserves the dtype, so the column
  is decomposed into year/month/day_of_week/is_weekend.

## Tests

```bash
python -m pytest Test -q
```

`Test/unit/` covers the deterministic stages and tool wrappers with no mocking;
`Test/integration/` covers the agents and orchestrator using the fake OpenAI
client in `Test/conftest.py`, so the suite never makes a network call or costs
money. `Test/debug_scripts/` holds manual scripts that DO hit the real API —
they are deliberately named without a `test_` prefix so pytest never collects
them.
