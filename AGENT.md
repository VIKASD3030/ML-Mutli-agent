# AGENT.md — Codebase Map & Deep Analysis

> Purpose: a durable, pre-computed analysis of this repo so an agent (or a human)
> can act immediately without re-reading every file. Rewritten 2026-09-16 against
> the tree as it stands (previous version was written 2026-08-21 and is now
> substantially stale — the orchestrator/agent wiring, caching layer, API,
> database, and both frontends described below did not exist yet). If you change
> architecture, update this file.

---

## 1. What this project is

An **LLM-orchestrated ML pipeline agent**, now a real full-stack app: React
frontend (or a Streamlit alternative) → FastAPI → Postgres, wrapping a 5-agent
pipeline that turns a plain-English ML request into a trained, evaluated model.

A user's free-text request ("predict tumor malignancy, aim for 0.9 f1"), an
uploaded CSV/Parquet file, or a fully structured spec is turned into a validated
`ProblemSpec`, then pushed through 5 stages: data profiling → feature
engineering/EDA → hyperparameter tuning → training/evaluation → pass/fail with
loop-back routing.

The organizing principle of the whole codebase, unchanged from Phase 1 and still
the thing to preserve when editing:

> **Deterministic Python does all computation and all routing.
> The LLM only reads structured reports and produces a judgment.**

The LLM never sees raw data, never builds a model, never decides where control
flows. Every agent hands the model a Pydantic object serialized with
`model_dump_json()` and receives back a narrow structured verdict.

**What changed since the last analysis, in one paragraph:** the five agents are
no longer standalone demos — `pipeline/run_pipeline.py` now instantiates and
calls all four pipeline agents directly, threading a `RunCache` (in-memory,
per-run data-sharing) and a `PipelineRun` (persisted state + LLM call trace)
through the whole loop. Every LLM call goes through `pipeline/observability.py`,
which records token counts, latency, and estimated cost as a `TraceEntry`. A
FastAPI + SQLAlchemy + Alembic backend persists every run to Postgres and
exposes it over HTTP; a React frontend (and a Streamlit test-harness UI) consume
that API for upload → kick off a run → watch it live → inspect results/cost.
CSV/Parquet upload, categorical encoding, and datetime feature decomposition —
all called out as gaps in the previous analysis — are implemented.

---

## 2. Layout

```
schemas/        Pydantic hand-off objects (the contracts between stages/layers)
pipeline/       Phase 1 stage functions + the orchestrator + cache + observability
tools/          Flat-argument, LLM-callable wrappers over pipeline/
agents/         LLM agents that call one tool then judge its output
api/            FastAPI app: HTTP routes, request/response schemas
db/             SQLAlchemy ORM models, session/engine, Pydantic<->ORM sync
alembic/        Migration history for db/models.py's tables
frontend/       React (Vite) app — the primary UI
streamlit_app.py  Secondary/test-harness UI, talks to the API over HTTP only
Test/           Real pytest suite: Test/unit, Test/integration
Test/debug_scripts/  Ad-hoc print-and-eyeball scripts (NOT pytest)
.env            OPENAI_API_KEY (loaded via python-dotenv in every agent module)
.venv/          Python 3.13.14 virtualenv
```

Layer direction, still strictly one-directional at the core, now with two more
layers wrapped around it:

```
frontend/ (React)  →  api/ (FastAPI)  →  db/ (SQLAlchemy) ┐
                              ↓                            │ persists/restores
                    pipeline/run_pipeline.py (orchestrator) ┘
                              ↓
agents/  →  tools/     →  pipeline/   →  schemas/
(LLM)       (flat args,    (pandas /      (pure data
             validation)    sklearn)       contracts)
```

`schemas/` imports nothing from the other core layers. `pipeline/` (stage
functions) imports only `schemas/`. `api/` and `db/` sit outside that core chain
and talk to it only through `pipeline.run_pipeline.run()` and the `PipelineRun`
object. Never invert this.

---

## 3. The data contracts (`schemas/`)

All re-exported from `schemas/__init__.py`.

| File | Object | Produced by | Key fields / properties |
|---|---|---|---|
| `problem_spec.py` | `ProblemSpec` | RequirementAgent | `task_type`, `target_column`, `success_metric`, `metric_threshold`, `data_source`, `constraints` — **unchanged** from the previous analysis |
| | `PipelineConstraints` | | `max_tuning_trials=50`, `max_loop_backs=3`, `interpretability_required`, `max_training_seconds` |
| | `ClarificationNeeded` | | returned *instead of* a spec when the request is ambiguous |
| `dataset_peek.py` | `DatasetPeek` / `ColumnPeek` | **NEW** — `peek_dataset_tool()` | Per-column schema-only summary (`name`, `dtype`, `missing_pct`, `unique_count`, up to 5 `example_value`s) of an uploaded file, read BEFORE a `ProblemSpec` exists — grounds RequirementAgent's extraction in real column names instead of letting it guess |
| `data_profile.py` | `DataProfile` | `clean_and_profile()` | `column_schema`, `class_balance`, `outlier_flags`, `cleaning_actions_taken`, `warnings`, `blocking_issue`; `.is_usable` |
| `eda_report.py` | `EDAReport` | `engineer_features()` | `correlation_summary`, `engineered_features`, `dropped_features`, `leakage_warnings`, **`data_quality_warnings`** (new — NaT/missing-datetime warnings, kept deliberately separate from `leakage_warnings`: "will this run complete" vs "can this result be trusted"), `final_feature_names`; `.has_leakage_risk` |
| `tuning_result.py` | `TuningResult` | `tune_hyperparameters()` | `best_params`, `best_cv_score`, `search_budget_used/total`, `convergence_notes`, `converged`, `search_history` (per-trial records — now actually populated, see §4) |
| `evaluation_report.py` | `EvaluationReport` | `train_and_evaluate()` | `test_metrics`, `metric_value/threshold`, `pass_fail`, `failure_analysis`; `.passed` |
| `trace_entry.py` | `TraceEntry` | **NEW** — `observability.timed_llm_call()` | One record per LLM API call: `agent`, `model`, `prompt_tokens`, `completion_tokens`, `total_tokens`, `latency_ms`, `estimated_cost_usd`, `timestamp` |
| `pipeline_run.py` | `PipelineRun` | orchestrator | `run_id`, `status`, `loop_count`, all four stage reports, `history[]` (`.log()`), **`trace: list[TraceEntry]`** (`.record_call()`), `.total_cost_usd`, `.total_tokens` |

Closed vocabularies, unchanged: `CLASSIFICATION_METRICS`, `REGRESSION_METRICS`,
`RecommendedNextStep = Literal["revisit_features", "expand_hyperparam_search",
"insufficient_data", "bad_spec"]`, and `RunStatus = Literal["created",
"data_ready", "features_ready", "tuned", "evaluated", "done", "escalated"]`.

---

## 4. The deterministic stages (`pipeline/`)

### `data_stage.py`
- `load_data(spec) -> DataFrame` now supports **three source kinds**:
  1. `"builtin:breast_cancer"` — sklearn breast-cancer frame, `target` renamed
     to `spec.target_column` (unchanged).
  2. Any path ending `.csv` — `pd.read_csv`. No column renaming: an uploaded
     file's `target_column` is already grounded against its real column names
     via `peek_dataset_tool()` before a `ProblemSpec` is ever built, so nothing
     needs remapping (unlike the builtin dataset's generic `target` column).
  3. Any path ending `.parquet` — `pd.read_parquet`, same no-rename reasoning.
  Both file branches re-raise `FileNotFoundError`/other read failures as
  `ValueError` with the path and underlying error included, so callers only
  ever need to handle one exception type. Anything else still raises
  `ValueError(f"Unsupported data_source: ...")`.
- **`column_schema` indentation bug from the previous analysis is fixed** — the
  `ColumnSchema` append now sits correctly inside the `for col in df.columns`
  loop (§7's old item 1 no longer applies), so every column gets recorded, not
  just the last one.
- `is_datetime_column()` / `is_categorical_column()` now live here (not in
  `feature_stage.py`) as shared dtype predicates, specifically so the two
  stages can never disagree about what counts as "categorical." Notably,
  `is_categorical_column()` checks `pd.StringDtype` explicitly — as of pandas
  3.0 (pinned here at 3.0.5) plain text columns infer as `StringDtype`, not
  `object`, so `is_object_dtype()` alone would have silently matched nothing.
- `clean_and_profile()` behavior otherwise unchanged: target-missing/blocking
  checks, drop-missing-target rows, drop exact duplicates, median-impute
  numeric columns, **mode-impute categorical columns (new)**, class balance for
  classification, IQR-based outlier flagging. Datetime columns are left
  untouched here on purpose — decomposed in `engineer_features()` instead.

### `feature_stage.py`
Thresholds unchanged: `Leakage_correlation_threshold = 0.98`,
`Low_signal_correlation_threshold = 0.01`, `Low_variance_threshold = 1e-8`. New:
`MAX_CATEGORICAL_CARDINALITY = 20`.

**Categorical encoding and datetime decomposition are now implemented** (both
were the single biggest gaps flagged previously):
- Bool columns → cast to `int` (checked first — one-hot on a bool would just
  produce two anti-correlated copies of the same bit).
- Datetime columns → decomposed into `{col}__year`, `{col}__month`,
  `{col}__day_of_week`, `{col}__is_weekend`; the raw column is dropped with a
  `FeatureDecision` explaining why (RandomForest can't consume `datetime64`).
  Missing dates (NaT) produce a `data_quality_warnings` entry, since NaN
  downstream would otherwise crash training several stages later with an error
  naming neither the column nor this stage.
- Categorical columns → one-hot encoded via `pd.get_dummies(dtype=float)`
  **unless** cardinality exceeds `MAX_CATEGORICAL_CARDINALITY` (20), in which
  case the column is dropped outright (avoids a dimensionality explosion from a
  high-cardinality ID-like column). Target/hash encoding remains future scope.
- All of the above feed through the **same** correlation/variance/leakage
  filter as original numeric columns — no separate filtering path.
- Leakage handling (`corr >= 0.98` → warn but keep) and low-signal dropping
  (`corr < 0.01`) unchanged. One "engineered" interaction feature (product of
  the two highest-correlation survivors) still added at the end.

### `tuning_stage.py`
Unchanged from the previous analysis and still correct: Optuna,
`direction="maximize"`, RandomForest search space, `f1_macro` /
`neg_root_mean_squared_error` scoring, 3-fold CV, plateau-based
`converged`/`convergence_notes` computed from one shared boolean, and
`search_history` actually populated and returned (the "computed but dropped on
the floor" bug noted before is confirmed fixed — `search_history=history` is
present in the return).

### `training_stage.py`
Unchanged and confirmed still correct: uses `root_mean_squared_error()` (the
`squared=False` removal from sklearn 1.6+ was already fixed), computes every
metric each task_type's validator allows and does an **exact key lookup** for
`metric_value` (raises rather than silently substituting a different metric),
and the same three-rule rule-based `failure_analysis` ladder.

### `run_pipeline.py` — now actually wires the agents together
This is the largest structural change since the previous analysis. Previously:
"the orchestrator is purely deterministic today; the five agents are
independent entry points... `agents/` and `run_pipeline.py` never call each
other." **That gap is now closed.**

`run(spec, run_id=None, on_update=None) -> PipelineRun`:
- `run_id`: optional, lets a caller (the API) choose the run's identity up
  front instead of generating one internally — needed so the id returned to an
  API client and the id actually persisted are the same one.
- `on_update`: optional callback invoked after every point `run_state`
  materially changes. `run_pipeline.py` has zero knowledge of the database —
  the API's background task passes a callback that calls `persist_run_state()`,
  which is what makes live progress polling possible while keeping the
  orchestrator decoupled from persistence.
- Instantiates `DataAgent`, `FeatureAgent`, `TuningAgent`, `TrainingAgent` and
  a single `RunCache()` per run. Calls each agent's `.run(...,
  pipeline_run=run_state, cache=cache)` in sequence — **agents themselves now
  raise/lower `run_state.status` and append to `run_state.history` via
  `.log()`**, not the orchestrator directly for most of it.
- Loop-back logic is effectively unchanged in spirit (`revisit_features` →
  restart from feature stage; `expand_hyperparam_search` → `max_tuning_trials
  *= 1.5`, resume from tuning; anything else → escalate; `loop_count >=
  max_loop_backs` → escalate) but is now expressed via a `resume_from` string
  ("feature" | "tuning") rather than a bare `continue`, since a
  `expand_hyperparam_search` loop-back deliberately **skips re-running
  FeatureAgent** — features didn't change, only the tuning budget did, and the
  cached `X`/`y` in `RunCache` make that skip actually reuse prior work instead
  of recomputing it.
- `DataAgent`/`FeatureAgent`/`TuningAgent`/`TrainingAgent` returning
  `proceed=False` (or raising a caught `PipelineBlockedError` inside the tool)
  escalates the run immediately, except `TuningAgent`: a `proceed=False`
  judgment from `TuningAgent` is logged as a concern but **does not** stop the
  pipeline — training still runs for a real evaluation, since the tuning
  agent's judgment is advisory, not a hard gate (mirrors `TrainingAgent`'s own
  advisory role on the *next* loop-back decision).

### `run_cache.py` — **NEW**, the caching layer the previous roadmap flagged as missing
`RunCache`: a minimal in-memory `dict[run_id][key] -> value` store — nothing
more (no persistence, no eviction, no thread-safety; explicitly "not production
infrastructure," the smallest thing that lets agents within one run share data).
Documented keys in use: `"cleaned_df"` (set by `DataAgent`'s tool, read by
`FeatureAgent`'s), `"X"`/`"y"` (set by `FeatureAgent`'s tool, read by
`TuningAgent`'s and `TrainingAgent`'s). This is what turns "every tool
recomputes the full prefix chain from scratch" — flagged as the single biggest
cost problem in the previous analysis — into "compute once per run, reuse
downstream," **when** a `PipelineRun`+`RunCache` are actually threaded through
(a standalone/test call with no cache still recomputes everything, unchanged
behavior).

`TuningAgent`'s cache write of `pipeline_run.tuning_result` is what additionally
lets `TrainingAgent` skip re-running the entire hyperparameter search — its
tool accepts an optional `tuning_result` and only calls `tune_hyperparameters()`
if one wasn't already supplied.

### `observability.py` — **NEW**, per-call LLM cost/latency tracking
`timed_llm_call(client, agent_name, model, messages, response_format,
temperature=0) -> (parsed_object, TraceEntry)`. Every agent calls this instead
of `client.beta.chat.completions.parse()` directly, so cost/latency tracking is
automatic and consistent across all 5 agents rather than each reimplementing
it. Computes `estimated_cost_usd` from a `PRICING_PER_1K_TOKENS` table
(currently only `gpt-4o-mini` is priced; **the rates are explicitly flagged in
the module docstring as illustrative placeholders, not verified-current
pricing** — check before trusting cost figures beyond rough relative
comparison). Returns the `TraceEntry`; it's the caller's job (each agent) to
append it to `pipeline_run.trace` via `pipeline_run.record_call()` — this
function doesn't take a `PipelineRun` itself, keeping it usable standalone/in
tests.

**Known gap, called out directly in `requirement_agent.py`'s own comments**:
`RequirementAgent.run()` also calls `timed_llm_call()` (it runs before a
`ProblemSpec`/`PipelineRun` exists), but its `TraceEntry` is discarded — there's
no `PipelineRun` yet to attach it to. Requirement-gathering LLM cost is
therefore **not** currently reflected in `PipelineRun.total_cost_usd`/
`total_tokens`, or in the frontend's LLM-usage view. Left as an open item in
the code itself.

---

## 5. The tool layer (`tools/`)

Same flat-primitives-in, one-report-out philosophy as before, now with two
additive optional parameters threaded through every tool: `cache: RunCache |
None` and `run_id: str | None`. **Neither changes a tool's return value or
behavior when omitted** — a standalone call with just the original arguments
behaves identically to before this change; a cache miss (either param `None`,
or the cache empty) always degrades gracefully to a full recompute, never
raises, never uses stale data.

| Tool | Cache reads | Cache writes | Returns |
|---|---|---|---|
| `profile_dataset_tool` | — | `"cleaned_df"` | `DataProfile` |
| `engineer_features_tool` | `"cleaned_df"` (+ requires a `profile` arg passed in, else treated as a miss) | `"cleaned_df"` (on miss), `"X"`, `"y"` | `EDAReport` |
| `tune_model_tool` | `"X"`, `"y"` | — | `TuningResult` |
| `train_and_evaluate_tool` | `"X"`, `"y"`; optional `tuning_result` arg skips re-tuning entirely | — | `EvaluationReport` |

`PipelineBlockedError(stage, reason)` — unchanged, still defined once in
`data_tools.py` and reused by all four tools whenever a stage's
`blocking_issue` is set.

**NEW: `tools/data_peek_tools.py`** — `peek_dataset_tool(file_path) ->
DatasetPeek`. Deliberately independent of `load_data()`/`ProblemSpec` (peeking
happens *before* a spec exists — there's nothing to route through yet). Reads
only schema (`dtype`, `missing_pct`, `unique_count`, up to 5 example values per
column), never full rows — same "LLM never touches raw data" boundary as
`DataProfile`/`EDAReport`, just applied one step earlier. Raises `ValueError`
for anything other than `.csv`/`.parquet`.

Known dead import, unchanged from before: `feature_tools.py` still imports
`pandas as pd` without using it.

---

## 6. The agents (`agents/`)

All five still share the core shape: `load_dotenv()` at import time, an
`OpenAI()` client, `model="gpt-4o-mini"`, a narrow private `_Judgment` schema
(verdict fields only, never the report itself), and the public `*AgentResult`
re-attaching the real report. **What's new**: `DataAgent`, `FeatureAgent`,
`TuningAgent`, `TrainingAgent` now each expose a `.run(...)` method (not just a
`__main__` demo) taking `pipeline_run: PipelineRun | None = None, cache:
RunCache | None = None` — this is exactly the signature `run_pipeline.py`
calls. All LLM calls go through `observability.timed_llm_call()`, not
`client.beta.chat.completions.parse()` directly.

Per-agent behavior on a successful `.run()` call (all four follow this
pattern): call the tool (cache/run_id passed through) → judge via
`timed_llm_call()` → build the `*AgentResult` → **if `pipeline_run` was
given**, attach the real report to the matching `PipelineRun` field, advance
`pipeline_run.status`, `.log(...)`, and `.record_call(trace_entry)`.

| Agent | Wraps | Result type | `pipeline_run` side effects on success | What the LLM judges |
|---|---|---|---|---|
| `DataAgent` | `profile_dataset_tool` | `DataAgentResult` | `.data_profile`, `status="data_ready"` | Data quality; "proceed" and "no concerns" are explicitly not the same thing |
| `FeatureAgent` | `engineer_features_tool` | `FeatureAgentResult` | `.eda_report`, `status="features_ready"` | Leakage is the top priority — `leakage_warnings` is the system prompt's stated **sole authority**; the model is explicitly told not to second-guess it off raw `correlation_summary` values alone |
| `TuningAgent` | `tune_model_tool` | `TuningAgentResult` | `.tuning_result`, `status="tuned"` — **this write is what lets TrainingAgent skip re-tuning** | Weighs `convergence_notes` against budget used; told explicitly not to speculate about search-space bounds it can't see |
| `TrainingAgent` | `train_and_evaluate_tool` (passes prior `tuning_result` from `pipeline_run` if present) | `TrainingAgentResult` | `.evaluation_report`, `status="evaluated"` | Advisory only — `agrees_with_rule_based_recommendation`, `None` on a pass, never invents a competing `recommended_next_step` |
| `RequirementAgent` | `peek_dataset_tool` (only when `file_path` given) + text-in/spec-out | `ProblemSpec \| ClarificationNeeded` | none (runs before a `PipelineRun` exists) | Extracts spec fields; grounded against real columns when a file is attached |

On a `PipelineBlockedError` from the tool, all four now set
`pipeline_run.status = "escalated"` and log the reason — this is the fix for a
previously-present typo bug (the module comments literally say `# FIX: typo`
next to this line in `data_agent.py`) and a bug where the result object wasn't
named before being returned, which would have made the state-update code
unreachable (comment: `# FIX: name the result BEFORE returning`).

### `RequirementAgent` — significantly upgraded
Still the odd one out (no tool wrapping a stage — text/file in, `ProblemSpec`
out), but now handles **two genuinely different prompt paths** (not one
conditional string, by design — easier to verify each independently):

1. **No file** (`file_path=None`): unchanged from before — the LLM must also
   determine `data_source`, and is told explicitly not to guess a
   plausible-sounding value or target_column.
2. **File attached**: `peek_dataset_tool(file_path)` runs first,
   deterministically, no LLM involved. The LLM is shown the file's *actual*
   schema and told `target_column` **must** exactly match a real column name —
   if it can't confidently identify one, it must add `target_column` to
   `unclear_or_missing` rather than guess. `data_source` is **not** something
   the LLM produces in this path at all — Python already knows the real path
   (`resolved_data_source = file_path`), which eliminates an entire failure
   class (a paraphrased `data_source` not matching `load_data()`'s literal
   string requirement) rather than just detecting it after the fact.

A Python-side backstop (`extracted.target_column not in real_columns`) still
catches a model that invents a column name anyway. **Ordering matters and is
commented as deliberate**: this backstop check runs *above* the
`unclear_or_missing` branch, because both return `ClarificationNeeded` and
whichever runs first wins — if a model both invents a bad column name AND
volunteers its own vaguer `clarifying_question`, the precise deterministic
message (naming the bad column and listing the real ones) must win, not the
model's own commentary about its own mistake.

The four-step validation ladder from the previous analysis
(unclear_or_missing → missing required field → invalid `data_source` →
`ProblemSpec(...)` inside `try/except ValidationError`) is otherwise unchanged,
just now gated by the file/no-file branch for the `data_source` step.

---

## 7. The API layer (`api/`)

FastAPI app (`api/main.py`), CORS restricted to local Vite dev origins
(`localhost:5173`, `127.0.0.1:5173` — **not deployment-ready as-is**, flagged
in the file's own comments). Mounts two routers.

### `api/routes/runs.py` — the one actually wired up (`api/main.py` imports
`runs`, not `run`)
- `POST /runs` — body is `RunCreateRequest`: **exactly one** of `problem_spec`
  (structured) or `context` (free text, optionally with `file_path`) — enforced
  by a `model_validator`. If `context` is given, runs `RequirementAgent`
  synchronously in the request; a `ClarificationNeeded` result returns a
  `ClarificationResponse` (HTTP 200, not an error) instead of starting a run.
  Otherwise: generates a `run_id`, immediately `persist_run_state()`s a fresh
  `PipelineRun` (so `GET /runs/{id}` can find it right away, before the
  background task has done anything), and schedules
  `_execute_run_in_background()` via FastAPI's `BackgroundTasks`. Returns
  `RunCreatedResponse{run_id, status="queued"}` immediately — the actual
  pipeline run happens after the HTTP response.
- `_execute_run_in_background(run_id, spec)`: opens its own DB session (a
  `BackgroundTasks` callback runs outside the request's session lifecycle),
  calls `run_pipeline.run(spec, run_id=run_id, on_update=lambda pr:
  persist_run_state(db, pr))`, and on any uncaught exception marks the run row
  `status="escalated"` directly rather than leaving it stuck at whatever status
  it last reached.
- `GET /runs/{run_id}` — `load_run_state()` then `RunDetail.from_pipeline_run()`
  (404 if not found). This is what the frontend's live-run polling hits.
- `GET /runs` — lightweight `RunSummary` list (no nested reports — a detail
  query per row "would get slow fast once there's real run history," per the
  file's own comment), ordered newest-first.

### `api/routes/uploads.py`
- `POST /uploads` — accepts `.csv`/`.parquet` only (400 otherwise, with the
  received filename echoed back — deliberately, since an empty filename
  usually means the *client* didn't send one, e.g. PowerShell's `curl` alias
  for `Invoke-WebRequest`, not a genuinely wrong file type). Saves under a
  random UUID filename (blocks path traversal via a crafted name, avoids
  collisions) in `uploads/`, streamed in 1MB chunks with a 50MB hard cap.
  Returns `{file_path, filename, size_bytes}` — `file_path` is fed straight
  into `RunCreateRequest.file_path` / `RequirementAgent.run(file_path=...)`,
  and from there straight into `ProblemSpec.data_source` unchanged (no
  `"upload:<id>"` indirection scheme — `load_data()` already accepts any real
  `.csv`/`.parquet` path).

### `api/schemas_api.py`
Deliberately its own top-level request/response types (`RunCreateRequest`,
`ClarificationResponse`, `RunCreatedResponse`, `UploadResponse`, `RunSummary`,
`RunDetail`), **not** a re-export of `schemas/pipeline_run.py`'s `PipelineRun` —
the API's own shape shouldn't silently change every time an internal schema
changes for pipeline-fix reasons (the file's docstring cites `converged` and
`data_quality_warnings` as examples of internal-only-motivated field changes).
Nested fields *within* `RunDetail` (DataProfile, EDAReport, etc.) do reuse the
real internal Pydantic types directly, rather than being re-declared —
duplicating every nested field just to keep them "API-owned" was judged pure
busywork; FastAPI also gets accurate OpenAPI docs for free this way. The
asymmetry (top level separate, nested levels shared) is called out as
deliberate in the file's docstring.

### Known issue: `api/routes/run.py` is dead code
`api/routes/run.py` and `api/routes/runs.py` are near-duplicate files (only
whitespace/formatting differs — confirmed via diff). `api/main.py` imports only
`runs` (`from api.routes import runs, uploads`); `run.py` is never mounted and
has no effect at runtime. Likely a leftover from a rename. Safe to delete, or
worth understanding why it still exists before doing so.

---

## 8. The persistence layer (`db/`, `alembic/`)

**Fully normalized relational schema** — every list-typed or dict-typed field
on a Pydantic report gets its own child table (`ColumnSchemaRecord`,
`CleaningActionRecord`, `ClassBalanceEntryRecord`, `FeatureCorrelationRecord`,
`FeatureDecisionRecord` — one table with a `kind` discriminator covering both
`engineered_features` and `dropped_features`, since both are the same
Pydantic `FeatureDecision` type — `TuningBestParamRecord`, `TrialRecordRow` +
`TrialParamRecord` nested one level deeper, `TestMetricEntryRecord`,
`ConfusionMatrixCellRecord` (row/col/count — 2D matrix flattened to rows),
`ResidualSummaryEntryRecord`, `FailureAnalysisRecord`, `HistoryEntryRecord`,
`TraceEntryRecord`, etc.) All hang off one root `RunRecord` (`__tablename__ =
"runs"`) via `run_id`, with `cascade="all, delete-orphan"` so deleting a run
deletes its whole tree. Class naming convention: ORM classes are suffixed
`Record` (or `Row` in the one case — `TrialRecordRow` — where `Record` would
collide with the Pydantic `TrialRecord` of the same concept) specifically to
coexist with same-named Pydantic classes in the same file (`db/sync.py`).

### `db/sync.py` — the Pydantic ⇄ ORM bridge
- `persist_run_state(session, pipeline_run)` — **write**, full-replace
  strategy: every stage report currently present on the Pydantic object has
  its old child rows deleted and new ones inserted (leaning on
  `cascade="all, delete-orphan"` firing when a relationship attribute is
  reassigned), rather than diffing old vs new. Simpler and more robust; the
  honest cost is that child row ids churn on every sync — acceptable since
  nothing external references those ids, only `run_id`. `ProblemSpec` is
  resynced unconditionally every call (unlike the other four reports, gated
  behind `is not None`) because it can genuinely change mid-run — e.g.
  `constraints.max_tuning_trials` growing on an `expand_hyperparam_search`
  loop-back. `history`/`trace` are reassigned wholesale each call too (some
  redundant delete+insert of already-synced entries, judged negligible at this
  project's actual scale — tens of entries per run, not thousands).
- `load_run_state(session, run_id) -> PipelineRun | None` — **read**, the exact
  reverse translation, including reassembling `class_balance`/`test_metrics`
  dicts from row-lists and the 2D `confusion_matrix` from individual
  `(row_idx, col_idx, count)` cell rows. Returns `None` for a missing run
  (mirrors `dict.get()`) rather than raising — "not found" is a routine,
  expected outcome for an API 404, not exceptional.
- **Known, explicitly flagged limitation**: `best_params`/`TrialRecord.params`
  are `dict[str, Any]` on the Pydantic side but stored as `str(value)` per row
  (a relational column can't type-vary per row), best-effort re-parsed as
  int/float on read via `_parse_param_value()`. Lossy only for a theoretically
  string-valued hyperparameter, which never occurs in this project's actual
  RandomForest search space (`n_estimators`/`max_depth`/`min_samples_split` are
  always int).
- **Known, explicitly flagged performance note**: `load_run_state()` does not
  use `selectinload`/`joinedload` — every relationship access is its own
  lazy-loaded query (N+1). Judged acceptable for one-run-detail-at-a-time
  reads; would need fixing before listing many runs' full detail at once.

### `db/session.py`
`engine = create_engine(settings.database_url)`, `SessionLocal` with
`expire_on_commit=False` (load-bearing for `load_run_state()` — without it,
every attribute would need re-fetching immediately after `session.commit()`).
`get_db()` is the FastAPI dependency (`yield`/`finally` close pattern).
`init_db()` is explicitly **dev-only** — `Base.metadata.create_all()`, no
migration history, no ability to alter existing columns; the docstring says
plainly that Alembic is the real source of truth for schema changes once it's
in place, which it now is (`alembic/versions/f6f4f15a97e1_initial_schema.py`,
one migration so far).

### `core/config.py`
`Settings(BaseSettings)`: `database_url` (default
`postgresql+psycopg://postgres:root%40123@localhost:5432/ml_pipeline` — **a
literal default password checked into source**, worth rotating/removing before
this is ever shared or deployed anywhere) and `openai_api_key`, loaded from
`.env` via `pydantic-settings`.

---

## 9. The frontends

### `frontend/` — React + Vite (primary UI)
React 19, TypeScript, Tailwind 4, `react-router-dom` 7, `recharts` (no
react-query/axios — API calls and polling are hand-rolled in `src/hooks/`).

Pages (`src/pages/`): `Overview`, `NewRun` (structured spec, free-text+file
upload, or presumably both — this is where `POST /uploads` then `POST /runs`
get called), `LiveRun` (polls `GET /runs/{id}` — `useRunPolling.ts` — to show a
run's live status/stage progress), `RunHistory` (`GET /runs` list),
`Performance`, `LlmUsage` (surfaces `PipelineRun.total_cost_usd`/
`total_tokens`/per-call `trace`), `Architecture`, `Settings`.

Components are organized by concern: `components/pipeline/` (stage rows,
per-report views for each of the four stage schemas, a loop-back banner, an
agent judgment panel, a history timeline, a `TuningChart`), `components/ui/`
(generic table/panel/badge/file-upload primitives), `components/layout/`
(shell/sidebar/topbar/logo). `src/lib/` holds pure helper modules
(`stages.ts`, `status.ts`, `routing.ts`, `judgment.ts`, `history.ts`,
`format.ts`) rather than putting that logic in components.

### `streamlit_app.py` — secondary/test-harness UI
663 lines, single file. Per `requirements.txt`'s own comment: "talks to the API
over HTTP only; it does not import the pipeline, except for the metric-name
constants in `schemas/problem_spec.py` so the form's dropdowns cannot drift out
of sync with `ProblemSpec`'s own validator." Functions as a lighter-weight
alternative/testing surface alongside the React app, not a replacement for it.

---

## 10. Known bugs and sharp edges

Confirmed by reading the code, not inferred. Several items from the previous
analysis are now fixed and are listed here only as confirmations (so this
section stays a reliable "what's actually wrong today" list, not stale
history).

1. **`api/routes/run.py` is dead, near-duplicate code** — see §7. Not imported
   anywhere; `runs.py` is the live file. Confusing to future editors who might
   edit the wrong one.

2. **`core/config.py`'s default `database_url` embeds a literal password**
   (`root%40123`, i.e. `root@123` URL-encoded). Low risk as long as this stays
   local-dev-only and `.env` overrides it in any real environment, but worth
   removing the credential from the default rather than relying on every
   environment remembering to override it.

3. **`RequirementAgent`'s LLM call cost/latency is untracked.** It calls
   `timed_llm_call()` like every other agent, but has no `PipelineRun` to
   attach the resulting `TraceEntry` to (it runs *before* one exists) — so
   `PipelineRun.total_cost_usd`/`total_tokens` and the frontend's `LlmUsage`
   page under-report actual spend by however many requirement-gathering calls
   a run took to clarify. Explicitly flagged as an open item in the agent's own
   comments, not silently missed.

4. **CORS is hardcoded to local Vite dev origins only** (`api/main.py`) — the
   file's own comment says this needs to become env-driven before any real
   deployment. Not a bug for local dev, but will silently block the frontend
   the moment either is deployed to a non-localhost origin.

5. **`bad_spec`** is a valid `RecommendedNextStep` but is still never produced
   by `training_stage.py` — only three of the four verbs are reachable (carried
   over from the previous analysis, unchanged).

6. **Dead import**: `tools/feature_tools.py` still imports `pandas as pd`
   without using it.

7. **No `__init__.py`** in `agents/`, `tools/`, `pipeline/` — imports are
   absolute (`from pipeline.data_stage import ...`), relying on implicit
   namespace packages. Everything must still be run from the project root.

8. **`.env` and `uploads/`/`sample_data/` outputs sit on disk** — confirm
   `.gitignore` covers `.env`, `.venv/`, and `uploads/*` (uploaded user files
   and generated parquet artifacts) before any commit, given this is now a real
   git repo (it was not, per the previous analysis) with a live git history.

Fixed since the previous analysis (kept here briefly as confirmation, not as
open items):
- `data_stage.py`'s `column_schema` indentation bug (§4) — fixed.
- `mean_squared_error(squared=False)` crash on sklearn 1.6+ — fixed
  (`root_mean_squared_error()` used).
- `run_pipeline.py`'s `"feature_ready"` vs `RunStatus`'s declared
  `"features_ready"` typo — not present in the current file; current code uses
  the correct `"features_ready"` string (confirmed via `feature_agent.py`'s
  `pipeline_run.status = "features_ready"`).
- The two-different-meanings-of-"converged" bug, and `TuningResult.search_history`
  being computed then dropped before reaching the caller — both fixed and
  confirmed still fixed (`converged=plateaued`, `search_history=history` both
  present in `tuning_stage.py`'s return).
- `metric_value`'s silent-fallback bug — fixed, still an explicit raise on a
  missing key.
- CSV/Parquet loading, categorical encoding, datetime decomposition — all
  implemented (§4), closing the single biggest functional gap flagged
  previously.
- The `agents/`↔`run_pipeline.py` non-integration gap and the "every tool
  recomputes the full chain from scratch" cost problem — both closed via
  `RunCache` + the agents' `.run()` methods (§4, §5).

---

## 11. Running things

Always from the project root, always with the venv interpreter.

**Backend/pipeline:**
```bash
.venv/Scripts/python.exe -m pipeline.run_pipeline       # full deterministic+agent run, needs OPENAI_API_KEY
uvicorn api.main:app --reload                            # FastAPI dev server (per api/main.py's own docstring)
```

**Frontend:**
```bash
cd frontend && npm run dev     # Vite dev server, expects the API at the CORS-allowed localhost origin
```

**Streamlit alt UI:**
```bash
streamlit run streamlit_app.py   # talks to the FastAPI backend over HTTP; start the API first
```

**Tests** — `Test/` is now a **real pytest suite** (a change from the previous
analysis, which described `pipeline/Test/*.py` as ad-hoc print scripts; that
directory no longer exists):
```bash
.venv/Scripts/python.exe -m pytest Test/unit          # deterministic stage/tool tests, no API calls
.venv/Scripts/python.exe -m pytest Test/integration    # agent tests — spend real OpenAI credits; also test_db_sync.py, test_run_pipeline.py
```
`Test/debug_scripts/` holds ad-hoc, non-pytest print-and-eyeball scripts
(`feature_nondeterminism.py`, `inspect_eda_report.py`,
`intake_from_uploaded_file.py`, `loopback_manual.py`) — same role
`pipeline/Test/` used to play, just relocated and narrowed now that the real
suite covers the systematic cases.

**Migrations:**
```bash
alembic upgrade head   # apply db/models.py's schema (currently one migration: f6f4f15a97e1)
```

Environment: Python 3.13.14. `requirements.txt` now pins real sections (core
pipeline / API layer / persistence / Streamlit UI / dev-test) with inline
comments explaining *why* each dependency is needed (e.g. `python-multipart`
for `UploadFile` parsing, `pyarrow` as pandas' parquet engine, `psycopg[binary]`
specifically because `postgresql+psycopg://` resolves to psycopg **3**, not
psycopg2) — still no exact version pins beyond `pydantic>=2.0`.

---

## 12. Conventions to follow when editing

Unchanged from the previous analysis, still the load-bearing rules:

- **Never let the LLM compute or route.** A new agent = deterministic tool call
  + narrow `_Judgment` schema + re-attach the real report.
- **Stage functions take the whole `ProblemSpec`**, not loose strings.
- **Tools take flat primitives** (what LLM function-calling can populate) plus
  now, optionally, `cache`/`run_id` — always optional, always safe to omit,
  always must degrade to a correct (if wasteful) recompute on a miss.
- **Reports are Pydantic; data is a DataFrame.** Never wrap a DataFrame in a
  model — this is also why `RunCache` exists as a separate, deliberately
  unstructured store rather than growing new fields on `PipelineRun`.
- **Flag, don't silently drop, anything a human should see** — the leakage
  branch in `feature_stage.py` remains the canonical example; the newer
  `data_quality_warnings` field for NaT-handling follows the same instinct for
  a different kind of risk ("will this crash" vs "can this be trusted").
- Module-level constants hold every threshold; casing is inconsistent
  (`Missing_blocking_threshold` vs `HIGHER_IS_BETTER`) — match the file you're
  in rather than mass-renaming.
- **New pattern established in this phase**: when adding a cross-cutting
  concern that every agent needs (cost tracking, caching), add ONE shared
  module (`observability.py`, `run_cache.py`) and thread it through via
  optional constructor/method parameters, rather than letting each agent
  reimplement its own version. Keep every such parameter optional and
  behavior-preserving when omitted.
- Comments here explain *why*, often referencing earlier design decisions or
  bugs that motivated a change (search for `# FIX:` and `NEW —` for recent,
  deliberate call-outs). Keep that voice — it's how this file gets kept honest.

---

## 13. Roadmap the code still anticipates

With the orchestrator/agent wiring and caching layer now built, the remaining
flagged gaps are narrower:

- Wire `RequirementAgent`'s pre-run `TraceEntry` into cost tracking somehow
  (§6/§10 item 3) — needs a decision on where a pre-run trace should live.
- Make CORS origins env-driven before any real deployment (§10 item 4).
- Remove the dead `api/routes/run.py` file, or understand why it's still there
  (§10 item 1).
- `bad_spec` remains an unreachable `RecommendedNextStep` — no code path in
  `training_stage.py` ever produces it.
- `load_run_state()`'s N+1 query pattern (§8) would need `selectinload`/
  `joinedload` before a "list many runs' full detail" use case appears.
- Target/hash encoding for high-cardinality categorical columns (currently
  dropped outright above `MAX_CATEGORICAL_CARDINALITY = 20`).
- A real convergence-detection upgrade and LLM-authored `failure_analysis`
  (replacing the rule ladder) are still only docstring-level aspirations in
  `tuning_agent.py`/`training_stage.py`, not implemented.
