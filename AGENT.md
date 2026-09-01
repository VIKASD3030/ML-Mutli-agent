# AGENT.md — Codebase Map & Deep Analysis

> Purpose: a durable, pre-computed analysis of this repo so an agent (or a human)
> can act immediately without re-reading every file. Written 2026-08-21 against
> the tree as it stands. If you change architecture, update this file.

---

## 1. What this project is

An **LLM-orchestrated ML pipeline agent**. A user's free-text request
("predict tumor malignancy, aim for 0.9 f1") is turned into a validated
`ProblemSpec`, then pushed through five stages: data profiling → feature
engineering/EDA → hyperparameter tuning → training/evaluation → pass/fail with
loop-back routing.

The organizing principle of the whole codebase, and the thing to preserve when
editing:

> **Deterministic Python does all computation and all routing.
> The LLM only reads structured reports and produces a judgment.**

The LLM never sees raw data, never builds a model, never decides where control
flows. Every agent hands the model a Pydantic object serialized with
`model_dump_json()` and receives back a narrow structured verdict.

---

## 2. Layout

```
schemas/        Pydantic hand-off objects (the contracts between stages)
pipeline/       Phase 1: deterministic stage functions + the orchestrator
tools/          Phase 2: flat-argument, LLM-callable wrappers over pipeline/
agents/         Phase 3: LLM agents that call one tool then judge its output
pipeline/Test/  Ad-hoc print-based scripts (NOT pytest — see §8)
.env            OPENAI_API_KEY (loaded via python-dotenv in every agent module)
.venv/          Python 3.13.14 virtualenv
```

Four layers, strictly one-directional:

```
agents/  →  tools/     →  pipeline/   →  schemas/
(LLM)       (flat args,    (pandas /      (pure data
             validation)    sklearn)       contracts)
```

`schemas/` imports nothing from the other three. `pipeline/` imports only
`schemas/`. Never invert this.

---

## 3. The data contracts (`schemas/`)

All re-exported from `schemas/__init__.py`.

| File | Object | Produced by | Key fields / properties |
|---|---|---|---|
| `problem_spec.py` | `ProblemSpec` | RequirementAgent | `task_type`, `target_column`, `success_metric`, `metric_threshold`, `data_source`, `constraints` |
| | `PipelineConstraints` | | `max_tuning_trials=50`, `max_loop_backs=3`, `interpretability_required`, `max_training_seconds` |
| | `ClarificationNeeded` | | returned *instead of* a spec when the request is ambiguous |
| `data_profile.py` | `DataProfile` | `clean_and_profile()` | `column_schema`, `class_balance`, `outlier_flags`, `cleaning_actions_taken`, `warnings`, `blocking_issue`; `.is_usable` |
| `eda_report.py` | `EDAReport` | `engineer_features()` | `correlation_summary`, `engineered_features`, `dropped_features`, **`leakage_warnings`**, `final_feature_names`; `.has_leakage_risk` |
| `tuning_result.py` | `TuningResult` | `tune_hyperparameters()` | `best_params`, `best_cv_score`, `search_budget_used/total`, `convergence_notes`, `converged` (required field) |
| `evaluation_report.py` | `EvaluationReport` | `train_and_evaluate()` | `test_metrics`, `metric_value/threshold`, `pass_fail`, `failure_analysis`; `.passed` |
| `pipeline_run.py` | `PipelineRun` | orchestrator | `run_id`, `status`, `loop_count`, all five reports, `history[]`, `.log(stage, event)` |

Three closed vocabularies worth memorizing:

- **Metrics** — `CLASSIFICATION_METRICS = {accuracy, f1, precision, recall, roc_auc}`,
  `REGRESSION_METRICS = {rmse, mae, r2}`. `ProblemSpec._metric_matches_task`
  (a `field_validator`) rejects a mismatch. Field order matters: the validator
  reads `info.data["task_type"]`, so `task_type` must stay declared before
  `success_metric` — it currently is.
- **Routing verbs** — `RecommendedNextStep = Literal["revisit_features",
  "expand_hyperparam_search", "insufficient_data", "bad_spec"]`. This is the
  only vocabulary the state machine understands.
- **Run statuses** — `created, data_ready, features_ready, tuned, evaluated,
  done, escalated`.

---

## 4. The deterministic stages (`pipeline/`)

### `data_stage.py`
- `load_data(spec) -> DataFrame`. **Only `"builtin:breast_cancer"` is actually
  implemented** — anything else raises `ValueError`. It loads sklearn's
  breast-cancer frame and renames `target` → `spec.target_column`. (Despite
  several docstrings and the RequirementAgent prompt mentioning `.csv`/`.parquet`,
  no file-loading branch exists. This is the single biggest functional gap.)
- `clean_and_profile(df, spec) -> (cleaned_df, DataProfile)`:
  1. target column absent from data → `blocking_issue`, early return
  2. target missingness > `Missing_blocking_threshold` (0.5) → `blocking_issue`
     (processing continues, but the run is doomed)
  3. drop rows with missing target; drop exact duplicate rows
  4. per-column: median-impute numeric non-target columns that have missingness
  5. classification only: `class_balance`; minority share < 10% → warning
  6. numeric non-target: IQR outlier flag (`q1 - 3*iqr`, `q3 + 3*iqr`; flagged if
     >2% of rows fall outside)

### `feature_stage.py`
Thresholds: `Leakage_correlation_threshold = 0.98`,
`Low_signal_correlation_threshold = 0.01`, `Low_variance_threshold = 1e-8`.

Numeric columns only — there is no categorical encoding anywhere. For each
column it computes `abs(corr(col, y))`, coercing NaN to 0.0, then:
- variance < 1e-8 → drop
- corr ≥ 0.98 → **warn but keep** (asymmetric on purpose: leakage is surfaced to
  a human/agent, never silently acted on)
- corr < 0.01 → drop
- else keep

Then one "engineered" feature: the product of the two highest-correlation
survivors, named `"{a}__x__{b}"`. Returns `(X, y, EDAReport)`.

### `tuning_stage.py`
Optuna, `direction="maximize"`, `n_trials = spec.constraints.max_tuning_trials or 25`.
Search space fixed inside `_build_model()`: RandomForest
`n_estimators 50–300`, `max_depth 2–20`, `min_samples_split 2–10`, `random_state=42`.
Scoring is `f1_macro` (classification) or `neg_root_mean_squared_error`
(regression) — negated because `cross_val_score` always maximizes. CV = 3-fold.
Each trial appends a `TrialRecord` immediately, so a mid-search crash still
leaves a usable history.

**Convergence heuristic:** take the last 20% of trials; if their best score
equals the overall best within `1e-6`, call it plateaued. That single `plateaued`
boolean now feeds BOTH the `convergence_notes` text and the `converged` field on
`TuningResult` — one notion of "converged", computed once. (It previously fed
only the text, while `converged` was a separate `used < total` property that
could never be True; see §7.4.)

### `training_stage.py`
80/20 `train_test_split(random_state=42)`, stratified for classification.
Refits a RandomForest on `tuning.best_params` plus `random_state=42`.

- classification → `{"f1", "accuracy", "precision", "recall"}` + `"roc_auc"`
  when `predict_proba` and a binary target allow it + confusion matrix
- regression → `{"rmse", "mae", "r2"}` + residual mean/std

`metric_value = test_metrics[spec.success_metric]` — an **exact lookup** in both
branches, raising `ValueError` (naming the requested metric and the available
ones) if the key is absent. `ProblemSpec`'s validator should make that
unreachable; it is kept because a silent wrong-metric fallback is worse than a
crash. Keys are deliberately named to match `success_metric` exactly.

`HIGHER_IS_BETTER = {accuracy, f1, precision, recall, roc_auc, r2}` picks the
comparison direction. On failure, the **rule-based** `failure_analysis` fires in
this order:
1. `tuning.converged` and `gap > 0.1 * |threshold|` → `revisit_features`
2. `not tuning.converged` → `expand_hyperparam_search`
3. otherwise → `insufficient_data`

### `run_pipeline.py` — the orchestrator / state machine
The data stage runs **once, outside the loop** (re-running it with identical
inputs cannot fix bad data). Unusable data → `status="escalated"`, return.

Then `while True:` over features → tuning → train/eval:
- `eval_report.passed` → `status="done"`, return
- else `loop_count += 1`; if `loop_count >= max_loop_backs` → escalate
- `revisit_features` → `continue`
- `expand_hyperparam_search` → `max_tuning_trials *= 1.5`, `continue`
- anything else (`insufficient_data`, `bad_spec`) → escalate

This `while True` **is** the entire state machine, and it is the only place in
the codebase that reads `recommended_next_step`. Keep it that way.

Verified end-to-end run (`python -m pipeline.run_pipeline`, 15 trials):
`status=done, loops=0, f1=0.9526 vs threshold 0.90, 29 features kept, 0 leakage warnings`.

---

## 5. The tool layer (`tools/`)

Each tool takes **flat primitives** (what an LLM function call can populate),
rebuilds a minimal `ProblemSpec` internally, runs the pipeline prefix it needs,
and returns exactly one report object — never a DataFrame.

| Tool | Runs internally | Returns |
|---|---|---|
| `profile_dataset_tool` | load → clean | `DataProfile` |
| `engineer_features_tool` | load → clean → features | `EDAReport` (X/y discarded) |
| `tune_model_tool` | load → clean → features → tune | `TuningResult` |
| `train_and_evaluate_tool` | **the whole chain**, tuning included | `EvaluationReport` |

`PipelineBlockedError(stage, reason)` is defined once in `data_tools.py` and
reused by all four — a distinct type so agent-level `except` blocks can route to
escalation deliberately instead of swallowing it alongside unrelated bugs. Every
tool raises it when `profile.blocking_issue is not None`.

**No shared state store exists.** Each tool recomputes everything from scratch;
`train_and_evaluate_tool` re-runs the *entire* hyperparameter search on every
call, making it by far the most expensive call in the project. Tools that don't
need a real metric pass dummies (`success_metric="accuracy"`,
`metric_threshold=0.0`); that is safe only because no stage upstream of training
reads those fields.

---

## 6. The agents (`agents/`)

All five share one shape: `load_dotenv()` at import time, an `OpenAI()` client,
`model="gpt-4o-mini"`, `client.beta.chat.completions.parse(...,
response_format=_Judgment)`, and (all but `data_agent.py`) an
`if __name__ == "__main__":` demo block.

The recurring trick: a **narrow private `_Judgment` / `_Judgement` schema**
containing *only* the verdict fields, never the report itself. The LLM
physically cannot return its own version of the `DataProfile`/`EDAReport` — it
can only comment on the real one. That is the "LLM never touches computation"
boundary enforced at the type level rather than by convention. The public
`*AgentResult` then re-attaches the real report alongside the verdict.

| Agent | Wraps | Result type | What the LLM actually does |
|---|---|---|---|
| `DataAgent` | `profile_dataset_tool` | `DataAgentResult(profile, proceed, summary, concerns)` | Judge data quality in plain language; "proceed" and "no concerns" are explicitly not the same thing |
| `FeatureAgent` | `engineer_features_tool` | `FeatureAgentResult(report, proceed, ...)` | **Leakage is the top priority** — a leak is the one failure mode that looks like success |
| `TuningAgent` | `tune_model_tool` | `TuningAgentResult(result, proceed, ...)` | Weigh `convergence_notes` against budget used; a high score from an unconverged search is less trustworthy |
| `TrainingAgent` | `train_and_evaluate_tool` | `TrainingAgentResult(report, summary, agrees_with_rule_based_recommendation, concerns)` | **Advisory only** — sanity-checks the rule-based `recommended_next_step`, never replaces it; the field is `None` on a pass |
| `RequirementAgent` | *(no tool — text in, spec out)* | `ProblemSpec \| ClarificationNeeded` | Extract spec fields from free text |

### `RequirementAgent` is the odd one out — read this before touching it
The LLM fills a loose, **all-Optional `_Extraction`** schema. Python then runs
four checks in order, each returning `ClarificationNeeded` rather than raising:
1. `unclear_or_missing` non-empty → surface the model's own clarifying question
2. any required field still `None` → ask about those
3. `data_source` neither in `SUPPORTED_DATA_SOURCES` nor a `.csv`/`.parquet`
   path → ask
4. finally `ProblemSpec(...)` inside `try/except ValidationError` → a
   metric/task mismatch surfaces as a clarification

Why the ceremony: OpenAI structured outputs enforce **JSON shape only** — types
and enum membership — not Pydantic cross-field validators. The model can legally
propose `task_type="regression", success_metric="f1"`; only step 4 catches it.
`SUPPORTED_DATA_SOURCES = ["builtin:breast_cancer"]` is the single source of
truth, referenced by both the system prompt and the Python check.
`_Extraction.metric_threshold` is typed `str` and relies on Pydantic coercing it
to `float` on the way into `ProblemSpec`.

---

## 7. Known bugs and sharp edges

Confirmed by running the code, not inferred.

1. **`data_stage.py`: `column_schema.append(...)` sits outside the `for col`
   loop** (indentation). Only the *last* column ever gets a `ColumnSchema`.
   Verified: 31 columns in, `len(profile.column_schema) == 1` (`['diagnosis']`),
   while `column_count` correctly reports 31. Nothing downstream reads
   `column_schema`, so the pipeline still passes — but any LLM judging the
   profile is shown a near-empty schema. **Fix: indent the append block one
   level.** The `cleaning_actions.append(...)` call just above is also
   mis-indented relative to its `CleaningAction(` arguments — it parses, but it
   reads wrong.

2. **`training_stage.py`: `mean_squared_error(..., squared=False)` is dead on
   this environment.** The `squared` parameter was removed in scikit-learn 1.6;
   this venv has **1.9.0**, and the installed signature is
   `(y_true, y_pred, *, sample_weight, multioutput)`. Any regression run raises
   `TypeError`. Replace with `root_mean_squared_error(y_test, preds)`. Nobody has
   hit it only because the sole supported dataset is classification.

3. **`run_pipeline.py` sets `status = "feature_ready"`, but `RunStatus` declares
   `"features_ready"`.** Pydantic does not validate on assignment by default, so
   the invalid value is stored silently (verified). Either fix the string or set
   `model_config = ConfigDict(validate_assignment=True)` on `PipelineRun` —
   doing the latter first will immediately surface this.

4. ~~**Two different meanings of "converged".**~~ **FIXED.** `converged` is now
   a required field on `TuningResult`, set from the same `plateaued` boolean that
   writes `convergence_notes`. It was previously a derived property
   (`search_budget_used < search_budget_total`) which, because Optuna always
   spends its full budget, was permanently `False` — making `revisit_features`
   and `insufficient_data` unreachable dead code. Both are now live.

   Residual limitation, NOT a bug: `revisit_features` needs
   `gap > 0.1 * threshold`. On `builtin:breast_cancer` a tuned forest scores
   f1 ~= 0.95, so that would need a threshold above ~1.06 — unreachable for a
   bounded metric. The branch fires correctly on genuinely poor models; see
   `Test/unit/test_training_stage.py::..._on_wide_converged_gap`.

5. ~~**`metric_value` soft fallback**~~ **FIXED.** All classification metrics
   the validator allows are now computed and keyed by their exact names, and the
   lookup raises instead of falling back. `roc_auc` is omitted (not guessed) when
   `predict_proba` is unavailable or the target is not binary.

5b. **`mean_squared_error(squared=False)` was removed in scikit-learn 1.6** and
   this project pins 1.9, so every regression run raised `TypeError` before this
   was switched to `root_mean_squared_error()`. The regression branch had no test
   coverage at all, which is why a total crash went unnoticed. **FIXED**, but
   worth noting regression remains lightly exercised overall.

6. **`bad_spec`** is a valid `RecommendedNextStep` but is never produced by
   `training_stage.py` — only three of the four verbs are reachable.

7. **No `__init__.py`** in `agents/`, `tools/`, `pipeline/`, `pipeline/Test/`.
   Imports are absolute (`from pipeline.data_stage import ...`), so **everything
   must be run from the project root**, relying on implicit namespace packages.

8. **No `.git` directory in this working tree** — no version-control safety net
   for edits.

9. **`.env` sits on disk with a live `OPENAI_API_KEY`** and there is no
   `.gitignore`. If this ever becomes a git repo, ignore `.env` and `.venv/`
   before the first commit. (The key line is written `OPENAI_API_KEY =...` with
   a space; python-dotenv strips it, so it does load correctly.)

10. **Dead imports:** `feature_tools.py` imports `pandas as pd` and never uses
    it; several test scripts import `PipelineBlockedError` unused.

11. **Prompt/behavior drift:** `RequirementAgent` tells the model that
    `.csv`/`.parquet` paths are acceptable, but `load_data()` rejects them. A
    user supplying a CSV gets a valid `ProblemSpec` that then explodes with
    `ValueError` at the data stage.

---

## 8. Running things

Always from the project root, always with the venv interpreter:

```bash
.venv/Scripts/python.exe -m pipeline.run_pipeline       # full deterministic run, no API key needed
.venv/Scripts/python.exe pipeline/Test/test_feature.py  # a stage script
.venv/Scripts/python.exe -m agents.feature_agent        # agent demo (spends API credits)
```

`pipeline/Test/*.py` are **not pytest tests** — no `assert`, no test functions,
no `pytest` in `requirements.txt`. They are print-and-eyeball scripts, and the
agent/tool ones spend real OpenAI credits. Naming convention:
`test_<stage>.py` = deterministic stage, `test_<x>_tools.py` = tool layer,
`test_<x>_agent.py` = LLM layer. There is no `test_requirement_agent.py`.

Environment (verified): Python 3.13.14, pydantic 2.13.4, scikit-learn 1.9.0,
pandas 3.0.5, optuna 4.9.0, openai 2.53.0. `requirements.txt` pins nothing except
`pydantic>=2.0` — the sklearn 1.6 breaking change in §7.2 is a direct
consequence.

---

## 9. Conventions to follow when editing

- **Never let the LLM compute or route.** A new agent = deterministic tool call
  + narrow `_Judgment` schema + re-attach the real report.
- **Stage functions take the whole `ProblemSpec`**, not loose strings, so every
  stage shares one signature and new fields don't ripple through call sites.
- **Tools take flat primitives**, because that is what function-calling can fill.
- **Reports are Pydantic; data is a DataFrame.** Pydantic models describe
  *decisions*, not tabular payloads. Don't wrap a DataFrame in a model.
- **Flag, don't silently drop, anything a human should see** — the leakage
  branch in `feature_stage.py` is the canonical example.
- Module-level constants hold every threshold. Casing is inconsistent
  (`Missing_blocking_threshold` vs `HIGHER_IS_BETTER`) — match the file you're in
  rather than mass-renaming.
- Comments here explain *why*, often referencing earlier design decisions. Keep
  that voice.

---

## 10. Roadmap the code already anticipates

Every "KNOWN LIMITATION" comment points at the same missing piece: **a
`PipelineRun`-backed shared state store**. Once stage outputs are cached on the
run object, tools stop recomputing the prefix chain (§5),
`train_and_evaluate_tool` stops re-running the whole search, and the agents can
be wired into `run_pipeline.py`'s loop instead of running as standalone demos —
which is the one integration that does not yet exist: **`agents/` and
`run_pipeline.py` never call each other.** The orchestrator is purely
deterministic today; the five agents are independent entry points.

Other flagged extensions: real convergence detection plus search bounds carried
on `TuningResult` (`tuning_agent.py` docstring), an LLM-authored
`failure_analysis` replacing the rule ladder (`training_stage.py` docstring),
a Postgres-backed `PipelineRun` (`pipeline_run.py` docstring), categorical
feature encoding, and CSV/parquet loading.
