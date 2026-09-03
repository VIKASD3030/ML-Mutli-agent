"""
streamlit_app.py — a test harness UI for the FastAPI ML pipeline backend.

Run the backend first (uvicorn api.main:app --reload), then:
    streamlit run streamlit_app.py

DESIGN CONSTRAINT WORTH STATING UP FRONT
----------------------------------------
Each agent's own judgment — its `summary` prose, its `concerns` list, its
reasoning for proceed=True/False — is NOT persisted anywhere the API exposes.
The agents return it, the orchestrator uses it, and then it is dropped; only a
bare string like "proceed=True" reaches PipelineRun.history.

So this UI never renders per-agent commentary. Everything shown below comes
from a real field on a real response object. Where a stage's narrative would
naturally go, this app shows the stage's actual structured output instead
(warnings, dropped features, convergence numbers, failure_analysis), which is
genuinely richer than a sentence of prose anyway.

Derived-value note: DataProfile.is_usable, EDAReport.has_leakage_risk /
has_data_quality_risk and EvaluationReport.passed are @property, not fields,
so pydantic does NOT serialize them and they are absent from the JSON. This
app recomputes the equivalents client-side from the underlying fields rather
than reading keys that will never exist.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
import requests
import streamlit as st

# Metric sets are imported from the backend so they cannot drift out of sync
# with ProblemSpec's own validator. The fallback exists only so the file still
# runs if it is ever copied somewhere without the package on sys.path.
try:
    from schemas.problem_spec import CLASSIFICATION_METRICS, REGRESSION_METRICS
except ImportError:  # pragma: no cover - standalone-copy fallback
    CLASSIFICATION_METRICS = {"accuracy", "f1", "precision", "recall", "roc_auc"}
    REGRESSION_METRICS = {"rmse", "mae", "r2"}

DEFAULT_API_URL = "http://127.0.0.1:8000"
TERMINAL_STATUSES = {"done", "escalated"}
REFRESH_SECONDS = 2
REQUEST_TIMEOUT = 15

st.set_page_config(page_title="ML Pipeline Agent", page_icon="🧪", layout="wide")


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

def _friendly_error(exc: Exception, url: str) -> str:
    """Turn a requests exception into something a human can act on. A raw
    ConnectionError traceback tells a user nothing about the one thing that is
    almost always wrong: the backend is not running."""
    if isinstance(exc, requests.exceptions.ConnectionError):
        return f"Could not connect to the API at {url} — is it running? Start it with:  uvicorn api.main:app --reload"
    if isinstance(exc, requests.exceptions.Timeout):
        return f"The API at {url} did not respond within {REQUEST_TIMEOUT}s."
    return f"Request to {url} failed: {type(exc).__name__}: {exc}"


def api_request(
    method: str,
    base_url: str,
    path: str,
    *,
    json_body: Optional[dict] = None,
    files: Optional[dict] = None,
) -> tuple[Optional[Any], Optional[str]]:
    """Single choke point for every HTTP call. Returns (data, error_message);
    exactly one is non-None. Returning errors instead of raising keeps every
    call site free of try/except and guarantees no raw traceback reaches the UI."""
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method, url, json=json_body, files=files, timeout=REQUEST_TIMEOUT
        )
    except Exception as exc:
        return None, _friendly_error(exc, url)

    if response.status_code >= 400:
        # FastAPI puts human-readable messages in "detail"; fall back to the
        # raw body for anything that isn't a FastAPI-shaped error.
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"HTTP {response.status_code}: {detail}"

    try:
        return response.json(), None
    except ValueError:
        return None, f"API returned non-JSON response: {response.text[:200]}"


def upload_file(base_url: str, uploaded) -> tuple[Optional[dict], Optional[str]]:
    """POST /uploads. The multipart field name MUST be "file" — that is the
    parameter name on upload_dataset(file: UploadFile) in api/routes/uploads.py,
    and FastAPI matches the form field to it by name."""
    files = {"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")}
    return api_request("POST", base_url, "/uploads", files=files)


# ---------------------------------------------------------------------------
# Small formatting helpers
# ---------------------------------------------------------------------------

def _fmt_money(value: float) -> str:
    return f"${value:.6f}"


def _decisions_table(decisions: list[dict]) -> pd.DataFrame:
    """FeatureDecision list -> DataFrame. Fields are exactly name + rationale."""
    return pd.DataFrame(
        [{"feature": d["name"], "rationale": d["rationale"]} for d in decisions]
    )


def _bullet_list(items: list[str]) -> None:
    for item in items:
        st.markdown(f"- {item}")


def _stage_state(detail: dict, report_key: str) -> str:
    """A stage is 'done' when its report object is present on the run.

    Inferred from the report objects themselves because there is no per-stage
    status field on PipelineRun to read. Note this means 'done' really means
    'has completed at least once' — on a loop-back the reports are overwritten
    in place, so an earlier stage re-running does not revert to pending."""
    return "done" if detail.get(report_key) else "pending"


# ---------------------------------------------------------------------------
# Stage renderers — every field below is real; see module docstring
# ---------------------------------------------------------------------------

def render_data_profile(profile: dict) -> None:
    if profile.get("blocking_issue"):
        st.error(f"Blocking issue: {profile['blocking_issue']}")
    else:
        # is_usable is a @property and is not serialized; this is the
        # equivalent condition, computed from the field that is.
        st.success("Data is usable (no blocking issue).")

    cols = st.columns(3)
    cols[0].metric("Rows", f"{profile['row_count']:,}")
    cols[1].metric("Columns", profile["column_count"])
    cols[2].metric("Cleaning actions", len(profile.get("cleaning_actions_taken", [])))

    if profile.get("class_balance"):
        st.markdown("**Class balance**")
        st.dataframe(
            pd.DataFrame(
                [{"class": k, "proportion": v} for k, v in profile["class_balance"].items()]
            ),
            hide_index=True,
            width="stretch",
        )

    if profile.get("warnings"):
        st.markdown("**Warnings**")
        _bullet_list(profile["warnings"])

    if profile.get("outlier_flags"):
        st.markdown(f"**Outlier-flagged columns:** {', '.join(profile['outlier_flags'])}")

    actions = profile.get("cleaning_actions_taken", [])
    if actions:
        st.markdown("**Cleaning actions taken**")
        st.dataframe(pd.DataFrame(actions)[["column", "action", "rationale"]],
                     hide_index=True, width="stretch")

    schema = profile.get("column_schema", [])
    if schema:
        with st.expander(f"Column schema ({len(schema)} columns)"):
            st.dataframe(pd.DataFrame(schema), hide_index=True, width="stretch")


def render_eda_report(report: dict) -> None:
    leakage = report.get("leakage_warnings", [])
    quality = report.get("data_quality_warnings", [])

    if leakage:
        st.error(f"Leakage risk — {len(leakage)} warning(s)")
        _bullet_list(leakage)
    if quality:
        st.warning(f"Data quality — {len(quality)} warning(s)")
        _bullet_list(quality)
    if not leakage and not quality:
        st.success("No leakage or data-quality warnings.")

    kept = report.get("final_feature_names", [])
    cols = st.columns(3)
    cols[0].metric("Final features", len(kept))
    cols[1].metric("Engineered", len(report.get("engineered_features", [])))
    cols[2].metric("Dropped", len(report.get("dropped_features", [])))

    if report.get("target_distribution_notes"):
        st.caption(report["target_distribution_notes"])

    if report.get("engineered_features"):
        with st.expander("Engineered features"):
            st.dataframe(_decisions_table(report["engineered_features"]),
                         hide_index=True, width="stretch")
    if report.get("dropped_features"):
        with st.expander("Dropped features"):
            st.dataframe(_decisions_table(report["dropped_features"]),
                         hide_index=True, width="stretch")
    if kept:
        with st.expander(f"Final feature names ({len(kept)})"):
            st.write(", ".join(kept))
    if report.get("correlation_summary"):
        with st.expander("Correlation with target"):
            corr = pd.DataFrame(
                [{"feature": k, "abs_correlation": v}
                 for k, v in report["correlation_summary"].items()]
            ).sort_values("abs_correlation", ascending=False)
            st.dataframe(corr, hide_index=True, width="stretch")


def render_tuning_result(result: dict) -> None:
    cols = st.columns(3)
    cols[0].metric("Best CV score", f"{result['best_cv_score']:.4f}")
    cols[1].metric("Trials used",
                   f"{result['search_budget_used']} / {result['search_budget_total']}")
    cols[2].metric("Converged", "yes" if result["converged"] else "no")

    if result.get("convergence_notes"):
        st.caption(result["convergence_notes"])
    st.markdown(f"**Metric optimized:** `{result['metric_optimized']}`")

    if result.get("best_params"):
        st.markdown("**Best parameters**")
        st.dataframe(
            pd.DataFrame([{"parameter": k, "value": v}
                          for k, v in result["best_params"].items()]),
            hide_index=True, width="stretch",
        )

    history = result.get("search_history", [])
    if history:
        with st.expander(f"Search history ({len(history)} trials)"):
            st.dataframe(
                pd.DataFrame([{"trial": t["trial_number"], "score": t["score"], **t["params"]}
                              for t in history]),
                hide_index=True, width="stretch",
            )


def render_evaluation_report(report: dict) -> None:
    passed = report["pass_fail"] == "pass"  # .passed is a @property, not serialized
    threshold = report["metric_threshold"]
    value = report["metric_value"]

    if passed:
        st.success(f"PASS — {report['metric_optimized']} = {value:.4f} (threshold {threshold})")
    else:
        st.error(f"FAIL — {report['metric_optimized']} = {value:.4f} (threshold {threshold})")

    cols = st.columns(3)
    cols[0].metric(report["metric_optimized"], f"{value:.4f}", delta=f"{value - threshold:+.4f}")
    cols[1].metric("Threshold", f"{threshold}")
    cols[2].metric("Result", report["pass_fail"].upper())

    analysis = report.get("failure_analysis")
    if analysis:
        st.warning(f"**Recommended next step:** `{analysis['recommended_next_step']}`")
        st.markdown(analysis["summary"])

    if report.get("test_metrics"):
        st.markdown("**All test metrics**")
        st.dataframe(
            pd.DataFrame([{"metric": k, "value": v} for k, v in report["test_metrics"].items()]),
            hide_index=True, width="stretch",
        )

    if report.get("confusion_matrix"):
        with st.expander("Confusion matrix"):
            st.dataframe(pd.DataFrame(report["confusion_matrix"]), width="stretch")
    if report.get("residual_summary"):
        with st.expander("Residual summary"):
            st.json(report["residual_summary"])


# ---------------------------------------------------------------------------
# Run detail view (shared by Live Run and Run History)
# ---------------------------------------------------------------------------

STAGES = [
    ("Data", "data_profile", render_data_profile),
    ("Feature", "eda_report", render_eda_report),
    ("Tuning", "tuning_result", render_tuning_result),
    ("Training", "evaluation_report", render_evaluation_report),
]


def _escalation_reason(detail: dict) -> Optional[str]:
    """Why did this run escalate? Prefer the rule-based failure_analysis; fall
    back to the most recent history entry mentioning escalation, which is where
    the orchestrator and agents both write their reason."""
    report = detail.get("evaluation_report")
    if report and report.get("failure_analysis"):
        analysis = report["failure_analysis"]
        return f"{analysis['summary']} (next step: {analysis['recommended_next_step']})"
    for entry in reversed(detail.get("history", [])):
        if "escalat" in entry["event"].lower():
            return entry["event"]
    return None


def render_run_detail(detail: dict) -> None:
    spec = detail["problem_spec"]
    status = detail["status"]
    max_loops = spec["constraints"]["max_loop_backs"]

    header = st.columns(4)
    header[0].metric("Status", status)
    header[1].metric("Loop-backs", f"{detail['loop_count']} / {max_loops}")
    header[2].metric("Total cost", _fmt_money(detail["total_cost_usd"]))
    header[3].metric("Total tokens", f"{detail['total_tokens']:,}")

    if status == "done":
        st.success("Run complete.")
    elif status == "escalated":
        reason = _escalation_reason(detail)
        st.error("Run escalated to a human.")
        if reason:
            st.markdown(f"**Why:** {reason}")
    else:
        st.info(f"Run in progress — status `{status}`.")

    with st.expander("Problem spec"):
        st.json(spec)

    # --- stage progress, inferred from which reports exist ---
    st.subheader("Stages")
    progress = st.columns(len(STAGES))
    for col, (label, key, _) in zip(progress, STAGES):
        state = _stage_state(detail, key)
        col.markdown(f"**{label}**\n\n{'✅ done' if state == 'done' else '⏳ pending'}")
    st.caption(
        "Stage state is inferred from whether that stage's report object exists on "
        "the run — there is no per-stage status field. A stage that re-runs during a "
        "loop-back overwrites its report in place, so it stays 'done'."
    )

    for label, key, renderer in STAGES:
        report = detail.get(key)
        if not report:
            continue
        with st.expander(f"{label} stage", expanded=(label == "Training")):
            renderer(report)

    # --- trace ---
    st.subheader("LLM call trace")
    trace = detail.get("trace", [])
    if trace:
        st.dataframe(
            pd.DataFrame(trace)[
                ["timestamp", "agent", "model", "prompt_tokens", "completion_tokens",
                 "total_tokens", "latency_ms", "estimated_cost_usd"]
            ],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("No LLM calls recorded yet.")

    # --- history ---
    st.subheader("History log")
    history = detail.get("history", [])
    if history:
        st.dataframe(
            pd.DataFrame(history)[["timestamp", "stage", "event"]],
            hide_index=True, width="stretch",
        )
    else:
        st.caption("No history entries yet.")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def page_new_run(base_url: str) -> None:
    st.title("New run")

    mode = st.radio("Input mode", ["Structured spec", "Describe in plain English"],
                    horizontal=True)

    # A clarification response is stored in session state so it survives the
    # rerun caused by the user editing the form to answer it.
    clarification = st.session_state.get("clarification")
    if clarification:
        st.warning("The requirement agent needs more detail before it can start.")
        st.markdown(f"**Question:** {clarification['question_for_user']}")
        st.markdown(f"**Missing fields:** `{', '.join(clarification['missing_fields'])}`")
        st.caption("This is a normal response, not an error. Revise below and resubmit.")

    if mode == "Structured spec":
        _structured_form(base_url)
    else:
        _freetext_form(base_url)


def _data_source_input(base_url: str, key_prefix: str) -> Optional[str]:
    """Returns a data_source string, or None if the user has not supplied one
    yet. Uploading happens here so the caller only ever deals with a path."""
    choice = st.radio("Data source", ["Upload a file", "Enter a path manually"],
                      horizontal=True, key=f"{key_prefix}_src_mode")

    if choice == "Enter a path manually":
        return st.text_input(
            "data_source",
            value="builtin:breast_cancer",
            help="A builtin token (builtin:breast_cancer) or a .csv/.parquet path.",
            key=f"{key_prefix}_manual_path",
        ) or None

    uploaded = st.file_uploader("Dataset (.csv or .parquet)", type=["csv", "parquet"],
                                key=f"{key_prefix}_uploader")
    if uploaded is None:
        return None

    # Cache by filename+size so re-renders don't re-upload the same bytes.
    cache_key = f"{key_prefix}_uploaded::{uploaded.name}::{uploaded.size}"
    if cache_key not in st.session_state:
        with st.spinner(f"Uploading {uploaded.name}…"):
            data, error = upload_file(base_url, uploaded)
        if error:
            st.error(error)
            return None
        st.session_state[cache_key] = data["file_path"]
        st.success(f"Uploaded {data['filename']} ({data['size_bytes']:,} bytes)")
    return st.session_state[cache_key]


def _submit_run(base_url: str, payload: dict) -> None:
    """POST /runs handles two success shapes: a queued run, or a request for
    clarification. Only genuine transport/HTTP failures are errors."""
    with st.spinner("Creating run…"):
        data, error = api_request("POST", base_url, "/runs", json_body=payload)
    if error:
        st.error(error)
        return

    if data.get("status") == "needs_clarification":
        st.session_state["clarification"] = data
        st.rerun()

    st.session_state.pop("clarification", None)
    st.session_state["run_id"] = data["run_id"]
    st.session_state["page"] = "Live Run"
    st.rerun()


def _structured_form(base_url: str) -> None:
    task_type = st.selectbox("task_type", ["classification", "regression"])
    # Options come from ProblemSpec's own validator sets, so an invalid
    # metric/task combination cannot be submitted from this form at all.
    metrics = sorted(CLASSIFICATION_METRICS if task_type == "classification"
                     else REGRESSION_METRICS)

    col_a, col_b = st.columns(2)
    target_column = col_a.text_input("target_column", value="churned")
    success_metric = col_b.selectbox("success_metric", metrics)
    metric_threshold = st.number_input("metric_threshold", value=0.85, step=0.01, format="%.4f")

    with st.expander("Constraints (optional)"):
        max_tuning_trials = st.number_input(
            "max_tuning_trials", min_value=1, value=50,
            help="Default is 50. Lower it (e.g. 5) for a fast smoke test — each trial is a 3-fold CV fit.",
        )
        max_loop_backs = st.number_input("max_loop_backs", min_value=0, value=3)

    data_source = _data_source_input(base_url, "structured")

    if st.button("Create run", type="primary"):
        if not data_source:
            st.error("Provide a data source — upload a file or enter a path.")
            return
        if not target_column.strip():
            st.error("target_column is required.")
            return
        _submit_run(base_url, {
            "problem_spec": {
                "task_type": task_type,
                "target_column": target_column.strip(),
                "success_metric": success_metric,
                "metric_threshold": float(metric_threshold),
                "data_source": data_source,
                "constraints": {
                    "max_tuning_trials": int(max_tuning_trials),
                    "max_loop_backs": int(max_loop_backs),
                },
            }
        })


def _freetext_form(base_url: str) -> None:
    context = st.text_area(
        "Describe what you want to predict",
        value="Predict whether a customer will churn, target column churned, aim for at least 0.85 f1.",
        height=120,
        help="Name the target column exactly as it appears in your file — the agent validates it against the real schema.",
    )
    uploaded = st.file_uploader("Dataset (optional, .csv or .parquet)",
                                type=["csv", "parquet"], key="freetext_uploader")

    file_path = None
    if uploaded is not None:
        cache_key = f"freetext_uploaded::{uploaded.name}::{uploaded.size}"
        if cache_key not in st.session_state:
            with st.spinner(f"Uploading {uploaded.name}…"):
                data, error = upload_file(base_url, uploaded)
            if error:
                st.error(error)
                return
            st.session_state[cache_key] = data["file_path"]
            st.success(f"Uploaded {data['filename']} ({data['size_bytes']:,} bytes)")
        file_path = st.session_state[cache_key]

    if st.button("Create run", type="primary"):
        if not context.strip():
            st.error("Describe what you want to predict.")
            return
        # file_path is omitted entirely when absent — sending null would still
        # be valid here, but omitting keeps the request minimal and matches
        # RunCreateRequest's "context alone is a complete request" contract.
        payload: dict = {"context": context.strip()}
        if file_path:
            payload["file_path"] = file_path
        _submit_run(base_url, payload)


def page_live_run(base_url: str) -> None:
    st.title("Live run")

    run_id = st.text_input("run_id", value=st.session_state.get("run_id", ""))
    if not run_id:
        st.info("Create a run, or paste a run_id above.")
        return
    st.session_state["run_id"] = run_id

    if st.button("Refresh now"):
        st.rerun()

    detail, error = api_request("GET", base_url, f"/runs/{run_id}")
    if error:
        st.error(error)
        return

    if detail["status"] in TERMINAL_STATUSES:
        st.caption("Run is in a terminal state — auto-refresh is off.")
        render_run_detail(detail)
    else:
        _live_fragment(base_url, run_id)


@st.fragment(run_every=REFRESH_SECONDS)
def _live_fragment(base_url: str, run_id: str) -> None:
    """Auto-refreshing region.

    st.fragment(run_every=...) is used rather than the time.sleep()+st.rerun()
    loop or the third-party streamlit-autorefresh package because it reruns
    ONLY this block: the sidebar and the run_id input keep their state, nothing
    blocks the script thread, and no extra dependency is needed. Once the run
    reaches a terminal state the fragment escapes to a full-app rerun, after
    which page_live_run takes the static branch and refreshing stops.
    """
    detail, error = api_request("GET", base_url, f"/runs/{run_id}")
    if error:
        st.error(error)
        return
    if detail["status"] in TERMINAL_STATUSES:
        st.rerun(scope="app")
    st.caption(f"Auto-refreshing every {REFRESH_SECONDS}s while the run is active.")
    render_run_detail(detail)


def page_history(base_url: str) -> None:
    st.title("Run history")

    if st.button("Refresh list"):
        st.rerun()

    runs, error = api_request("GET", base_url, "/runs")
    if error:
        st.error(error)
        return
    if not runs:
        st.info("No runs yet.")
        return

    # GET /runs is already ordered newest-first server-side; sorting here too
    # keeps the view correct if that ever changes.
    frame = pd.DataFrame(runs).sort_values("created_at", ascending=False)
    st.dataframe(
        frame[["run_id", "status", "task_type", "pass_fail", "metric_value",
               "loop_count", "created_at"]],
        hide_index=True, width="stretch",
    )

    chosen = st.selectbox(
        "View a run's detail",
        options=[r["run_id"] for r in runs],
        format_func=lambda rid: f"{rid[:8]}…  ({next(r['status'] for r in runs if r['run_id'] == rid)})",
    )
    if st.button("Open this run"):
        st.session_state["run_id"] = chosen
        st.session_state["page"] = "Live Run"
        st.rerun()

    if chosen:
        detail, detail_error = api_request("GET", base_url, f"/runs/{chosen}")
        if detail_error:
            st.error(detail_error)
        else:
            st.divider()
            render_run_detail(detail)


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------

def main() -> None:
    st.sidebar.title("ML Pipeline Agent")
    base_url = st.sidebar.text_input("API base URL", value=DEFAULT_API_URL)

    pages = ["New Run", "Live Run", "Run History"]
    # Navigation lives in session state so a page can redirect to another one
    # (New Run -> Live Run after a successful submit).
    current = st.session_state.get("page", "New Run")
    choice = st.sidebar.radio("Page", pages, index=pages.index(current))
    if choice != current:
        st.session_state["page"] = choice
        current = choice

    if st.session_state.get("run_id"):
        st.sidebar.caption(f"Current run: `{st.session_state['run_id'][:8]}…`")

    health, health_error = api_request("GET", base_url, "/health")
    if health_error:
        st.sidebar.error("API unreachable")
    else:
        st.sidebar.success(f"API {health.get('status', 'ok')}")

    if current == "New Run":
        page_new_run(base_url)
    elif current == "Live Run":
        page_live_run(base_url)
    else:
        page_history(base_url)


if __name__ == "__main__":
    main()
