import json
import uuid

from agents.data_agent import DataAgent
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec
from tools.feature_tools import engineer_features_tool

spec = ProblemSpec(
    task_type="classification",
    target_column="diagnosis",
    success_metric="f1",
    metric_threshold=0.90,
    data_source="builtin:breast_cancer",
)

run_state = PipelineRun(run_id=str(uuid.uuid4()), problem_spec=spec)
cache = RunCache()

# Populate pipeline_run.data_profile + cache["cleaned_df"], same as the real orchestrator.
DataAgent().run(
    data_source=spec.data_source,
    target_column=spec.target_column,
    task_type=spec.task_type,
    pipeline_run=run_state,
    cache=cache,
)

# Call the TOOL directly — no LLM call, just the raw deterministic report
# exactly as FeatureAgent would receive it via report.model_dump_json().
report = engineer_features_tool(
    data_source=spec.data_source,
    target_column=spec.target_column,
    task_type=spec.task_type,
    cache=cache,
    run_id=run_state.run_id,
    profile=run_state.data_profile,
)

print("=== leakage_warnings ===")
print(report.leakage_warnings)

print("\n=== correlation_summary, sorted highest first ===")
for name, corr in sorted(report.correlation_summary.items(), key=lambda x: -x[1]):
    print(f"{corr:.4f}  {name}")

print("\n=== dropped_features ===")
for d in report.dropped_features:
    print(f"- {d.name}: {d.rationale}")

print("\n=== engineered_features ===")
for e in report.engineered_features:
    print(f"- {e.name}: {e.rationale}")

print("\n=== FULL raw JSON (exactly what the LLM receives) ===")
print(json.dumps(json.loads(report.model_dump_json()), indent=2))