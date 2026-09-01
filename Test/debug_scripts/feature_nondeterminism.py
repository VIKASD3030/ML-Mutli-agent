from agents.data_agent import DataAgent
from agents.feature_agent import FeatureAgent
from pipeline.run_cache import RunCache
from schemas.pipeline_run import PipelineRun
from schemas.problem_spec import ProblemSpec
import uuid

spec = ProblemSpec(
    task_type="classification",
    target_column="diagnosis",
    success_metric="f1",
    metric_threshold=0.90,
    data_source="builtin:breast_cancer",
)

run_state = PipelineRun(run_id=str(uuid.uuid4()), problem_spec=spec)
cache = RunCache()

# Populate pipeline_run.data_profile + cache["cleaned_df"], same as the real orchestrator does.
data_agent = DataAgent()
data_agent.run(
    data_source=spec.data_source,
    target_column=spec.target_column,
    task_type=spec.task_type,
    pipeline_run=run_state,
    cache=cache,
)

feature_agent = FeatureAgent()

print("=== FeatureAgent call #1 ===")
result_1 = feature_agent.run(
    data_source=spec.data_source,
    target_column=spec.target_column,
    task_type=spec.task_type,
    pipeline_run=run_state,
    cache=cache,
)
print("proceed:", result_1.proceed)
print("summary:", result_1.summary)
print("concerns:", result_1.concerns)
print("report (leakage_warnings):", result_1.report.leakage_warnings)
print("report (final_feature_names count):", len(result_1.report.final_feature_names))

print("\n=== FeatureAgent call #2 (same pipeline_run, same cache) ===")
result_2 = feature_agent.run(
    data_source=spec.data_source,
    target_column=spec.target_column,
    task_type=spec.task_type,
    pipeline_run=run_state,
    cache=cache,
)
print("proceed:", result_2.proceed)
print("summary:", result_2.summary)
print("concerns:", result_2.concerns)
print("report (leakage_warnings):", result_2.report.leakage_warnings)
print("report (final_feature_names count):", len(result_2.report.final_feature_names))

print("\n=== Are the two EDAReports byte-identical? ===")
print(result_1.report.model_dump_json() == result_2.report.model_dump_json())