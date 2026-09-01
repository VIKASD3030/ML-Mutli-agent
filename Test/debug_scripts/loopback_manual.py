from pipeline.run_pipeline import run
from schemas.problem_spec import ProblemSpec

spec = ProblemSpec(
    task_type="classification",
    target_column="diagnosis",
    success_metric="f1",
    metric_threshold=0.999,  # deliberately unreachable — forces loop-back
    data_source="builtin:breast_cancer",
)
spec.constraints.max_tuning_trials = 10
spec.constraints.max_loop_backs = 3

final_state = run(spec)

print("\n=== PIPELINE RUN SUMMARY ===")
print(f"status:  {final_state.status}")
print(f"loops:   {final_state.loop_count}")
print("\n--- history ---")
for entry in final_state.history:
    print(f"[{entry.stage}] {entry.event}")