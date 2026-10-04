/**
 * Types mirroring the REAL FastAPI backend exactly.
 *
 * Every type here was verified against the actual source files, not assumed:
 *   api/schemas_api.py, api/routes/runs.py, api/routes/uploads.py
 *   schemas/problem_spec.py, schemas/data_profile.py, schemas/eda_report.py,
 *   schemas/tuning_result.py, schemas/evaluation_report.py,
 *   schemas/pipeline_run.py, schemas/trace_entry.py, schemas/dataset_peek.py
 *
 * Fields that exist on the backend Pydantic model as a `@property` (not a
 * declared field) are NOT included here, because pydantic does not serialize
 * properties to JSON — they are simply absent from the wire response.
 * Confirmed absent this way: DataProfile.is_usable, EDAReport.has_leakage_risk,
 * EDAReport.has_data_quality_risk, EvaluationReport.passed,
 * PipelineRun.total_cost_usd / total_tokens (RunDetail re-declares these two
 * as real fields instead, which is why they DO appear below).
 *
 * Fields that exist on a backend model but are never populated by any code
 * path are still included (they are real, typed, optional fields on the
 * wire) — see the comment on `model_artifact_path` below for the one
 * concrete case of this.
 */

// ---------------------------------------------------------------------------
// ProblemSpec — schemas/problem_spec.py
// ---------------------------------------------------------------------------

export type TaskType = 'classification' | 'regression'

/** schemas/problem_spec.py: CLASSIFICATION_METRICS / REGRESSION_METRICS.
 *  ProblemSpec._metric_matches_task rejects any other value for a given
 *  task_type — these sets are the complete, closed list on both sides. */
export const CLASSIFICATION_METRICS = [
  'accuracy',
  'f1',
  'precision',
  'recall',
  'roc_auc',
] as const
export const REGRESSION_METRICS = ['rmse', 'mae', 'r2'] as const
export type ClassificationMetric = (typeof CLASSIFICATION_METRICS)[number]
export type RegressionMetric = (typeof REGRESSION_METRICS)[number]

export interface PipelineConstraints {
  /** Declared on the backend but read by no pipeline code — wiring it up
   *  would be a backend change, out of scope here. Exposed as an input
   *  (it IS a real field the API accepts and stores) but never claim in
   *  the UI that it currently bounds anything. */
  max_training_seconds: number | null
  max_tuning_trials: number | null
  /** Same story as max_training_seconds — accepted and stored, not
   *  currently read by training_stage.py's model selection. */
  interpretability_required: boolean
  max_loop_backs: number
}

export interface ProblemSpec {
  task_type: TaskType
  target_column: string
  success_metric: string
  metric_threshold: number
  data_source: string
  constraints: PipelineConstraints
  notes: string | null
}

// ---------------------------------------------------------------------------
// DataProfile — schemas/data_profile.py (Data Agent)
// ---------------------------------------------------------------------------

export interface ColumnSchema {
  name: string
  dtype: string
  missing_pct: number
  is_target: boolean
}

export interface CleaningAction {
  column: string
  action: string
  rationale: string
}

export interface DataProfile {
  row_count: number
  column_count: number
  column_schema: ColumnSchema[]
  /** Only populated for classification tasks. */
  class_balance: Record<string, number> | null
  outlier_flags: string[]
  cleaning_actions_taken: CleaningAction[]
  warnings: string[]
  blocking_issue: string | null
}

// ---------------------------------------------------------------------------
// EDAReport — schemas/eda_report.py (Feature Agent)
// ---------------------------------------------------------------------------

export interface FeatureDecision {
  name: string
  rationale: string
}

export interface EDAReport {
  /** FLAT map: feature name -> absolute correlation with the TARGET.
   *  NOT a pairwise feature-vs-feature matrix. */
  correlation_summary: Record<string, number>
  engineered_features: FeatureDecision[]
  dropped_features: FeatureDecision[]
  /** Plain warning strings — NOT structured objects with severity/confidence. */
  leakage_warnings: string[]
  /** Deliberately separate from leakage_warnings on the backend: leakage is
   *  "can this result be trusted", data quality is "will this run crash".
   *  Never merge these two lists in the UI. */
  data_quality_warnings: string[]
  target_distribution_notes: string
  final_feature_names: string[]
}

// ---------------------------------------------------------------------------
// TuningResult — schemas/tuning_result.py (Tuning Agent)
// ---------------------------------------------------------------------------

export interface TrialRecord {
  trial_number: number
  params: Record<string, unknown>
  score: number
}

export interface TuningResult {
  best_params: Record<string, unknown>
  best_cv_score: number
  metric_optimized: string
  search_history: TrialRecord[]
  search_budget_used: number
  search_budget_total: number
  convergence_notes: string
  /** Genuine plateau-based detection (last-20%-of-trials comparison), not a
   *  simple "did it use its whole budget" flag. See the backend's own
   *  extensive comment on this field for why it was made a required field. */
  converged: boolean
}

// ---------------------------------------------------------------------------
// EvaluationReport — schemas/evaluation_report.py (Training Agent)
// ---------------------------------------------------------------------------

export type RecommendedNextStep =
  | 'revisit_features'
  | 'expand_hyperparam_search'
  | 'insufficient_data'
  | 'bad_spec'

export interface FailureAnalysis {
  summary: string
  recommended_next_step: RecommendedNextStep
}

export interface EvaluationReport {
  test_metrics: Record<string, number>
  /** Classification only. */
  confusion_matrix: number[][] | null
  /** Regression only. */
  residual_summary: Record<string, number> | null
  metric_optimized: string
  metric_value: number
  metric_threshold: number
  pass_fail: 'pass' | 'fail'
  failure_analysis: FailureAnalysis | null
  /** Declared on the backend but NEVER assigned by any code path —
   *  tools/training_tools.py fits a model, then discards it
   *  (`_model, eval_report = train_and_evaluate(...)`) and only returns
   *  the report. This field is therefore always null today. The UI must
   *  not build any "download model" affordance around it. */
  model_artifact_path: string | null
}

// ---------------------------------------------------------------------------
// PipelineRun / RunDetail — schemas/pipeline_run.py, api/schemas_api.py
// ---------------------------------------------------------------------------

/** schemas/pipeline_run.py RunStatus — the COMPLETE, verified set of values
 *  the orchestrator can persist on a run. "queued" is deliberately NOT in
 *  this union: it only ever appears as the literal on RunCreatedResponse
 *  (the POST /runs acknowledgement), never as a value written back onto
 *  PipelineRun.status. Immediately after creation a run's real status is
 *  "created", not "queued". */
export type RunStatus =
  | 'created'
  | 'data_ready'
  | 'features_ready'
  | 'tuned'
  | 'evaluated'
  | 'done'
  | 'escalated'

export const TERMINAL_STATUSES: readonly RunStatus[] = ['done', 'escalated']

export interface HistoryEntry {
  timestamp: string
  stage: string
  event: string
}

export interface TraceEntry {
  timestamp: string
  agent: string
  model: string
  prompt_tokens: number
  completion_tokens: number
  total_tokens: number
  latency_ms: number
  estimated_cost_usd: number
}

/** api/schemas_api.py RunDetail — GET /runs/{run_id}. */
export interface RunDetail {
  run_id: string
  status: RunStatus
  loop_count: number
  problem_spec: ProblemSpec
  data_profile: DataProfile | null
  eda_report: EDAReport | null
  tuning_result: TuningResult | null
  evaluation_report: EvaluationReport | null
  history: HistoryEntry[]
  trace: TraceEntry[]
  total_cost_usd: number
  total_tokens: number
}

/** api/schemas_api.py RunSummary — GET /runs (list). Deliberately lightweight
 *  — no nested reports, no trace, no cost/token totals. A "global LLM cost
 *  trend across all runs" is therefore NOT cheaply derivable from this list
 *  alone; see src/pages/LlmUsage.tsx for how that constraint is handled. */
export interface RunSummary {
  run_id: string
  status: RunStatus
  task_type: TaskType | null
  pass_fail: 'pass' | 'fail' | null
  metric_value: number | null
  loop_count: number
  created_at: string
  updated_at: string
}

// ---------------------------------------------------------------------------
// Requests / other responses
// ---------------------------------------------------------------------------

export interface UploadResponse {
  file_path: string
  filename: string
  size_bytes: number
}

/** api/schemas_api.py RunCreateRequest — validated server-side as
 *  "exactly one of problem_spec OR context", never both, never neither. */
export type RunCreateRequest =
  | { problem_spec: ProblemSpec; context?: never; file_path?: never; constraints?: never }
  | {
      context: string
      file_path?: string
      /** Overrides layered on the spec the Requirement Agent extracts; only
       *  the keys sent replace the backend defaults. */
      constraints?: Partial<PipelineConstraints>
      problem_spec?: never
    }

/** api/routes/datasets.py DatasetInfo — GET /datasets. */
export interface DatasetInfo {
  source: string
  label: string
  description: string
  task_type: TaskType | null
}

/** schemas/dataset_peek.py — GET /datasets/peek. Schema only, never rows. */
export interface ColumnPeek {
  name: string
  dtype: string
  missing_pct: number
  unique_count: number
  example_value: string[]
}

export interface DatasetPeek {
  row_count: number
  column_count: number
  columns: ColumnPeek[]
}

export interface RunCreatedResponse {
  run_id: string
  status: 'queued'
}

export interface ClarificationResponse {
  status: 'needs_clarification'
  missing_fields: string[]
  question_for_user: string
}

export type CreateRunResponse = RunCreatedResponse | ClarificationResponse

export function isClarification(
  response: CreateRunResponse,
): response is ClarificationResponse {
  return response.status === 'needs_clarification'
}
