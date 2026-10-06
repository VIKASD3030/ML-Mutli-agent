/** Wire types for the real FastAPI backend (api/schemas_api.py and the schemas/ it re-uses). */

export type WireTaskType = 'classification' | 'regression';

export interface WireConstraints {
  max_training_seconds: number | null;
  max_tuning_trials: number | null;
  interpretability_required: boolean;
  max_loop_backs: number;
}

export interface WireProblemSpec {
  task_type: WireTaskType;
  target_column: string;
  success_metric: string;
  metric_threshold: number;
  data_source: string;
  constraints: WireConstraints;
  notes: string | null;
}

export interface WireDataProfile {
  row_count: number;
  column_count: number;
  column_schema: { name: string; dtype: string; missing_pct: number; is_target: boolean }[];
  class_balance: Record<string, number> | null;
  outlier_flags: string[];
  cleaning_actions_taken: { column: string; action: string; rationale: string }[];
  warnings: string[];
  blocking_issue: string | null;
}

export interface WireEDAReport {
  /** feature name -> |correlation with the TARGET| (not pairwise). */
  correlation_summary: Record<string, number>;
  engineered_features: { name: string; rationale: string }[];
  dropped_features: { name: string; rationale: string }[];
  leakage_warnings: string[];
  data_quality_warnings: string[];
  final_feature_names: string[];
}

export interface WireTuningResult {
  best_params: Record<string, unknown>;
  best_cv_score: number;
  search_history: { trial_number: number; params: Record<string, unknown>; score: number }[];
  convergence_notes: string;
  converged: boolean;
}

export interface WireEvaluationReport {
  test_metrics: Record<string, number>;
  confusion_matrix: number[][] | null;
  residual_summary: Record<string, number> | null;
  metric_value: number;
  metric_threshold: number;
  pass_fail: 'pass' | 'fail';
  failure_analysis: { summary: string; recommended_next_step: string } | null;
}

export interface WireHistoryEntry {
  timestamp: string;
  stage: string;
  event: string;
}

export interface WireTraceEntry {
  agent: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
}

export type WireRunStatus =
  | 'created'
  | 'data_ready'
  | 'features_ready'
  | 'tuned'
  | 'evaluated'
  | 'done'
  | 'escalated';

export interface WireRunDetail {
  run_id: string;
  status: WireRunStatus;
  loop_count: number;
  problem_spec: WireProblemSpec;
  data_profile: WireDataProfile | null;
  eda_report: WireEDAReport | null;
  tuning_result: WireTuningResult | null;
  evaluation_report: WireEvaluationReport | null;
  history: WireHistoryEntry[];
  trace: WireTraceEntry[];
  total_cost_usd: number;
  total_tokens: number;
}

export interface WireRunSummary {
  run_id: string;
  status: WireRunStatus;
  task_type: WireTaskType | null;
  pass_fail: 'pass' | 'fail' | null;
  metric_value: number | null;
  loop_count: number;
  created_at: string;
  updated_at: string;
}

export interface WireDatasetInfo {
  source: string;
  label: string;
  description: string;
  task_type: WireTaskType | null;
}

export interface WireDatasetPeek {
  row_count: number;
  column_count: number;
  columns: {
    name: string;
    dtype: string;
    missing_pct: number;
    unique_count: number;
    example_value: string[];
  }[];
}

export interface WireUpload {
  file_path: string;
  filename: string;
  size_bytes: number;
}

export type WireCreateRunResponse =
  | { run_id: string; status: 'queued' }
  | { status: 'needs_clarification'; missing_fields: string[]; question_for_user: string };

export type WireCreateRunRequest =
  | { problem_spec: WireProblemSpec }
  | {
      context: string;
      file_path?: string;
      constraints?: Partial<WireConstraints>;
    };
