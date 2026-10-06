/**
 * View-model types for the UI. These are the shapes the views render; the
 * real FastAPI responses are declared in services/apiTypes.ts and converted
 * by services/adapter.ts.
 *
 * Fields the backend does not persist (LLM "confidence", agent reasoning
 * text, the training agent's agree/disagree verdict) are deliberately
 * optional or absent here — the views must not invent them.
 */

export type TaskType = 'classification' | 'regression';
export type SuccessMetric = string;
export type RunStatus = 'queued' | 'running' | 'completed' | 'failed' | 'looping_back';
export type PipelineStage = 'idle' | 'requirement' | 'data' | 'feature' | 'tuning' | 'training' | 'completed';

/** The sidebar sliders. Only the first two reach the backend. */
export interface PipelineConstraints {
  max_loop_backs: number;
  max_tuning_trials: number;
  /** Preview only: the pipeline's fold count (cv=3 in tuning_stage.py) is not a run setting. */
  cv_folds: number;
}

export interface ProblemSpec {
  task_type: TaskType;
  target_column: string;
  success_metric: SuccessMetric;
  metric_threshold: number;
  data_source: string;
  constraints: { max_loop_backs: number; max_tuning_trials: number | null };
}

export interface ColumnSchema {
  name: string;
  dtype: string;
  missing_pct: number;
  cardinality: number;
  sample_values: (string | number)[];
}

/** What the backend log lets us say about an agent's LLM verdict. */
export interface AgentJudgment {
  /** Parsed from the "proceed=True/False" history line; null if not logged. */
  proceed: boolean | null;
}

export interface DataProfile {
  column_schema: { name: string; dtype: string; missing_pct: number; is_target: boolean }[];
  class_balance: Record<string, number> | null;
  outlier_flags: string[];
  cleaning_actions_taken: string[];
  warnings: string[];
  blocking_issue: string | null;
  agent_judgment: AgentJudgment;
}

export interface CorrelationItem {
  feature_a: string;
  feature_b: string;
  correlation: number;
}

export interface EngineeredFeature {
  name: string;
  rationale: string;
}

export interface DroppedFeature {
  name: string;
  rationale: string;
}

export interface EDAReport {
  correlation_summary: CorrelationItem[];
  engineered_features: EngineeredFeature[];
  dropped_features: DroppedFeature[];
  leakage_warnings: string[];
  data_quality_warnings: string[];
  final_feature_names: string[];
  agent_judgment: AgentJudgment;
}

export interface OptunaTrial {
  trial: number;
  cv_score: number;
  params: Record<string, any>;
}

export interface TuningResult {
  best_params: Record<string, any>;
  best_cv_score: number;
  search_history: OptunaTrial[];
  converged: boolean;
  convergence_notes: string;
  agent_judgment: AgentJudgment;
}

export interface ResidualSummary {
  mean: number;
  std: number;
}

export interface FailureAnalysis {
  summary: string;
  recommended_next_step: 'revisit_features' | 'expand_hyperparam_search' | 'insufficient_data' | 'bad_spec' | 'none';
}

export interface EvaluationReport {
  test_metrics: Record<string, number>;
  confusion_matrix: number[][] | null;
  matrix_labels?: string[];
  residual_summary: ResidualSummary | null;
  pass_fail: 'PASS' | 'FAIL';
  failure_analysis: FailureAnalysis;
}

export interface TraceEntry {
  agent: string;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  latency_ms: number;
  estimated_cost_usd: number;
}

export interface HistoryEntry {
  timestamp: string;
  stage: string;
  message: string;
  type: 'info' | 'warning' | 'success' | 'loop_back' | 'error';
}

export interface PipelineRun {
  run_id: string;
  status: RunStatus;
  active_stage: PipelineStage;
  loop_count: number;
  max_loop_backs: number;
  /** Where the orchestrator last resumed from; null when no loop-back happened. */
  resume_from: string | null;
  created_at: string;
  updated_at: string;
  problem_spec: ProblemSpec;
  data_profile?: DataProfile | null;
  eda_report?: EDAReport | null;
  tuning_result?: TuningResult | null;
  evaluation_report?: EvaluationReport | null;
  history: HistoryEntry[];
  trace: TraceEntry[];
  total_cost_usd: number;
  total_tokens: number;
}

export interface DatasetSummary {
  /** The data_source token sent to the backend (builtin:… or an uploads/ / sample_data/ path). */
  id: string;
  name: string;
  description: string;
  rows: number;
  columns: number;
  suggested_task: TaskType;
  suggested_target: string;
  suggested_metric: SuccessMetric;
  sample_prompt: string;
  schema: ColumnSchema[];
}
