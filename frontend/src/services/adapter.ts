/**
 * Converts real backend responses into the view-model types in ../types.
 *
 * Rule for this file: derive, never invent. Anything the backend does not
 * persist (LLM confidence, agent reasoning, the training agent's agree /
 * disagree verdict) is simply not produced — the views render around it.
 */
import type {
  ColumnSchema,
  DatasetSummary,
  FailureAnalysis,
  HistoryEntry,
  PipelineRun,
  PipelineStage,
  RunStatus,
  TaskType,
} from '../types';
import type {
  WireDatasetInfo,
  WireDatasetPeek,
  WireHistoryEntry,
  WireRunDetail,
} from './apiTypes';

// ---------------------------------------------------------------------------
// History-derived facts (strings are the backend's own log text)
// ---------------------------------------------------------------------------

/** "proceed=True/False" is only ever logged as text; null means the agent hasn't logged it. */
function parseProceed(history: WireHistoryEntry[], stage: string): boolean | null {
  for (let i = history.length - 1; i >= 0; i--) {
    const h = history[i];
    if (h.stage !== stage) continue;
    if (h.event.includes('proceed=True')) return true;
    if (h.event.includes('proceed=False')) return false;
  }
  return null;
}

function resumeFrom(history: WireHistoryEntry[]): string | null {
  for (let i = history.length - 1; i >= 0; i--) {
    const m = history[i].event.match(/fail -> recommended_next_step=(\w+)/);
    if (!m) continue;
    if (m[1] === 'revisit_features') return 'feature';
    if (m[1] === 'expand_hyperparam_search') return 'tuning';
    return null;
  }
  return null;
}

function formatTime(ts: string): string {
  const d = new Date(ts);
  return Number.isNaN(d.getTime()) ? ts : d.toLocaleTimeString('en-GB');
}

function historyType(event: string): HistoryEntry['type'] {
  const e = event.toLowerCase();
  if (e.includes('fail ->')) return 'loop_back';
  if (e.includes('escalat')) return 'error';
  if (e.includes('proceed=false') || e.includes('concerns')) return 'warning';
  if (e.includes('proceed=true') || e.includes('complete') || e.includes('passed')) return 'success';
  return 'info';
}

// ---------------------------------------------------------------------------
// Status / stage
// ---------------------------------------------------------------------------

function mapStatus(d: WireRunDetail): RunStatus {
  if (d.status === 'done') return 'completed';
  if (d.status === 'escalated') return 'failed';
  // A failed evaluation on a non-terminal run means the orchestrator is routing back.
  if (d.status === 'evaluated' && d.evaluation_report?.pass_fail === 'fail') return 'looping_back';
  return 'running';
}

function mapStage(d: WireRunDetail): PipelineStage {
  switch (d.status) {
    case 'done':
      return 'completed';
    case 'created':
      return 'data';
    case 'data_ready':
      return 'feature';
    case 'features_ready':
      return 'tuning';
    case 'tuned':
    case 'evaluated':
      return 'training';
    case 'escalated':
      // Terminal: show the furthest stage that actually produced a report.
      if (d.evaluation_report) return 'training';
      if (d.tuning_result) return 'tuning';
      if (d.eda_report) return 'feature';
      return 'data';
  }
}

// ---------------------------------------------------------------------------
// RunDetail -> PipelineRun
// ---------------------------------------------------------------------------

export function toPipelineRun(d: WireRunDetail): PipelineRun {
  const h = d.history;
  const spec = d.problem_spec;
  const skippedFeature = h.some((x) => x.event.includes('skipping feature_agent'));

  const eval_ = d.evaluation_report;
  const metricsLabels = (m: number[][] | null) =>
    m ? m.map((_, i) => `Class ${i}`) : undefined;

  return {
    run_id: d.run_id,
    status: mapStatus(d),
    active_stage: mapStage(d),
    loop_count: d.loop_count,
    max_loop_backs: spec.constraints.max_loop_backs,
    resume_from: skippedFeature ? 'tuning' : resumeFrom(h),
    created_at: h[0]?.timestamp ?? '',
    updated_at: h[h.length - 1]?.timestamp ?? '',
    problem_spec: {
      task_type: spec.task_type,
      target_column: spec.target_column,
      success_metric: spec.success_metric,
      metric_threshold: spec.metric_threshold,
      data_source: spec.data_source,
      constraints: {
        max_loop_backs: spec.constraints.max_loop_backs,
        max_tuning_trials: spec.constraints.max_tuning_trials,
      },
    },
    data_profile: d.data_profile && {
      column_schema: d.data_profile.column_schema,
      class_balance: d.data_profile.class_balance,
      outlier_flags: d.data_profile.outlier_flags,
      cleaning_actions_taken: d.data_profile.cleaning_actions_taken.map(
        (a) => `${a.column}: ${a.action} — ${a.rationale}`,
      ),
      warnings: d.data_profile.warnings,
      blocking_issue: d.data_profile.blocking_issue,
      agent_judgment: { proceed: parseProceed(h, 'data_agent') },
    },
    eda_report: d.eda_report && {
      correlation_summary: Object.entries(d.eda_report.correlation_summary)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 8)
        .map(([feature, corr]) => ({
          feature_a: feature,
          feature_b: spec.target_column,
          correlation: corr,
        })),
      engineered_features: d.eda_report.engineered_features,
      dropped_features: d.eda_report.dropped_features,
      leakage_warnings: d.eda_report.leakage_warnings,
      data_quality_warnings: d.eda_report.data_quality_warnings,
      final_feature_names: d.eda_report.final_feature_names,
      agent_judgment: { proceed: parseProceed(h, 'feature_agent') },
    },
    tuning_result: d.tuning_result && {
      best_params: d.tuning_result.best_params as Record<string, unknown>,
      best_cv_score: d.tuning_result.best_cv_score,
      search_history: d.tuning_result.search_history.map((t) => ({
        trial: t.trial_number + 1,
        cv_score: t.score,
        params: t.params as Record<string, unknown>,
      })),
      converged: d.tuning_result.converged,
      convergence_notes: d.tuning_result.convergence_notes,
      agent_judgment: { proceed: parseProceed(h, 'tuning_agent') },
    },
    evaluation_report: eval_ && {
      test_metrics: eval_.test_metrics,
      confusion_matrix: eval_.confusion_matrix,
      matrix_labels: metricsLabels(eval_.confusion_matrix),
      residual_summary:
        eval_.residual_summary &&
        typeof eval_.residual_summary.mean === 'number' &&
        typeof eval_.residual_summary.std === 'number'
          ? { mean: eval_.residual_summary.mean, std: eval_.residual_summary.std }
          : null,
      pass_fail: eval_.pass_fail === 'pass' ? 'PASS' : 'FAIL',
      failure_analysis: eval_.failure_analysis
        ? {
            summary: eval_.failure_analysis.summary,
            recommended_next_step: eval_.failure_analysis
              .recommended_next_step as FailureAnalysis['recommended_next_step'],
          }
        : {
            summary:
              eval_.pass_fail === 'pass'
                ? `Test ${spec.success_metric} ${eval_.metric_value.toFixed(4)} meets the ${eval_.metric_threshold} threshold.`
                : 'No failure analysis was recorded for this run.',
            recommended_next_step: 'none',
          },
    },
    history: h.map((x) => ({
      timestamp: formatTime(x.timestamp),
      stage: x.stage,
      message: x.event,
      type: historyType(x.event),
    })),
    trace: d.trace,
    total_cost_usd: d.total_cost_usd,
    total_tokens: d.total_tokens,
  };
}

export const isTerminal = (run: PipelineRun) => run.status === 'completed' || run.status === 'failed';

// ---------------------------------------------------------------------------
// Datasets
// ---------------------------------------------------------------------------

const TARGET_NAME_HINTS = ['diagnosis', 'target', 'label', 'churned', 'defaulted', 'class', 'y'];

function guessTarget(columns: string[]): string {
  const hit = columns.find((c) => TARGET_NAME_HINTS.includes(c.toLowerCase()));
  return hit ?? columns[columns.length - 1] ?? '';
}

export function toDatasetSummary(
  info: Pick<WireDatasetInfo, 'source' | 'label' | 'description' | 'task_type'>,
  peek: WireDatasetPeek | null,
): DatasetSummary {
  const schema: ColumnSchema[] = (peek?.columns ?? []).map((c) => ({
    name: c.name,
    dtype: c.dtype,
    missing_pct: c.missing_pct,
    cardinality: c.unique_count,
    sample_values: c.example_value,
  }));
  const task: TaskType = info.task_type ?? 'classification';
  const target = guessTarget(schema.map((c) => c.name));
  const metric = task === 'classification' ? 'f1' : 'r2';
  return {
    id: info.source,
    name: info.label,
    description: info.description,
    rows: peek?.row_count ?? 0,
    columns: peek?.column_count ?? 0,
    suggested_task: task,
    suggested_target: target,
    suggested_metric: metric,
    sample_prompt: target
      ? `Predict ${target} from the other columns, aim for at least 0.9 ${metric}.`
      : '',
    schema,
  };
}
