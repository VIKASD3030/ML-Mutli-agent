import type { RunDetail } from '@/api/types'
import type { StageState } from './status'

export type StageKey = 'requirement' | 'data' | 'feature' | 'tuning' | 'training'

export interface StageConfig {
  key: StageKey
  label: string
  /** Real history "stage" value this agent logs under — null for
   * Requirement, which runs before a PipelineRun exists and therefore
   * never appears in history at all. */
  historyStage: string | null
  /** Key on RunDetail holding this stage's real report object — null for
   * Requirement, whose only output (ProblemSpec) is not a "stage report",
   * it's the input every other stage reads. */
  reportKey: 'data_profile' | 'eda_report' | 'tuning_result' | 'evaluation_report' | null
  deterministicTool: string
  llmResponsibility: string
}

/** Descriptions here are drawn directly from this project's own agent
 * source files (agents/*.py module/class docstrings and system prompts),
 * not invented copy — used by both the pipeline row tooltips and the
 * Architecture page. */
export const STAGES: StageConfig[] = [
  {
    key: 'requirement',
    label: 'Requirement',
    historyStage: null,
    reportKey: null,
    deterministicTool: 'peek_dataset_tool() — reads the uploaded file\'s schema (columns, dtypes, missingness) with no cleaning or LLM involvement',
    llmResponsibility: 'Extracts task_type, target_column, success_metric, and metric_threshold from free text, grounded in the real column names above — never invents a column name',
  },
  {
    key: 'data',
    label: 'Data',
    historyStage: 'data_agent',
    reportKey: 'data_profile',
    deterministicTool: 'profile_dataset_tool() — cleans the data and computes a DataProfile (missingness, class balance, outliers)',
    llmResponsibility: 'Judges whether the profiled data is good enough to proceed to feature engineering (proceed: true/false)',
  },
  {
    key: 'feature',
    label: 'Feature',
    historyStage: 'feature_agent',
    reportKey: 'eda_report',
    deterministicTool: 'engineer_features_tool() — correlation filtering, leakage detection, encoding, interaction features',
    llmResponsibility: 'Judges whether the engineered feature set is safe to tune on, weighing leakage_warnings most heavily (proceed: true/false)',
  },
  {
    key: 'tuning',
    label: 'Tuning',
    historyStage: 'tuning_agent',
    reportKey: 'tuning_result',
    deterministicTool: 'tune_model_tool() — Optuna hyperparameter search with cross-validation',
    llmResponsibility: 'Judges whether the search converged well enough (proceed is advisory only here — see the loop-back note below)',
  },
  {
    key: 'training',
    label: 'Training',
    historyStage: 'training_agent',
    reportKey: 'evaluation_report',
    deterministicTool: 'train_and_evaluate_tool() — trains the final model, computes test metrics, and (on failure) a rule-based failure_analysis',
    llmResponsibility: "Sanity-checks the deterministic failure_analysis recommendation in plain language (advisory only — never overrides the routing decision)",
  },
]

/** "done" means this stage's report object is present on the run — the only
 * signal available, since there is no per-stage status field. A stage that
 * was skipped on a LATER loop-back iteration (see wasFeatureAgentSkipped in
 * lib/history.ts) still reports "done" here, correctly: its report from the
 * earlier pass is still the real, current eda_report on the run. The skip
 * ITSELF — a fact about one specific iteration, not the stage's current
 * state — is surfaced separately, in the loop-back banner and timeline. */
export function stageState(detail: RunDetail, stage: StageConfig): StageState {
  if (stage.key === 'requirement') return 'done' // a run only exists once this succeeded
  if (stage.reportKey && detail[stage.reportKey]) return 'done'
  return 'pending'
}
