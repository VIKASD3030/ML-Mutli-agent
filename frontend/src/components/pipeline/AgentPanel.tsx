import type { RunDetail } from '@/api/types'
import type { StageConfig } from '@/lib/stages'
import { DecisionCard } from './DecisionCard'
import { TraceStat } from './TraceStat'
import { describeRouting } from '@/lib/routing'
import { parseProceed } from '@/lib/judgment'
import { DataProfileView } from './reports/DataProfileView'
import { EdaReportView } from './reports/EdaReportView'
import { TuningResultView } from './reports/TuningResultView'
import { EvaluationReportView } from './reports/EvaluationReportView'
import { EmptyState } from '@/components/ui/States'
import { HelpCircle, MinusCircle } from 'lucide-react'

function ProceedBadge({ value }: { value: true | false | 'unknown' }) {
  if (value === 'unknown') {
    return (
      <span className="inline-flex items-center gap-1 text-sm text-[var(--color-text-tertiary)]">
        <HelpCircle className="h-3.5 w-3.5" /> Not run yet
      </span>
    )
  }
  return (
    <span
      className={`inline-flex items-center gap-1.5 text-sm font-medium ${value ? 'text-violet-300' : 'text-amber-300'}`}
    >
      <span className={`h-2 w-2 rounded-full ${value ? 'bg-violet-400' : 'bg-amber-400'}`} />
      proceed = {value ? 'true' : 'false'}
    </span>
  )
}

function deterministicLines(detail: RunDetail, stage: StageConfig): string[] {
  switch (stage.key) {
    case 'requirement':
      return [
        'peek_dataset_tool() read the real column schema before any LLM call',
        `ProblemSpec produced: target_column = "${detail.problem_spec.target_column}"`,
      ]
    case 'data': {
      const p = detail.data_profile
      if (!p) return []
      return [
        `${p.row_count.toLocaleString()} rows, ${p.column_count} columns profiled`,
        p.cleaning_actions_taken.length > 0
          ? `${p.cleaning_actions_taken.length} cleaning action(s) applied`
          : 'No cleaning actions were necessary',
        p.blocking_issue ? 'Blocking issue detected' : 'No blocking issue found',
      ]
    }
    case 'feature': {
      const r = detail.eda_report
      if (!r) return []
      return [
        `${r.final_feature_names.length} final feature(s) selected`,
        `Leakage check complete — ${r.leakage_warnings.length} warning(s)`,
        `${r.engineered_features.length} engineered, ${r.dropped_features.length} dropped`,
      ]
    }
    case 'tuning': {
      const t = detail.tuning_result
      if (!t) return []
      return [
        `${t.search_budget_used} / ${t.search_budget_total} trials completed`,
        `Best CV score: ${t.best_cv_score.toFixed(4)}`,
        `Convergence check: ${t.converged ? 'plateaued' : 'still improving at budget limit'}`,
      ]
    }
    case 'training': {
      const e = detail.evaluation_report
      if (!e) return []
      return [
        'Model trained and evaluated on held-out test data',
        `${Object.keys(e.test_metrics).length} test metric(s) computed`,
        `pass_fail = ${e.pass_fail}`,
        e.failure_analysis ? 'Rule-based failure_analysis generated' : 'No failure analysis needed',
      ]
    }
  }
}

export function AgentPanel({ detail, stage }: { detail: RunDetail; stage: StageConfig }) {
  const lines = deterministicLines(detail, stage)
  const routing = describeRouting(detail, stage.key)

  if (stage.key !== 'requirement' && lines.length === 0) {
    return <EmptyState title={`${stage.label} agent hasn't run yet`} />
  }

  let llmJudgment: React.ReactNode
  let notTrackedReason: string | undefined

  if (stage.key === 'requirement') {
    llmJudgment = (
      <div className="space-y-1 text-sm text-[var(--color-text-secondary)]">
        <p>
          <span className="text-[var(--color-text-tertiary)]">task_type:</span>{' '}
          <span className="font-mono text-violet-300">{detail.problem_spec.task_type}</span>
        </p>
        <p>
          <span className="text-[var(--color-text-tertiary)]">success_metric:</span>{' '}
          <span className="font-mono text-violet-300">{detail.problem_spec.success_metric}</span>
        </p>
        <p>
          <span className="text-[var(--color-text-tertiary)]">metric_threshold:</span>{' '}
          <span className="font-mono text-violet-300">{detail.problem_spec.metric_threshold}</span>
        </p>
      </div>
    )
    // The Requirement Agent's real LLM call happens BEFORE a PipelineRun
    // exists (see agents/requirement_agent.py's own comment: "trace_entry
    // isn't recorded to a PipelineRun here"). Its cost is real but
    // permanently untracked by this API — distinct from "hasn't run yet".
    notTrackedReason =
      'Not tracked — this call runs before the run record exists, so its cost/tokens are never persisted (a known backend limitation, not a bug in this UI).'
  } else if (stage.key === 'training') {
    // agrees_with_rule_based_recommendation is not persisted ANYWHERE —
    // see lib/judgment.ts's module comment for the full explanation.
    llmJudgment = (
      <span className="inline-flex items-center gap-1.5 text-sm text-[var(--color-text-tertiary)]">
        <MinusCircle className="h-3.5 w-3.5" />
        Not available — the backend does not persist this agent's advisory
        agreement flag past the run.
      </span>
    )
  } else {
    const historyStage = stage.historyStage!
    const proceed = parseProceed(detail.history, historyStage)
    llmJudgment = (
      <div>
        <ProceedBadge value={proceed} />
        <p className="mt-2 text-[11px] text-[var(--color-text-tertiary)]">
          Parsed from the run log ("proceed=…") — not a structured API field.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-5">
      <DecisionCard
        deterministic={lines}
        llmJudgment={
          <div>
            {llmJudgment}
            {stage.key !== 'requirement' && (
              <TraceStat trace={detail.trace} agentName={stage.historyStage!} />
            )}
            {stage.key === 'requirement' && (
              <TraceStat trace={detail.trace} agentName="requirement_agent" notTrackedReason={notTrackedReason} />
            )}
          </div>
        }
        routing={routing && <p className="text-sm text-[var(--color-text-secondary)]">{routing}</p>}
      />

      {stage.key === 'data' && detail.data_profile && <DataProfileView profile={detail.data_profile} />}
      {stage.key === 'feature' && detail.eda_report && <EdaReportView report={detail.eda_report} />}
      {stage.key === 'tuning' && detail.tuning_result && (
        <TuningResultView result={detail.tuning_result} />
      )}
      {stage.key === 'training' && detail.evaluation_report && (
        <EvaluationReportView report={detail.evaluation_report} />
      )}
    </div>
  )
}
