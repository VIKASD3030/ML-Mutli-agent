import type { RunDetail } from '@/api/types'
import type { StageKey } from './stages'

/**
 * Derives what the deterministic orchestrator actually did after each
 * stage, from real state + real history text — never from the LLM's
 * proceed value directly. Every string returned here traces back to an
 * exact substring written by pipeline/run_pipeline.py, verified against
 * the current file.
 */
export function describeRouting(detail: RunDetail, stage: StageKey): string | null {
  const { history, status } = detail

  switch (stage) {
    case 'requirement':
      // Reaching a run at all means Python already accepted the extracted
      // ProblemSpec (a ClarificationNeeded never becomes a run_id).
      return 'Accepted — a valid ProblemSpec was produced and the run was created.'

    case 'data': {
      if (history.some((h) => h.stage === 'data_agent' && h.event.startsWith('escalated:'))) {
        return 'Escalated — DataAgent did not recommend proceeding. The run halted here.'
      }
      if (detail.eda_report || detail.tuning_result || detail.evaluation_report) {
        return 'Continued to Feature stage.'
      }
      return null
    }

    case 'feature': {
      if (history.some((h) => h.stage === 'feature_agent' && h.event.startsWith('escalated:'))) {
        return 'Escalated — FeatureAgent did not recommend proceeding. The run halted here.'
      }
      if (detail.tuning_result || detail.evaluation_report) {
        return 'Continued to Tuning stage.'
      }
      return null
    }

    case 'tuning': {
      // The one deliberately asymmetric case: Tuning's proceed=False is
      // advisory ONLY and does not halt the pipeline — verified against
      // pipeline/run_pipeline.py, which logs a "note:" line and continues
      // to Training regardless of the LLM's opinion here.
      const flagged = history.find(
        (h) => h.stage === 'orchestrator' && h.event.includes('TuningAgent flagged concerns'),
      )
      if (flagged) {
        return 'Concerns noted, but the pipeline continued to Training anyway — Tuning\'s judgment is advisory only and never halts the run.'
      }
      if (detail.evaluation_report) {
        return 'Continued to Training stage.'
      }
      return null
    }

    case 'training': {
      if (status === 'done') return 'Run passed — pipeline complete.'
      if (!detail.evaluation_report) return null
      const nextStep = detail.evaluation_report.failure_analysis?.recommended_next_step
      if (!nextStep) return null
      const maxedOut = history.some((h) => h.event.includes('max_loop_backs') && h.event.includes('reached'))
      if (maxedOut) {
        return `Failed — ${nextStep}, but max_loop_backs was reached. Escalated to a human.`
      }
      if (nextStep === 'revisit_features' || nextStep === 'expand_hyperparam_search') {
        return `Failed — looping back (${nextStep}).`
      }
      return `Failed — ${nextStep} requires human input. Escalated.`
    }
  }
}
