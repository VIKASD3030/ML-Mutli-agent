import { RotateCcw } from 'lucide-react'
import type { RunDetail } from '@/api/types'
import { loopBackEvents, wasFeatureAgentSkipped } from '@/lib/history'

/**
 * §19 of the design brief. Every value here is real:
 *   - loop_count / max_loop_backs are both real fields.
 *   - recommended_next_step is EvaluationReport.failure_analysis's real
 *     field, explicitly attributed to "deterministic failure_analysis",
 *     never to the LLM.
 *   - the skip line is derived per lib/history.ts's documented method —
 *     the only trace resume_from ever leaves in the API response.
 */
export function LoopBackBanner({ detail }: { detail: RunDetail }) {
  if (detail.loop_count === 0) return null

  const maxLoopBacks = detail.problem_spec.constraints.max_loop_backs
  const events = loopBackEvents(detail.history)
  const latest = events[events.length - 1]
  const skipped = wasFeatureAgentSkipped(detail.history)
  const nextStep = detail.evaluation_report?.failure_analysis?.recommended_next_step

  return (
    <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm font-semibold text-amber-400">
          <RotateCcw className="h-4 w-4" />
          Loop-back #{detail.loop_count}
        </div>
        <span className="font-mono text-xs text-amber-400/80">
          {detail.loop_count} / {maxLoopBacks} retries used
        </span>
      </div>

      {nextStep && (
        <p className="mb-1 text-sm text-[var(--color-text-secondary)]">
          Recommended next step:{' '}
          <span className="rounded bg-amber-500/10 px-1.5 py-0.5 font-mono text-xs text-amber-300">
            {nextStep}
          </span>
          <span className="ml-2 text-xs text-[var(--color-text-tertiary)]">
            Triggered by: deterministic failure_analysis — not the LLM.
          </span>
        </p>
      )}

      {skipped && (
        <p className="mt-2 text-sm text-[var(--color-text-secondary)]">
          Feature stage was skipped on resume — the previous feature set was
          reused unchanged, and only Tuning/Training re-ran.
        </p>
      )}

      {latest && (
        <p className="mt-2 font-mono text-xs text-[var(--color-text-tertiary)]">{latest.event}</p>
      )}
    </div>
  )
}
