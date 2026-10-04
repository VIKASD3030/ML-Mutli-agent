/**
 * Everything here derives a UI-visible fact from the REAL history[] log
 * text — never adds narrative that isn't already in an entry. The exact
 * strings matched below were read directly out of pipeline/run_pipeline.py
 * (verified against the current file, not assumed). Matching is done with
 * substring `includes()` rather than exact equality so a minor wording
 * tweak in the backend doesn't silently break detection — but the
 * substrings themselves are the backend's own words, not invented ones.
 */
import type { HistoryEntry, RunDetail } from '@/api/types'

/** `resume_from` is a LOCAL VARIABLE inside run_pipeline.py's loop — it is
 * never persisted on PipelineRun and never appears in the API response.
 * The only observable trace of a feature-stage skip is this exact log line,
 * written when resume_from !== "feature":
 *   run_state.log("orchestrator", "skipping feature_agent — resuming from
 *   tuning, features unchanged")
 * This function is the ONLY place that "skip" fact is derived — nothing
 * else in the app assumes a skip happened. */
export function wasFeatureAgentSkipped(history: HistoryEntry[]): boolean {
  return history.some((h) => h.event.includes('skipping feature_agent'))
}

/** The most recent stage-log entry for a given agent stage, if the agent has
 * run at least once. Used to derive a real "when did this stage last
 * complete" timestamp — never a fabricated ETA or percentage. */
export function lastEntryForStage(
  history: HistoryEntry[],
  stage: string,
): HistoryEntry | undefined {
  for (let i = history.length - 1; i >= 0; i--) {
    if (history[i].stage === stage) return history[i]
  }
  return undefined
}

/** Loop-back events: run_pipeline.py logs exactly
 *   "fail -> recommended_next_step={next_step} (TrainingAgent agrees:
 *   {bool}), loop_count={n}"
 * on every failed training attempt that leads to a routing decision. */
export function loopBackEvents(history: HistoryEntry[]): HistoryEntry[] {
  return history.filter((h) => h.event.includes('fail -> recommended_next_step='))
}

/** Why did this run escalate? Prefers the real, structured
 * failure_analysis (the deterministic rule ladder's own explanation) over
 * free-text history, since it is the more precise and more clearly
 * attributable source. Falls back to the most recent history entry whose
 * text contains "escalat" — every escalation path in run_pipeline.py logs
 * a line containing that substring before returning. Returns null only if
 * genuinely nothing is available (should not happen for a run whose
 * status is actually "escalated", but this function must not throw or
 * fabricate a reason if it does). */
export function escalationReason(detail: RunDetail): string | null {
  const analysis = detail.evaluation_report?.failure_analysis
  if (analysis) {
    return `${analysis.summary} (recommended next step: ${analysis.recommended_next_step})`
  }
  for (let i = detail.history.length - 1; i >= 0; i--) {
    if (detail.history[i].event.toLowerCase().includes('escalat')) {
      return detail.history[i].event
    }
  }
  return null
}
