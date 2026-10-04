/**
 * A finding from the §0 ground-truth audit that goes beyond what the design
 * brief itself anticipated, worth stating precisely:
 *
 * Every agent's LLM judgment (DataAgentResult, FeatureAgentResult,
 * TuningAgentResult, TrainingAgentResult — all defined in agents/*.py) is a
 * Python-only return type. It is NEVER attached to PipelineRun as a
 * structured field, so it never reaches the API as JSON. What DOES survive,
 * for Data/Feature/Tuning only, is a single log line:
 *   pipeline_run.log("data_agent", f"proceed={judgment.proceed}")
 * (and the equivalent for feature_agent / tuning_agent) — i.e. the boolean
 * exists only as a substring inside a free-text history event.
 *
 * For Training specifically, `agrees_with_rule_based_recommendation` is not
 * persisted in ANY form — not as a field, not even as log text (the
 * training_agent.py log line only carries `pass_fail`, which is already a
 * first-class EvaluationReport field). There is therefore no honest way to
 * show it; see AgentPanel's Training case for the fallback this produces.
 */
import type { HistoryEntry } from '@/api/types'

export type Proceed = true | false | 'unknown'

/** Parses "proceed=True" / "proceed=False" out of the most recent history
 * entry for the given stage. Returns 'unknown' if the agent has not logged
 * a proceed line yet (hasn't run) — this is intentionally distinct from
 * `false`, which would misleadingly claim a real negative judgment. */
export function parseProceed(history: HistoryEntry[], stage: string): Proceed {
  for (let i = history.length - 1; i >= 0; i--) {
    const entry = history[i]
    if (entry.stage !== stage) continue
    if (entry.event.includes('proceed=True')) return true
    if (entry.event.includes('proceed=False')) return false
  }
  return 'unknown'
}
