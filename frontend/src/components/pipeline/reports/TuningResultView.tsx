import type { TuningResult } from '@/api/types'
import { MetricCard } from '@/components/ui/MetricCard'
import { TuningChart } from '@/components/charts/TuningChart'
import { formatNumber } from '@/lib/format'

/** §23: converged is real, genuine plateau-based detection — represented
 * honestly as a status derived from that boolean, never as a generic
 * "optimizing…" spinner. No mid-tuning trial-by-trial progress is shown
 * DURING a run: search_history only exists once the whole TuningResult is
 * written, after the stage completes — the backend has no endpoint for
 * partial/in-flight trial data. */
export function TuningResultView({ result }: { result: TuningResult }) {
  const bestParams = Object.entries(result.best_params)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-4 gap-3">
        <MetricCard label="Best CV score" value={formatNumber(result.best_cv_score)} accent />
        <MetricCard
          label="Trials"
          value={`${result.search_budget_used} / ${result.search_budget_total}`}
        />
        <MetricCard label="Converged" value={result.converged ? 'Yes' : 'No'} />
        <MetricCard label="Metric optimized" value={result.metric_optimized} />
      </div>

      {result.convergence_notes && (
        <p className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] p-3 text-sm text-[var(--color-text-secondary)]">
          {result.convergence_notes}
        </p>
      )}

      {result.search_history.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Trial score history ({result.search_history.length} trials, best highlighted)
          </p>
          <TuningChart history={result.search_history} />
        </div>
      )}

      {bestParams.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Best parameters
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {bestParams.map(([key, value]) => (
              <div
                key={key}
                className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2"
              >
                <p className="font-mono text-[11px] text-[var(--color-text-tertiary)]">{key}</p>
                <p className="font-mono text-sm text-[var(--color-text-primary)]">
                  {String(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
