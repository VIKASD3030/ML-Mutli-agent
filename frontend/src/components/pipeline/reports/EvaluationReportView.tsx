import type { EvaluationReport } from '@/api/types'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { passFailMeta } from '@/lib/status'
import { formatNumber } from '@/lib/format'
import { AlertTriangle } from 'lucide-react'

/** §24-25: primary metric shown prominently, then the rest of the real
 * test_metrics. Branches on which of confusion_matrix / residual_summary
 * is present rather than assuming classification — a regression run
 * legitimately has confusion_matrix: null and residual_summary populated,
 * and this component follows whichever is actually there. */
export function EvaluationReportView({ report }: { report: EvaluationReport }) {
  const meta = passFailMeta(report.pass_fail)
  const otherMetrics = Object.entries(report.test_metrics).filter(
    ([name]) => name !== report.metric_optimized,
  )

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4 rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] p-4">
        <StatusBadge {...meta} />
        <div className="flex-1">
          <p className="text-xs text-[var(--color-text-tertiary)] uppercase">
            {report.metric_optimized}
          </p>
          <p className="font-mono text-2xl font-semibold text-[var(--color-text-primary)]">
            {formatNumber(report.metric_value)}
            <span className="ml-2 text-sm font-normal text-[var(--color-text-tertiary)]">
              / threshold {formatNumber(report.metric_threshold)}
            </span>
          </p>
        </div>
      </div>

      {report.failure_analysis && (
        <div className="rounded-lg border border-rose-500/30 bg-rose-500/5 p-4">
          <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-rose-300">
            <AlertTriangle className="h-4 w-4" />
            Failure analysis
          </div>
          <p className="text-sm text-[var(--color-text-secondary)]">
            {report.failure_analysis.summary}
          </p>
          <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
            Recommended next step:{' '}
            <span className="font-mono text-rose-300">
              {report.failure_analysis.recommended_next_step}
            </span>
          </p>
        </div>
      )}

      {otherMetrics.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            All test metrics
          </p>
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
            {otherMetrics.map(([name, value]) => (
              <div
                key={name}
                className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2"
              >
                <p className="font-mono text-[11px] text-[var(--color-text-tertiary)]">{name}</p>
                <p className="font-mono text-sm text-[var(--color-text-primary)]">
                  {formatNumber(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {report.confusion_matrix && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Confusion matrix
          </p>
          <ConfusionMatrix matrix={report.confusion_matrix} />
        </div>
      )}

      {report.residual_summary && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Residual summary
          </p>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
            {Object.entries(report.residual_summary).map(([name, value]) => (
              <div
                key={name}
                className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2"
              >
                <p className="font-mono text-[11px] text-[var(--color-text-tertiary)]">{name}</p>
                <p className="font-mono text-sm text-[var(--color-text-primary)]">
                  {formatNumber(value)}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function ConfusionMatrix({ matrix }: { matrix: number[][] }) {
  const max = Math.max(...matrix.flat())
  return (
    <div className="inline-block rounded-lg border border-[var(--color-border-subtle)] p-3">
      {matrix.map((row, i) => (
        <div key={i} className="flex gap-1">
          {row.map((cell, j) => {
            const intensity = max > 0 ? cell / max : 0
            return (
              <div
                key={j}
                className="flex h-12 w-12 items-center justify-center rounded font-mono text-sm"
                style={{
                  backgroundColor: `color-mix(in srgb, var(--color-accent-secondary) ${intensity * 60}%, var(--color-bg-surface-2))`,
                  color: intensity > 0.5 ? '#0a0b0f' : 'var(--color-text-primary)',
                }}
              >
                {cell}
              </div>
            )
          })}
        </div>
      ))}
    </div>
  )
}
