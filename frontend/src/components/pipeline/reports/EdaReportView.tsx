import type { EDAReport } from '@/api/types'
import { MetricCard } from '@/components/ui/MetricCard'
import { WarningBanner } from '@/components/ui/WarningBanner'
import { DataTable } from '@/components/ui/DataTable'

/** §22: leakage_warnings and data_quality_warnings are deliberately shown
 * as two SEPARATE banners with different tones (danger vs warning) — they
 * are two distinct concerns on the real backend and must never be merged
 * into one list or treated as equal severity. Stage labels below name the
 * real logic in pipeline/feature_stage.py, not invented step names. */
export function EdaReportView({ report }: { report: EDAReport }) {
  const topCorrelations = Object.entries(report.correlation_summary)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 12)

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Final features" value={report.final_feature_names.length} />
        <MetricCard label="Engineered" value={report.engineered_features.length} />
        <MetricCard label="Dropped" value={report.dropped_features.length} />
      </div>

      {/* Leakage FIRST and visually louder — higher-priority concern. */}
      <WarningBanner title="Leakage warnings" items={report.leakage_warnings} tone="danger" />
      <WarningBanner title="Data quality warnings" items={report.data_quality_warnings} tone="warning" />

      {report.target_distribution_notes && (
        <p className="text-sm text-[var(--color-text-secondary)]">
          {report.target_distribution_notes}
        </p>
      )}

      {topCorrelations.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Correlation with target (top {topCorrelations.length})
          </p>
          <div className="space-y-1">
            {topCorrelations.map(([name, corr]) => (
              <div key={name} className="flex items-center gap-2 text-xs">
                <span className="w-40 truncate font-mono text-[var(--color-text-secondary)]">
                  {name}
                </span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--color-bg-surface-2)]">
                  <div
                    className="h-full rounded-full bg-[var(--color-accent-secondary)]"
                    style={{ width: `${Math.min(corr, 1) * 100}%` }}
                  />
                </div>
                <span className="w-12 text-right font-mono text-[var(--color-text-tertiary)]">
                  {corr.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {report.engineered_features.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Engineered features
          </p>
          <DataTable
            keyFor={(r) => r.name}
            columns={[
              { header: 'Feature', cell: (r) => r.name, mono: true },
              { header: 'Rationale', cell: (r) => r.rationale },
            ]}
            rows={report.engineered_features}
          />
        </div>
      )}

      {report.dropped_features.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Dropped features
          </p>
          <DataTable
            keyFor={(r) => r.name}
            columns={[
              { header: 'Feature', cell: (r) => r.name, mono: true },
              { header: 'Rationale', cell: (r) => r.rationale },
            ]}
            rows={report.dropped_features}
          />
        </div>
      )}
    </div>
  )
}
