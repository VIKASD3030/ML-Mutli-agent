import type { DataProfile } from '@/api/types'
import { MetricCard } from '@/components/ui/MetricCard'
import { WarningBanner } from '@/components/ui/WarningBanner'
import { DataTable } from '@/components/ui/DataTable'
import { AlertOctagon } from 'lucide-react'

export function DataProfileView({ profile }: { profile: DataProfile }) {
  return (
    <div className="space-y-4">
      {profile.blocking_issue && (
        <div className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
          <AlertOctagon className="mt-0.5 h-4 w-4 shrink-0" />
          <span>{profile.blocking_issue}</span>
        </div>
      )}

      <div className="grid grid-cols-3 gap-3">
        <MetricCard label="Rows" value={profile.row_count.toLocaleString()} />
        <MetricCard label="Columns" value={profile.column_count} />
        <MetricCard label="Cleaning actions" value={profile.cleaning_actions_taken.length} />
      </div>

      {profile.class_balance && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Class balance
          </p>
          <div className="flex gap-2">
            {Object.entries(profile.class_balance).map(([label, proportion]) => (
              <div
                key={label}
                className="flex-1 rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2"
              >
                <p className="font-mono text-xs text-[var(--color-text-tertiary)]">{label}</p>
                <p className="font-mono text-sm text-[var(--color-text-primary)]">
                  {(proportion * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      <WarningBanner title="Warnings" items={profile.warnings} tone="warning" />

      {profile.outlier_flags.length > 0 && (
        <div className="text-sm text-[var(--color-text-secondary)]">
          <span className="font-medium text-[var(--color-text-primary)]">Outlier-flagged columns: </span>
          {profile.outlier_flags.join(', ')}
        </div>
      )}

      {profile.cleaning_actions_taken.length > 0 && (
        <div>
          <p className="mb-1.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Cleaning actions
          </p>
          <DataTable
            keyFor={(row) => `${row.column}-${row.action}`}
            columns={[
              { header: 'Column', cell: (r) => r.column, mono: true },
              { header: 'Action', cell: (r) => r.action, mono: true },
              { header: 'Rationale', cell: (r) => r.rationale },
            ]}
            rows={profile.cleaning_actions_taken}
          />
        </div>
      )}
    </div>
  )
}
