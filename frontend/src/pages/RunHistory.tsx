import { useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { RotateCw } from 'lucide-react'
import { useRuns } from '@/hooks/useRuns'
import { LoadingState, ErrorState, EmptyState } from '@/components/ui/States'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { DataTable } from '@/components/ui/DataTable'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { runStatusMeta, passFailMeta } from '@/lib/status'
import { formatDate, truncateId } from '@/lib/format'
import type { RunStatus } from '@/api/types'

export function RunHistoryPage() {
  const { runs, error, loading, refetch } = useRuns()
  const navigate = useNavigate()

  const [statusFilter, setStatusFilter] = useState<string>('all')
  const [taskFilter, setTaskFilter] = useState<string>('all')
  const [sortBy, setSortBy] = useState<'created_at' | 'metric_value' | 'loop_count'>('created_at')

  const filtered = useMemo(() => {
    if (!runs) return []
    return runs
      .filter((r) => statusFilter === 'all' || r.status === statusFilter)
      .filter((r) => taskFilter === 'all' || r.task_type === taskFilter)
      .slice()
      .sort((a, b) => {
        if (sortBy === 'created_at') return b.created_at.localeCompare(a.created_at)
        if (sortBy === 'metric_value') return (b.metric_value ?? -Infinity) - (a.metric_value ?? -Infinity)
        return b.loop_count - a.loop_count
      })
  }, [runs, statusFilter, taskFilter, sortBy])

  return (
    <div>
      <SectionHeader
        title="Runs"
        description="Every run this backend has recorded. Dataset/data_source is not shown here — GET /runs is deliberately lightweight and omits it; open a run to see its full ProblemSpec."
        action={
          <button
            onClick={() => refetch()}
            className="flex items-center gap-1.5 rounded-md border border-[var(--color-border-default)] px-2.5 py-1.5 text-xs text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-surface-2)]"
          >
            <RotateCw className="h-3 w-3" /> Refresh
          </button>
        }
      />

      {loading && <LoadingState />}
      {error && <ErrorState title="Could not load runs" description={error.message} onRetry={refetch} />}

      {runs && runs.length === 0 && (
        <EmptyState
          title="No runs yet"
          description="Create your first run to see it appear here."
        />
      )}

      {runs && runs.length > 0 && (
        <>
          <div className="mb-4 flex flex-wrap gap-3">
            <Select
              value={statusFilter}
              onChange={setStatusFilter}
              options={['all', 'created', 'data_ready', 'features_ready', 'tuned', 'evaluated', 'done', 'escalated']}
              label="Status"
            />
            <Select
              value={taskFilter}
              onChange={setTaskFilter}
              options={['all', 'classification', 'regression']}
              label="Task"
            />
            <Select
              value={sortBy}
              onChange={(v) => setSortBy(v as typeof sortBy)}
              options={['created_at', 'metric_value', 'loop_count']}
              label="Sort by"
            />
          </div>

          <DataTable
            keyFor={(r) => r.run_id}
            onRowClick={(r) => navigate(`/runs/${r.run_id}`)}
            columns={[
              { header: 'Status', cell: (r) => <StatusCell status={r.status} /> },
              { header: 'Run ID', cell: (r) => truncateId(r.run_id, 10), mono: true },
              { header: 'Task', cell: (r) => r.task_type ?? '—', mono: true },
              { header: 'Result', cell: (r) => (r.pass_fail ? <PassFailCell v={r.pass_fail} /> : '—') },
              { header: 'Metric', cell: (r) => (r.metric_value !== null ? r.metric_value.toFixed(4) : '—'), align: 'right', mono: true },
              { header: 'Loops', cell: (r) => r.loop_count, align: 'right' },
              { header: 'Created', cell: (r) => formatDate(r.created_at) },
            ]}
            rows={filtered}
          />
        </>
      )}
    </div>
  )
}

function StatusCell({ status }: { status: RunStatus }) {
  const meta = runStatusMeta(status)
  return <StatusBadge {...meta} size="sm" />
}

function PassFailCell({ v }: { v: 'pass' | 'fail' }) {
  const meta = passFailMeta(v)
  return <StatusBadge {...meta} size="sm" />
}

function Select({
  value,
  onChange,
  options,
  label,
}: {
  value: string
  onChange: (v: string) => void
  options: string[]
  label: string
}) {
  return (
    <label className="flex items-center gap-2 text-xs text-[var(--color-text-tertiary)]">
      {label}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="rounded-md border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-2 py-1 text-xs text-[var(--color-text-primary)] outline-none"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
    </label>
  )
}
