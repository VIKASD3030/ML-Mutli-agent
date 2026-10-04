import { useMemo } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { Plus, Activity } from 'lucide-react'
import { useRuns } from '@/hooks/useRuns'
import { useCostAggregate } from '@/hooks/useCostAggregate'
import { useHealth } from '@/hooks/useHealth'
import { MetricCard } from '@/components/ui/MetricCard'
import { Panel } from '@/components/ui/Panel'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { DataTable } from '@/components/ui/DataTable'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { ErrorState, EmptyState } from '@/components/ui/States'
import { runStatusMeta } from '@/lib/status'
import { formatDate, formatDuration, formatMoney, truncateId } from '@/lib/format'
import { TERMINAL_STATUSES } from '@/api/types'

const COST_SAMPLE_SIZE = 8

export function OverviewPage() {
  const navigate = useNavigate()
  const { runs, error, refetch } = useRuns()
  const cost = useCostAggregate(runs, COST_SAMPLE_SIZE)
  const apiConnected = useHealth()

  const kpis = useMemo(() => {
    if (!runs) return null
    const terminal = runs.filter((r) => TERMINAL_STATUSES.includes(r.status))
    const evaluated = runs.filter((r) => r.pass_fail !== null)
    const passed = evaluated.filter((r) => r.pass_fail === 'pass')
    const durations = terminal
      .map((r) => {
        const ms = new Date(r.updated_at).getTime() - new Date(r.created_at).getTime()
        return ms > 0 ? ms : null
      })
      .filter((ms): ms is number => ms !== null)
    const avgMs = durations.length > 0 ? durations.reduce((a, b) => a + b, 0) / durations.length : null

    return {
      total: runs.length,
      successRate: evaluated.length > 0 ? passed.length / evaluated.length : null,
      avgDurationMs: avgMs,
      evaluatedCount: evaluated.length,
    }
  }, [runs])

  if (error) {
    return <ErrorState title="Could not load runs" description={error.message} onRetry={refetch} />
  }

  return (
    <div>
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-[var(--color-text-primary)]">
            {apiConnected ? 'Welcome back.' : 'API unreachable'}
          </h2>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">
            Autonomous ML Pipeline — five agents, each wrapping a deterministic tool with an
            advisory LLM judgment.
          </p>
        </div>
        <Link
          to="/new-run"
          className="flex items-center gap-1.5 rounded-md bg-[var(--color-accent-primary)] px-4 py-2 text-sm font-medium text-white transition hover:bg-[var(--color-accent-primary-hover)]"
        >
          <Plus className="h-4 w-4" /> New Pipeline Run
        </Link>
      </div>

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Total runs" value={kpis?.total ?? null} />
        <MetricCard
          label="Success rate"
          value={
            kpis === null
              ? null
              : kpis.successRate === null
                ? null
                : `${(kpis.successRate * 100).toFixed(0)}%`
          }
          hint={kpis && kpis.evaluatedCount === 0 ? 'No evaluated runs yet' : undefined}
        />
        <MetricCard
          label="Avg. runtime"
          value={
            kpis?.avgDurationMs != null
              ? formatDuration(new Date(0).toISOString(), new Date(kpis.avgDurationMs).toISOString())
              : null
          }
          hint={kpis && kpis.avgDurationMs === null ? 'No completed runs yet' : undefined}
        />
        <MetricCard
          label="LLM cost"
          value={cost.loading ? null : formatMoney(cost.totalCostUsd)}
          hint={`Last ${cost.fetchedCount} run(s) — not a full historical total`}
        />
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <SectionHeader title="Recent runs" />
          {runs && runs.length === 0 ? (
            <EmptyState
              title="No runs yet"
              description="Start your first pipeline run to see it here."
              action={
                <Link
                  to="/new-run"
                  className="rounded-md bg-[var(--color-accent-primary)] px-3 py-1.5 text-xs font-medium text-white"
                >
                  New Pipeline Run
                </Link>
              }
            />
          ) : (
            <DataTable
              keyFor={(r) => r.run_id}
              rows={(runs ?? []).slice(0, 8)}
              onRowClick={(r) => navigate(`/runs/${r.run_id}`)}
              columns={[
                {
                  header: 'Status',
                  cell: (r) => {
                    const meta = runStatusMeta(r.status)
                    return <StatusBadge {...meta} size="sm" />
                  },
                },
                { header: 'Run ID', cell: (r) => truncateId(r.run_id, 10), mono: true },
                { header: 'Task', cell: (r) => r.task_type ?? '—' },
                { header: 'Metric', cell: (r) => (r.metric_value !== null ? r.metric_value.toFixed(4) : '—'), align: 'right', mono: true },
                { header: 'Created', cell: (r) => formatDate(r.created_at) },
              ]}
            />
          )}
        </div>

        <div>
          <SectionHeader title="System health" />
          <Panel className="space-y-3">
            <HealthRow label="API" ok={apiConnected} />
            <p className="text-[11px] text-[var(--color-text-tertiary)]">
              Database and pipeline-engine status are not shown separately — the backend exposes
              no route that reports them independently of the API's own liveness.
            </p>
          </Panel>
        </div>
      </div>
    </div>
  )
}

function HealthRow({ label, ok }: { label: string; ok: boolean | null }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="flex items-center gap-2 text-[var(--color-text-secondary)]">
        <Activity className="h-3.5 w-3.5" />
        {label}
      </span>
      <span
        className={
          ok === null
            ? 'text-[var(--color-text-tertiary)]'
            : ok
              ? 'text-emerald-400'
              : 'text-rose-400'
        }
      >
        {ok === null ? 'Checking…' : ok ? '● Connected' : '● Disconnected'}
      </span>
    </div>
  )
}
