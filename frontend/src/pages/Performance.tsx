import { useMemo } from 'react'
import { useRuns } from '@/hooks/useRuns'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { MetricCard } from '@/components/ui/MetricCard'
import { Panel } from '@/components/ui/Panel'
import { LoadingState, ErrorState, EmptyState } from '@/components/ui/States'

/**
 * Aggregated entirely from GET /runs (RunSummary) — no per-run detail
 * fetches, so this page is cheap and loads instantly even with a lot of
 * history. That's also its honest limit: task_type/pass_fail/metric_value/
 * loop_count are the only fields available at this level, so that's all
 * that's shown — no per-agent breakdown here (see the Live Run page for
 * that, on a specific run).
 */
export function PerformancePage() {
  const { runs, error, loading, refetch } = useRuns()

  const stats = useMemo(() => {
    if (!runs) return null
    const evaluated = runs.filter((r) => r.pass_fail !== null)
    const passed = evaluated.filter((r) => r.pass_fail === 'pass').length
    const failed = evaluated.length - passed
    const escalated = runs.filter((r) => r.status === 'escalated').length
    const avgLoops =
      runs.length > 0 ? runs.reduce((sum, r) => sum + r.loop_count, 0) / runs.length : null
    const byTask = new Map<string, { pass: number; fail: number }>()
    for (const r of evaluated) {
      const key = r.task_type ?? 'unknown'
      const entry = byTask.get(key) ?? { pass: 0, fail: 0 }
      if (r.pass_fail === 'pass') entry.pass++
      else entry.fail++
      byTask.set(key, entry)
    }
    return { evaluated: evaluated.length, passed, failed, escalated, avgLoops, byTask }
  }, [runs])

  if (loading) return <LoadingState />
  if (error) return <ErrorState title="Could not load runs" description={error.message} onRetry={refetch} />
  if (!runs || runs.length === 0) return <EmptyState title="No runs yet" />

  return (
    <div>
      <SectionHeader
        title="Performance"
        description="Aggregated from run summaries across the whole history."
      />

      <div className="mb-6 grid grid-cols-2 gap-4 lg:grid-cols-4">
        <MetricCard label="Evaluated runs" value={stats!.evaluated} />
        <MetricCard label="Passed" value={stats!.passed} />
        <MetricCard label="Failed" value={stats!.failed} />
        <MetricCard label="Escalated" value={stats!.escalated} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel>
          <p className="mb-3 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Pass / fail by task type
          </p>
          {stats!.byTask.size === 0 ? (
            <p className="text-sm text-[var(--color-text-tertiary)]">
              No evaluated runs yet to break down.
            </p>
          ) : (
            <div className="space-y-3">
              {Array.from(stats!.byTask.entries()).map(([task, { pass, fail }]) => {
                const total = pass + fail
                return (
                  <div key={task}>
                    <div className="mb-1 flex justify-between text-xs text-[var(--color-text-secondary)]">
                      <span className="font-mono">{task}</span>
                      <span>
                        {pass} pass / {fail} fail
                      </span>
                    </div>
                    <div className="flex h-2 overflow-hidden rounded-full bg-[var(--color-bg-surface-2)]">
                      <div
                        className="bg-emerald-500"
                        style={{ width: `${(pass / total) * 100}%` }}
                      />
                      <div className="bg-rose-500" style={{ width: `${(fail / total) * 100}%` }} />
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </Panel>

        <Panel>
          <p className="mb-3 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase">
            Loop-back activity
          </p>
          <p className="font-mono text-2xl font-semibold text-[var(--color-text-primary)]">
            {stats!.avgLoops !== null ? stats!.avgLoops.toFixed(2) : '—'}
          </p>
          <p className="text-xs text-[var(--color-text-tertiary)]">Average loop_count across all runs</p>
        </Panel>
      </div>
    </div>
  )
}
