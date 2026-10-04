import { useState } from 'react'
import { Link } from 'react-router-dom'
import { useRuns } from '@/hooks/useRuns'
import { useCostAggregate } from '@/hooks/useCostAggregate'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { MetricCard } from '@/components/ui/MetricCard'
import { DataTable } from '@/components/ui/DataTable'
import { LoadingState, ErrorState, EmptyState } from '@/components/ui/States'
import { formatMoney, formatTokens, truncateId } from '@/lib/format'

const OPTIONS = [5, 10, 20]

/**
 * §29: "aggregate from real trace data across runs if that's efficiently
 * derivable client-side; otherwise show per-run observability only, and
 * say so rather than faking a trend chart." It is NOT efficiently
 * derivable — GET /runs omits cost/tokens entirely (see useCostAggregate's
 * module comment) — so this page does the honest middle ground: a real,
 * live, bounded N+1 fetch over the most recent runs, with the exact
 * sample size always visible and adjustable, never presented as an
 * all-time total.
 */
export function LlmUsagePage() {
  const { runs, error, loading, refetch } = useRuns()
  const [sampleSize, setSampleSize] = useState(10)
  const cost = useCostAggregate(runs, sampleSize)

  if (loading) return <LoadingState />
  if (error) return <ErrorState title="Could not load runs" description={error.message} onRetry={refetch} />
  if (!runs || runs.length === 0) return <EmptyState title="No runs yet" />

  return (
    <div>
      <SectionHeader
        title="LLM usage"
        description={`Live-computed from the ${cost.fetchedCount} most recent run(s) — GET /runs does not carry cost/token totals, so this is a real, bounded fetch, not a cached historical trend.`}
        action={
          <div className="flex gap-1">
            {OPTIONS.map((n) => (
              <button
                key={n}
                onClick={() => setSampleSize(n)}
                className={`rounded-md px-2.5 py-1 text-xs ${
                  sampleSize === n
                    ? 'bg-[var(--color-accent-primary)]/15 text-[var(--color-accent-primary)]'
                    : 'text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]'
                }`}
              >
                Last {n}
              </button>
            ))}
          </div>
        }
      />

      <div className="mb-6 grid grid-cols-3 gap-4">
        <MetricCard
          label={`Cost (last ${cost.fetchedCount})`}
          value={cost.loading ? null : formatMoney(cost.totalCostUsd)}
        />
        <MetricCard
          label={`Tokens (last ${cost.fetchedCount})`}
          value={cost.loading ? null : formatTokens(cost.totalTokens)}
        />
        <MetricCard
          label="Avg. cost / run"
          value={
            cost.loading || cost.fetchedCount === 0
              ? null
              : formatMoney(cost.totalCostUsd / cost.fetchedCount)
          }
        />
      </div>

      <DataTable
        keyFor={(r) => r.run_id}
        rows={cost.perRun}
        emptyLabel={cost.loading ? 'Loading…' : 'No data'}
        columns={[
          {
            header: 'Run',
            cell: (r) => (
              <Link to={`/runs/${r.run_id}`} className="text-[var(--color-accent-secondary)] hover:underline">
                {truncateId(r.run_id, 10)}
              </Link>
            ),
            mono: true,
          },
          { header: 'LLM calls', cell: (r) => r.callCount, align: 'right' },
          { header: 'Tokens', cell: (r) => formatTokens(r.tokens), align: 'right', mono: true },
          { header: 'Cost', cell: (r) => formatMoney(r.cost), align: 'right', mono: true },
        ]}
      />
    </div>
  )
}
