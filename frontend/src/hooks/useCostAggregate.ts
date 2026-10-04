import { useEffect, useState } from 'react'
import { api } from '@/api/client'
import type { RunSummary } from '@/api/types'

export interface CostAggregate {
  totalCostUsd: number
  totalTokens: number
  perRun: { run_id: string; cost: number; tokens: number; callCount: number }[]
  fetchedCount: number
  loading: boolean
}

/**
 * GET /runs (RunSummary) deliberately omits cost/token totals — those only
 * exist on the full RunDetail. There is no endpoint that aggregates cost
 * across runs server-side, so an honest "total LLM spend" figure can only
 * be computed by fetching each run's full detail individually (an N+1
 * pattern) — never by pretending RunSummary already has it.
 *
 * This hook fetches the `limit` most recent runs' full detail in parallel
 * and sums their real total_cost_usd/total_tokens. It is capped and the
 * cap is always surfaced back to the caller (`fetchedCount`) so no UI ever
 * claims to show "total cost" when it only actually covers a subset.
 */
export function useCostAggregate(runs: RunSummary[] | null, limit: number): CostAggregate {
  const [state, setState] = useState<CostAggregate>({
    totalCostUsd: 0,
    totalTokens: 0,
    perRun: [],
    fetchedCount: 0,
    loading: true,
  })

  useEffect(() => {
    if (!runs) return
    let cancelled = false
    const subset = runs.slice(0, limit)

    setState((s) => ({ ...s, loading: true }))
    Promise.all(
      subset.map((r) =>
        api
          .getRun(r.run_id)
          .then((d) => ({
            run_id: r.run_id,
            cost: d.total_cost_usd,
            tokens: d.total_tokens,
            callCount: d.trace.length,
          }))
          .catch(() => null),
      ),
    ).then((results) => {
      if (cancelled) return
      const perRun = results.filter((r): r is NonNullable<typeof r> => r !== null)
      setState({
        totalCostUsd: perRun.reduce((sum, r) => sum + r.cost, 0),
        totalTokens: perRun.reduce((sum, r) => sum + r.tokens, 0),
        perRun,
        fetchedCount: perRun.length,
        loading: false,
      })
    })

    return () => {
      cancelled = true
    }
  }, [runs, limit])

  return state
}
