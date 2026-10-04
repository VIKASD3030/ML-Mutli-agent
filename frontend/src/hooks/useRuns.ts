import { useCallback, useEffect, useState } from 'react'
import { api } from '@/api/client'
import type { RunSummary } from '@/api/types'

/** GET /runs — the lightweight list, used by Overview, Run History, and the
 * global command palette equivalent. No polling: the caller decides when a
 * refresh matters (a manual button, or navigating back to the page). */
export function useRuns() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)

  const refetch = useCallback(() => {
    setLoading(true)
    return api
      .listRuns()
      .then((r) => {
        setRuns(r)
        setError(null)
      })
      .catch((e) => setError(e as Error))
      .finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    refetch()
  }, [refetch])

  return { runs, error, loading, refetch }
}
