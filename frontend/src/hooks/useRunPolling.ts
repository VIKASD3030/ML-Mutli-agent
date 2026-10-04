import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '@/api/client'
import type { RunDetail } from '@/api/types'
import { TERMINAL_STATUSES } from '@/api/types'

const POLL_INTERVAL_MS = 2500

/** Polls GET /runs/{run_id} on an interval WHILE the run is not in a
 * terminal state, and stops automatically once it is — matching this
 * project's existing streamlit_app.py reference implementation, which
 * uses the same 2-3s interval and the same terminal-status stop condition
 * (see its `_live_fragment`). A manual refetch is always available too. */
export function useRunPolling(runId: string | null) {
  const [detail, setDetail] = useState<RunDetail | null>(null)
  const [error, setError] = useState<Error | null>(null)
  const [loading, setLoading] = useState(true)
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const fetchOnce = useCallback(async () => {
    if (!runId) return
    try {
      const d = await api.getRun(runId)
      setDetail(d)
      setError(null)
      return d
    } catch (e) {
      setError(e as Error)
      throw e
    } finally {
      setLoading(false)
    }
  }, [runId])

  useEffect(() => {
    if (!runId) return
    let cancelled = false
    setLoading(true)

    const tick = () => {
      fetchOnce()
        .then((d) => {
          if (cancelled || !d) return
          if (!TERMINAL_STATUSES.includes(d.status)) {
            timerRef.current = setTimeout(tick, POLL_INTERVAL_MS)
          }
        })
        .catch(() => {
          // Stop polling on error rather than hammering an unreachable API;
          // the manual refetch() below remains available to retry.
        })
    }
    tick()

    return () => {
      cancelled = true
      if (timerRef.current) clearTimeout(timerRef.current)
    }
  }, [runId, fetchOnce])

  const isTerminal = detail ? TERMINAL_STATUSES.includes(detail.status) : false

  return { detail, error, loading, isTerminal, refetch: fetchOnce }
}
