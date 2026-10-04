import { useEffect, useState } from 'react'
import { api } from '@/api/client'

/** Real connection status, polled every 10s. This is the ONLY source for
 * the sidebar's "API ● Connected" indicator — there is deliberately no
 * "Database ● Connected" indicator anywhere in the app, because no route
 * exposes DB connectivity separately from the API's own liveness, and
 * faking one would violate the truthfulness rule. */
export function useHealth() {
  const [connected, setConnected] = useState<boolean | null>(null)

  useEffect(() => {
    let cancelled = false
    const check = () => {
      api
        .health()
        .then(() => !cancelled && setConnected(true))
        .catch(() => !cancelled && setConnected(false))
    }
    check()
    const id = setInterval(check, 10_000)
    return () => {
      cancelled = true
      clearInterval(id)
    }
  }, [])

  return connected
}
