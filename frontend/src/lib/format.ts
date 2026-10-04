export function formatMoney(value: number): string {
  if (value === 0) return '$0.000000'
  // 6 decimals: per-call LLM cost is routinely sub-cent (see
  // pipeline/observability.py's PRICING_PER_1K_TOKENS), so 2 decimals
  // would round almost every real value to "$0.00" and hide the number
  // entirely.
  return `$${value.toFixed(6)}`
}

export function formatTokens(value: number): string {
  return value.toLocaleString('en-US')
}

export function formatNumber(value: number, decimals = 4): string {
  return value.toFixed(decimals)
}

export function formatPercent(value: number, decimals = 1): string {
  return `${(value * 100).toFixed(decimals)}%`
}

export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

export function formatTime(iso: string): string {
  return new Date(iso).toLocaleTimeString(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

/** Duration between two real ISO timestamps. Used for total run duration
 * (RunSummary.created_at -> updated_at, or first -> last history entry) and
 * for per-stage duration (consecutive stage-log timestamps) — never a
 * fabricated or estimated value. Returns null if end is before/equal to
 * start (e.g. a run with a single history entry so far). */
export function formatDuration(startIso: string, endIso: string): string | null {
  const ms = new Date(endIso).getTime() - new Date(startIso).getTime()
  if (ms <= 0) return null
  const totalSeconds = Math.round(ms / 1000)
  if (totalSeconds < 60) return `${totalSeconds}s`
  const minutes = Math.floor(totalSeconds / 60)
  const seconds = totalSeconds % 60
  if (minutes < 60) return `${minutes}m ${seconds}s`
  const hours = Math.floor(minutes / 60)
  return `${hours}h ${minutes % 60}m`
}

export function truncateId(id: string, length = 8): string {
  return `${id.slice(0, length)}…`
}
