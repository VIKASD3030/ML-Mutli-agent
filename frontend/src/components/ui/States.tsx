import type { ReactNode } from 'react'
import { Loader2, Inbox, WifiOff, RotateCw } from 'lucide-react'
import type { LucideIcon } from 'lucide-react'

export function LoadingState({ label = 'Loading…' }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-[var(--color-text-tertiary)]">
      <Loader2 className="h-6 w-6 animate-spin" />
      <span className="text-sm">{label}</span>
    </div>
  )
}

export function EmptyState({
  icon: Icon = Inbox,
  title,
  description,
  action,
}: {
  icon?: LucideIcon
  title: string
  description?: string
  action?: ReactNode
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 py-16 text-center">
      <div className="rounded-full bg-[var(--color-bg-surface-2)] p-3">
        <Icon className="h-5 w-5 text-[var(--color-text-tertiary)]" />
      </div>
      <div>
        <p className="text-sm font-medium text-[var(--color-text-primary)]">{title}</p>
        {description && (
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{description}</p>
        )}
      </div>
      {action}
    </div>
  )
}

/** "API unavailable" must be shown honestly with a Retry action — never
 * silently fall back to stale or fake data. This is the ONLY component in
 * the app that renders on a failed fetch; there is no fallback-to-cache
 * path anywhere. */
export function ErrorState({
  title,
  description,
  onRetry,
}: {
  title: string
  description?: string
  onRetry?: () => void
}) {
  return (
    <div className="flex flex-col items-center justify-center gap-3 rounded-lg border border-rose-500/30 bg-rose-500/5 py-16 text-center">
      <WifiOff className="h-5 w-5 text-rose-400" />
      <div>
        <p className="text-sm font-medium text-rose-300">{title}</p>
        {description && <p className="mt-1 text-sm text-rose-400/80">{description}</p>}
      </div>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-1 inline-flex items-center gap-1.5 rounded-md border border-rose-500/40 px-3 py-1.5 text-xs font-medium text-rose-300 transition hover:bg-rose-500/10"
        >
          <RotateCw className="h-3.5 w-3.5" />
          Retry
        </button>
      )}
    </div>
  )
}
