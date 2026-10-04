import type { HistoryEntry } from '@/api/types'
import { formatTime } from '@/lib/format'
import { EmptyState } from '@/components/ui/States'

/** §17: built directly from the real history[] list — every line rendered
 * here is a real HistoryEntry, nothing synthesized around them. */
export function HistoryTimeline({ history }: { history: HistoryEntry[] }) {
  if (history.length === 0) {
    return <EmptyState title="No history yet" />
  }

  return (
    <ol className="space-y-0">
      {history.map((entry, i) => (
        <li key={i} className="relative flex gap-3 pb-4 pl-1 last:pb-0">
          {i < history.length - 1 && (
            <span className="absolute top-3 left-[7px] h-full w-px bg-[var(--color-border-subtle)]" />
          )}
          <span className="relative z-10 mt-1.5 h-[7px] w-[7px] shrink-0 rounded-full bg-[var(--color-accent-secondary)]" />
          <div className="min-w-0 flex-1">
            <div className="flex items-baseline gap-2">
              <span className="rounded bg-[var(--color-bg-surface-2)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--color-text-tertiary)]">
                {entry.stage}
              </span>
              <span className="font-mono text-[11px] text-[var(--color-text-tertiary)]">
                {formatTime(entry.timestamp)}
              </span>
            </div>
            <p className="mt-0.5 text-sm break-words text-[var(--color-text-secondary)]">
              {entry.event}
            </p>
          </div>
        </li>
      ))}
    </ol>
  )
}
