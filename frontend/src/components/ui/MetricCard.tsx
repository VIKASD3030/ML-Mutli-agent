import type { LucideIcon } from 'lucide-react'
import { Panel } from './Panel'
import clsx from 'clsx'

/** value === null renders as an em dash, never a fabricated placeholder
 * number — this is the load-bearing honesty rule for every KPI card in the
 * app (§10, §38 of the design brief). */
export function MetricCard({
  label,
  value,
  icon: Icon,
  hint,
  accent = false,
}: {
  label: string
  value: string | number | null
  icon?: LucideIcon
  hint?: string
  accent?: boolean
}) {
  return (
    <Panel className="flex flex-col gap-2">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium tracking-wide text-[var(--color-text-secondary)] uppercase">
          {label}
        </span>
        {Icon && <Icon className="h-4 w-4 text-[var(--color-text-tertiary)]" />}
      </div>
      <span
        className={clsx(
          'font-mono text-2xl font-semibold tabular-nums',
          accent ? 'text-[var(--color-accent-secondary)]' : 'text-[var(--color-text-primary)]',
          value === null && 'text-[var(--color-text-tertiary)]',
        )}
      >
        {value === null ? '—' : value}
      </span>
      {hint && <span className="text-xs text-[var(--color-text-tertiary)]">{hint}</span>}
    </Panel>
  )
}
