import { AlertTriangle, ShieldAlert, Info } from 'lucide-react'
import type { Tone } from '@/lib/status'
import { toneClasses } from '@/lib/status'
import clsx from 'clsx'

const iconFor: Record<'danger' | 'warning' | 'neutral', typeof AlertTriangle> = {
  danger: ShieldAlert,
  warning: AlertTriangle,
  neutral: Info,
}

/** Generic banner for a list of plain-text warnings (leakage_warnings,
 * data_quality_warnings, DataProfile.warnings — all of which are
 * list[str] on the backend, never structured objects). Renders nothing if
 * the list is empty, so callers can render it unconditionally. */
export function WarningBanner({
  title,
  items,
  tone,
}: {
  title: string
  items: string[]
  tone: Extract<Tone, 'danger' | 'warning' | 'neutral'>
}) {
  if (items.length === 0) return null
  const Icon = iconFor[tone]
  const classes = toneClasses[tone]
  return (
    <div className={clsx('rounded-lg border p-4', classes.bg, classes.border)}>
      <div className={clsx('mb-2 flex items-center gap-2 text-sm font-semibold', classes.text)}>
        <Icon className="h-4 w-4" />
        {title}
        <span className="text-xs font-normal opacity-70">({items.length})</span>
      </div>
      <ul className="space-y-1.5 text-sm text-[var(--color-text-secondary)]">
        {items.map((item, i) => (
          <li key={i} className="pl-6 leading-relaxed">
            {item}
          </li>
        ))}
      </ul>
    </div>
  )
}
