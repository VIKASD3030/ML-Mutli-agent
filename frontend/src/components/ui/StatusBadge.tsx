import clsx from 'clsx'
import type { Tone } from '@/lib/status'
import { toneClasses } from '@/lib/status'
import type { LucideIcon } from 'lucide-react'

/**
 * Status must never be color-only (accessibility requirement) — every badge
 * pairs its color with an icon AND a text label, always.
 */
export function StatusBadge({
  label,
  tone,
  icon: Icon,
  spin = false,
  size = 'md',
}: {
  label: string
  tone: Tone
  icon: LucideIcon
  spin?: boolean
  size?: 'sm' | 'md'
}) {
  const classes = toneClasses[tone]
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-md border font-medium',
        classes.text,
        classes.bg,
        classes.border,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-sm',
      )}
    >
      <Icon className={clsx(size === 'sm' ? 'h-3 w-3' : 'h-3.5 w-3.5', spin && 'animate-spin')} />
      {label}
    </span>
  )
}

export function Dot({ tone }: { tone: Tone }) {
  return <span className={clsx('inline-block h-2 w-2 rounded-full', toneClasses[tone].dot)} />
}
