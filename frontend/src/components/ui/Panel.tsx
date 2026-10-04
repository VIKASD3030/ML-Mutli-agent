import type { ReactNode } from 'react'
import clsx from 'clsx'

export function Panel({
  children,
  className,
  padded = true,
}: {
  children: ReactNode
  className?: string
  padded?: boolean
}) {
  return (
    <div
      className={clsx(
        'rounded-lg border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)]',
        padded && 'p-5',
        className,
      )}
    >
      {children}
    </div>
  )
}
