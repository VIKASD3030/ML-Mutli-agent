import type { ReactNode } from 'react'
import { EmptyState } from './States'

export interface Column<T> {
  header: string
  cell: (row: T) => ReactNode
  align?: 'left' | 'right'
  mono?: boolean
}

export function DataTable<T>({
  columns,
  rows,
  keyFor,
  emptyLabel = 'Nothing here yet.',
  onRowClick,
}: {
  columns: Column<T>[]
  rows: T[]
  keyFor: (row: T) => string
  emptyLabel?: string
  onRowClick?: (row: T) => void
}) {
  if (rows.length === 0) {
    return <EmptyState title={emptyLabel} />
  }

  return (
    <div className="overflow-x-auto rounded-lg border border-[var(--color-border-subtle)]">
      <table className="w-full min-w-max text-left text-sm">
        <thead>
          <tr className="border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)]">
            {columns.map((col) => (
              <th
                key={col.header}
                className={`px-4 py-2.5 text-xs font-medium tracking-wide text-[var(--color-text-tertiary)] uppercase ${
                  col.align === 'right' ? 'text-right' : 'text-left'
                }`}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr
              key={keyFor(row)}
              onClick={onRowClick ? () => onRowClick(row) : undefined}
              className={`border-b border-[var(--color-border-subtle)] last:border-0 ${
                onRowClick ? 'cursor-pointer hover:bg-[var(--color-bg-surface-2)]' : ''
              }`}
            >
              {columns.map((col) => (
                <td
                  key={col.header}
                  className={`px-4 py-2.5 text-[var(--color-text-secondary)] ${
                    col.align === 'right' ? 'text-right' : 'text-left'
                  } ${col.mono ? 'font-mono text-xs' : ''}`}
                >
                  {col.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
