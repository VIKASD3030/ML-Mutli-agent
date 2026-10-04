import { ChevronRight } from 'lucide-react'
import clsx from 'clsx'
import type { RunDetail } from '@/api/types'
import { STAGES } from '@/lib/stages'
import { stageState } from '@/lib/stages'
import { stageStateMeta, toneClasses } from '@/lib/status'

export function PipelineStageRow({
  detail,
  activeStage,
  onSelectStage,
}: {
  detail: RunDetail
  activeStage: string | null
  onSelectStage: (key: string) => void
}) {
  return (
    <div className="flex items-stretch gap-1 overflow-x-auto">
      {STAGES.map((stage, i) => {
        const state = stageState(detail, stage)
        const meta = stageStateMeta(state)
        const classes = toneClasses[meta.tone]
        const isActive = activeStage === stage.key
        return (
          <div key={stage.key} className="flex flex-1 items-center gap-1 min-w-[140px]">
            <button
              onClick={() => onSelectStage(stage.key)}
              className={clsx(
                'flex w-full flex-col gap-2 rounded-lg border px-4 py-3 text-left transition',
                isActive
                  ? 'border-[var(--color-accent-primary)] bg-[var(--color-bg-surface-2)]'
                  : 'border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)] hover:border-[var(--color-border-default)]',
              )}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-[var(--color-text-primary)]">
                  {stage.label}
                </span>
                <meta.icon className={clsx('h-4 w-4', classes.text, meta.spin && 'animate-spin')} />
              </div>
              <span className={clsx('text-xs font-medium', classes.text)}>{meta.label}</span>
            </button>
            {i < STAGES.length - 1 && (
              <ChevronRight className="h-4 w-4 shrink-0 text-[var(--color-text-tertiary)]" />
            )}
          </div>
        )
      })}
    </div>
  )
}
