import { Sparkles } from 'lucide-react'
import type { TraceEntry } from '@/api/types'
import { formatMoney, formatTokens } from '@/lib/format'

/** §18: "If no trace entry exists yet for that agent, show a pending state,
 * not zeros." Finds the real matching TraceEntry by agent name; if none
 * exists, explains WHY rather than silently rendering $0.00 / 0 tokens,
 * which would misleadingly look like a real zero-cost call. */
export function TraceStat({
  trace,
  agentName,
  notTrackedReason,
}: {
  trace: TraceEntry[]
  agentName: string
  notTrackedReason?: string
}) {
  const entry = trace.find((t) => t.agent === agentName)

  if (!entry) {
    return (
      <div className="mt-3 rounded-md border border-dashed border-[var(--color-border-default)] px-3 py-2 text-xs text-[var(--color-text-tertiary)]">
        {notTrackedReason ?? 'No LLM call recorded yet for this agent.'}
      </div>
    )
  }

  return (
    <div className="mt-3 grid grid-cols-2 gap-x-3 gap-y-1.5 rounded-md border border-violet-500/20 bg-violet-500/[0.03] px-3 py-2 text-xs">
      <Stat label="Model" value={entry.model} mono />
      <Stat label="Latency" value={`${entry.latency_ms.toFixed(0)}ms`} mono />
      <Stat label="Tokens" value={`${formatTokens(entry.prompt_tokens)} in / ${formatTokens(entry.completion_tokens)} out`} mono />
      <Stat label="Cost" value={formatMoney(entry.estimated_cost_usd)} mono />
    </div>
  )
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <span className="flex items-center gap-1 text-[var(--color-text-tertiary)]">
        <Sparkles className="h-2.5 w-2.5" />
        {label}
      </span>
      <span className={mono ? 'font-mono text-violet-300' : 'text-violet-300'}>{value}</span>
    </div>
  )
}
