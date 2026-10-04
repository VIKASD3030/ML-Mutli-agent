import type { ReactNode } from 'react'
import { Check, Cpu, GitBranch, Sparkles } from 'lucide-react'
import { toneClasses } from '@/lib/status'

/**
 * THE core differentiator visualization (§16 of the design brief): every
 * agent panel is split into three visually distinct zones so the
 * deterministic/LLM/routing separation reads at a glance, not as a wall of
 * mixed text —
 *
 *   DETERMINISTIC — real pandas/sklearn/Optuna work, always neutral gray/blue
 *   LLM JUDGMENT  — the model's advisory opinion, always purple, reserved
 *                   for this and nothing else in the app
 *   ROUTING       — what the deterministic orchestrator actually decided,
 *                   shown separately from the LLM's opinion so it's never
 *                   implied the LLM controls it directly
 */
export function DecisionCard({
  deterministic,
  llmJudgment,
  routing,
}: {
  deterministic: string[]
  llmJudgment: ReactNode
  routing?: ReactNode
}) {
  return (
    <div className="grid gap-3 md:grid-cols-3">
      <Zone
        icon={Cpu}
        label="Deterministic"
        tone="neutral"
        subtitle="Real computation, no model involved"
      >
        <ul className="space-y-1.5">
          {deterministic.map((line, i) => (
            <li key={i} className="flex items-start gap-2 text-sm text-[var(--color-text-secondary)]">
              <Check className="mt-0.5 h-3.5 w-3.5 shrink-0 text-emerald-400" />
              <span>{line}</span>
            </li>
          ))}
        </ul>
      </Zone>

      <Zone icon={Sparkles} label="LLM Judgment" tone="llm" subtitle="Advisory opinion only">
        {llmJudgment}
      </Zone>

      <Zone
        icon={GitBranch}
        label="Routing"
        tone="active"
        subtitle="Decided by orchestrator code, not the LLM"
      >
        {routing ?? (
          <span className="text-sm text-[var(--color-text-tertiary)]">
            No routing decision yet.
          </span>
        )}
      </Zone>
    </div>
  )
}

function Zone({
  icon: Icon,
  label,
  subtitle,
  tone,
  children,
}: {
  icon: typeof Check
  label: string
  subtitle: string
  tone: 'neutral' | 'llm' | 'active'
  children: ReactNode
}) {
  const classes = toneClasses[tone]
  return (
    <div className={`rounded-lg border p-4 ${classes.border} ${classes.bg}`}>
      <div className="mb-1 flex items-center gap-1.5">
        <Icon className={`h-3.5 w-3.5 ${classes.text}`} />
        <span className={`text-xs font-semibold tracking-wide uppercase ${classes.text}`}>
          {label}
        </span>
      </div>
      <p className="mb-3 text-[11px] text-[var(--color-text-tertiary)]">{subtitle}</p>
      {children}
    </div>
  )
}
