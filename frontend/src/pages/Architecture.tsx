import { useState } from 'react'
import { ArrowDown, Cpu, Database, GitBranch, Server, Sparkles, User } from 'lucide-react'
import { SectionHeader } from '@/components/ui/SectionHeader'
import { Panel } from '@/components/ui/Panel'
import { STAGES } from '@/lib/stages'

interface Node {
  id: string
  label: string
  detail: string
  layer: 'user' | 'deterministic' | 'llm' | 'infra'
  icon: typeof User
  reportSchema?: string
}

const NODES: Node[] = [
  { id: 'user', label: 'User', detail: 'Uploads a dataset and describes (or configures) the prediction problem.', layer: 'user', icon: User },
  ...STAGES.map((s) => ({
    id: s.key,
    label: `${s.label} Agent`,
    detail: `${s.deterministicTool} → ${s.llmResponsibility}`,
    layer: 'llm' as const,
    icon: Sparkles,
    reportSchema:
      s.reportKey === 'data_profile' ? 'DataProfile'
      : s.reportKey === 'eda_report' ? 'EDAReport'
      : s.reportKey === 'tuning_result' ? 'TuningResult'
      : s.reportKey === 'evaluation_report' ? 'EvaluationReport'
      : 'ProblemSpec',
  })),
  {
    id: 'orchestrator',
    label: 'Deterministic Orchestrator',
    detail:
      'pipeline/run_pipeline.py — pure Python. Reads each agent\'s advisory judgment plus failure_analysis and decides sequencing, loop-backs, and escalation. The LLM never controls routing directly.',
    layer: 'deterministic',
    icon: GitBranch,
  },
  {
    id: 'pipelinerun',
    label: 'PipelineRun',
    detail: 'The single persisted state object for one run — status, loop_count, all five stage reports, history log, and the LLM trace.',
    layer: 'infra',
    icon: Database,
  },
  { id: 'postgres', label: 'PostgreSQL', detail: 'Fully relational persistence for PipelineRun, via SQLAlchemy.', layer: 'infra', icon: Database },
  { id: 'fastapi', label: 'FastAPI', detail: 'POST /uploads, POST /runs, GET /runs, GET /runs/{run_id} — the only surface this frontend talks to.', layer: 'infra', icon: Server },
  { id: 'frontend', label: 'This frontend', detail: 'Plain HTTP calls only — no LLM client of any kind, ever.', layer: 'infra', icon: Cpu },
]

const LAYER_COLOR: Record<Node['layer'], string> = {
  user: 'border-slate-500/30 bg-slate-500/5 text-slate-300',
  deterministic: 'border-sky-500/30 bg-sky-500/5 text-sky-300',
  llm: 'border-violet-500/30 bg-violet-500/5 text-violet-300',
  infra: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-300',
}

export function ArchitecturePage() {
  const [selected, setSelected] = useState<Node>(NODES[0])

  return (
    <div>
      <SectionHeader
        title="Architecture"
        description="Every agent pairs a deterministic tool with an advisory LLM judgment — routing always stays deterministic code."
      />

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-2 lg:col-span-2">
          {NODES.map((node, i) => (
            <div key={node.id}>
              <button
                onClick={() => setSelected(node)}
                className={`flex w-full items-center gap-3 rounded-lg border px-4 py-3 text-left transition ${LAYER_COLOR[node.layer]} ${
                  selected.id === node.id ? 'ring-1 ring-[var(--color-accent-primary)]' : ''
                }`}
              >
                <node.icon className="h-4 w-4 shrink-0" />
                <span className="text-sm font-medium">{node.label}</span>
                {node.layer === 'llm' && (
                  <span className="ml-auto rounded bg-violet-500/20 px-1.5 py-0.5 text-[10px] uppercase">
                    LLM
                  </span>
                )}
                {node.layer === 'deterministic' && (
                  <span className="ml-auto rounded bg-sky-500/20 px-1.5 py-0.5 text-[10px] uppercase">
                    Deterministic
                  </span>
                )}
              </button>
              {i < NODES.length - 1 && (
                <div className="flex justify-center py-1">
                  <ArrowDown className="h-3.5 w-3.5 text-[var(--color-text-tertiary)]" />
                </div>
              )}
            </div>
          ))}
        </div>

        <div>
          <Panel>
            <p className="mb-1 text-sm font-semibold text-[var(--color-text-primary)]">{selected.label}</p>
            <p className="mb-3 text-sm text-[var(--color-text-secondary)]">{selected.detail}</p>
            {selected.reportSchema && (
              <div className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2">
                <p className="text-[11px] text-[var(--color-text-tertiary)] uppercase">Output schema</p>
                <p className="font-mono text-sm text-[var(--color-text-primary)]">{selected.reportSchema}</p>
              </div>
            )}
          </Panel>
        </div>
      </div>
    </div>
  )
}
