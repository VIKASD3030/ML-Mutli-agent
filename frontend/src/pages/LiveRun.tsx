import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { Braces, Clock, Copy, ListTree, RotateCw, Sparkles, Workflow } from 'lucide-react'
import { useRunPolling } from '@/hooks/useRunPolling'
import { LoadingState, ErrorState, EmptyState } from '@/components/ui/States'
import { StatusBadge } from '@/components/ui/StatusBadge'
import { MetricCard } from '@/components/ui/MetricCard'
import { DataTable } from '@/components/ui/DataTable'
import { Panel } from '@/components/ui/Panel'
import { PipelineStageRow } from '@/components/pipeline/PipelineStageRow'
import { AgentPanel } from '@/components/pipeline/AgentPanel'
import { LoopBackBanner } from '@/components/pipeline/LoopBackBanner'
import { HistoryTimeline } from '@/components/pipeline/HistoryTimeline'
import { STAGES } from '@/lib/stages'
import { runStatusMeta, passFailMeta } from '@/lib/status'
import { escalationReason } from '@/lib/history'
import { formatDuration, formatMoney, formatTokens, truncateId } from '@/lib/format'
import { ApiError } from '@/api/client'
import type { RunDetail } from '@/api/types'

type Tab = 'pipeline' | 'timeline' | 'trace' | 'raw'

export function LiveRunPage() {
  const { runId } = useParams<{ runId: string }>()
  const { detail, error, loading, isTerminal, refetch } = useRunPolling(runId ?? null)
  const [activeStage, setActiveStage] = useState<string>('training')
  const [tab, setTab] = useState<Tab>('pipeline')

  if (!runId) return <EmptyState title="No run selected" />
  if (loading && !detail) return <LoadingState label="Loading run…" />
  if (error) {
    const notFound = error instanceof ApiError && error.status === 404
    return (
      <ErrorState
        title={notFound ? `No run found with id ${runId}` : 'Could not load this run'}
        description={error.message}
        onRetry={() => refetch()}
      />
    )
  }
  if (!detail) return <EmptyState title="No data" />

  return (
    <div className="space-y-5">
      <RunHeader detail={detail} isTerminal={isTerminal} onRefresh={() => refetch()} />

      {detail.status === 'escalated' && <EscalationBanner detail={detail} />}
      {detail.status === 'done' && detail.evaluation_report && (
        <PassBanner report={detail.evaluation_report} />
      )}

      <LoopBackBanner detail={detail} />

      <div className="flex gap-1 border-b border-[var(--color-border-subtle)]">
        <TabButton active={tab === 'pipeline'} onClick={() => setTab('pipeline')} icon={Workflow} label="Pipeline" />
        <TabButton active={tab === 'timeline'} onClick={() => setTab('timeline')} icon={Clock} label="Timeline" />
        <TabButton active={tab === 'trace'} onClick={() => setTab('trace')} icon={Sparkles} label="LLM Trace" />
        <TabButton active={tab === 'raw'} onClick={() => setTab('raw')} icon={Braces} label="Raw Outputs" />
      </div>

      {tab === 'pipeline' && (
        <div className="space-y-5">
          <PipelineStageRow detail={detail} activeStage={activeStage} onSelectStage={setActiveStage} />
          <Panel>
            <AgentPanel detail={detail} stage={STAGES.find((s) => s.key === activeStage)!} />
          </Panel>
        </div>
      )}

      {tab === 'timeline' && (
        <Panel>
          <HistoryTimeline history={detail.history} />
        </Panel>
      )}

      {tab === 'trace' && <TraceTab detail={detail} />}
      {tab === 'raw' && <RawOutputsTab detail={detail} />}
    </div>
  )
}

function TabButton({
  active,
  onClick,
  icon: Icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: typeof Workflow
  label: string
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center gap-1.5 border-b-2 px-3 py-2 text-sm font-medium transition ${
        active
          ? 'border-[var(--color-accent-primary)] text-[var(--color-text-primary)]'
          : 'border-transparent text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]'
      }`}
    >
      <Icon className="h-3.5 w-3.5" />
      {label}
    </button>
  )
}

function RunHeader({
  detail,
  isTerminal,
  onRefresh,
}: {
  detail: RunDetail
  isTerminal: boolean
  onRefresh: () => void
}) {
  const meta = runStatusMeta(detail.status)
  const [copied, setCopied] = useState(false)

  return (
    <Panel className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <StatusBadge {...meta} />
          <button
            onClick={() => {
              navigator.clipboard.writeText(detail.run_id)
              setCopied(true)
              setTimeout(() => setCopied(false), 1200)
            }}
            className="flex items-center gap-1.5 font-mono text-xs text-[var(--color-text-tertiary)] hover:text-[var(--color-text-secondary)]"
            title={detail.run_id}
          >
            {truncateId(detail.run_id, 12)}
            <Copy className="h-3 w-3" />
            {copied && <span className="text-emerald-400">copied</span>}
          </button>
        </div>
        <div className="flex items-center gap-2">
          {!isTerminal && (
            <span className="flex items-center gap-1.5 text-xs text-[var(--color-text-tertiary)]">
              <RotateCw className="h-3 w-3 animate-spin" /> Auto-refreshing every 2.5s
            </span>
          )}
          <button
            onClick={onRefresh}
            className="rounded-md border border-[var(--color-border-default)] px-2.5 py-1 text-xs text-[var(--color-text-secondary)] hover:bg-[var(--color-bg-surface-2)]"
          >
            Refresh now
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <HeaderStat label="Dataset" value={detail.problem_spec.data_source} mono />
        <HeaderStat label="Problem type" value={detail.problem_spec.task_type} />
        <HeaderStat label="Target" value={detail.problem_spec.target_column} mono />
        <HeaderStat
          label="Loop-backs"
          value={`${detail.loop_count} / ${detail.problem_spec.constraints.max_loop_backs}`}
        />
      </div>

      <div className="grid grid-cols-2 gap-3 border-t border-[var(--color-border-subtle)] pt-4 sm:grid-cols-4">
        <MetricCard label="Total cost" value={formatMoney(detail.total_cost_usd)} />
        <MetricCard label="Total tokens" value={formatTokens(detail.total_tokens)} />
        <MetricCard label="LLM calls" value={detail.trace.length} />
        <MetricCard
          label="Duration"
          value={
            detail.history.length >= 2
              ? (formatDuration(
                  detail.history[0].timestamp,
                  detail.history[detail.history.length - 1].timestamp,
                ) ?? '< 1s')
              : null
          }
          hint={detail.history.length >= 2 ? undefined : 'Not enough history yet'}
        />
      </div>
    </Panel>
  )
}

function HeaderStat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <p className="text-xs text-[var(--color-text-tertiary)] uppercase">{label}</p>
      <p className={`truncate text-sm text-[var(--color-text-primary)] ${mono ? 'font-mono' : ''}`}>
        {value}
      </p>
    </div>
  )
}

function EscalationBanner({ detail }: { detail: RunDetail }) {
  const reason = escalationReason(detail)
  return (
    <div className="rounded-lg border border-rose-500/30 bg-rose-500/10 p-4">
      <p className="text-sm font-semibold text-rose-300">Run escalated to a human.</p>
      {reason && <p className="mt-1 text-sm text-rose-200/80">{reason}</p>}
    </div>
  )
}

function PassBanner({ report }: { report: NonNullable<RunDetail['evaluation_report']> }) {
  const meta = passFailMeta(report.pass_fail)
  return (
    <div className="flex items-center gap-4 rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-4">
      <StatusBadge {...meta} size="md" />
      <p className="text-sm text-emerald-200">
        {report.metric_optimized} = {report.metric_value.toFixed(4)} (threshold{' '}
        {report.metric_threshold})
      </p>
    </div>
  )
}

function TraceTab({ detail }: { detail: RunDetail }) {
  return (
    <Panel padded={false}>
      <div className="p-4">
        <p className="text-xs text-[var(--color-text-tertiary)]">
          Every LLM call this run has made so far, in order. RequirementAgent's call (if any) is
          not included — its cost is real but never persisted onto this run; see the Requirement
          stage panel for why.
        </p>
      </div>
      <DataTable
        keyFor={(t) => `${t.agent}-${t.timestamp}`}
        emptyLabel="No LLM calls recorded yet."
        columns={[
          { header: 'Agent', cell: (t) => t.agent, mono: true },
          { header: 'Model', cell: (t) => t.model, mono: true },
          { header: 'Prompt tokens', cell: (t) => formatTokens(t.prompt_tokens), align: 'right' },
          { header: 'Completion tokens', cell: (t) => formatTokens(t.completion_tokens), align: 'right' },
          { header: 'Latency', cell: (t) => `${t.latency_ms.toFixed(0)}ms`, align: 'right' },
          { header: 'Cost', cell: (t) => formatMoney(t.estimated_cost_usd), align: 'right' },
        ]}
        rows={detail.trace}
      />
    </Panel>
  )
}

function RawOutputsTab({ detail }: { detail: RunDetail }) {
  const [expanded, setExpanded] = useState<StageObjKey | null>(null)
  const objects: { key: StageObjKey; label: string; value: unknown }[] = [
    { key: 'problem_spec', label: 'ProblemSpec', value: detail.problem_spec },
    { key: 'data_profile', label: 'DataProfile', value: detail.data_profile },
    { key: 'eda_report', label: 'EDAReport', value: detail.eda_report },
    { key: 'tuning_result', label: 'TuningResult', value: detail.tuning_result },
    { key: 'evaluation_report', label: 'EvaluationReport', value: detail.evaluation_report },
  ]

  return (
    <div className="space-y-2">
      {objects.map((obj) => (
        <Panel key={obj.key} padded={false}>
          <button
            onClick={() => setExpanded(expanded === obj.key ? null : obj.key)}
            className="flex w-full items-center justify-between px-4 py-3 text-left"
          >
            <div className="flex items-center gap-2">
              <ListTree className="h-3.5 w-3.5 text-[var(--color-text-tertiary)]" />
              <span className="text-sm font-medium text-[var(--color-text-primary)]">
                {obj.label}
              </span>
              {obj.value === null && (
                <span className="text-xs text-[var(--color-text-tertiary)]">not available yet</span>
              )}
            </div>
          </button>
          {expanded === obj.key && obj.value !== null && (
            <pre className="overflow-x-auto border-t border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)] p-4 font-mono text-xs text-[var(--color-text-secondary)]">
              {JSON.stringify(obj.value, null, 2)}
            </pre>
          )}
        </Panel>
      ))}
    </div>
  )
}

type StageObjKey = 'problem_spec' | 'data_profile' | 'eda_report' | 'tuning_result' | 'evaluation_report'
