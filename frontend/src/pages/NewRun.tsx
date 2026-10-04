import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import clsx from 'clsx'
import {
  AlertCircle,
  Database,
  Loader2,
  Rocket,
  SlidersHorizontal,
  Sparkles,
  Upload,
} from 'lucide-react'
import { api, ApiError, ApiUnreachableError, isClarificationResponse } from '@/api/client'
import type { ClassificationMetric, ProblemSpec, RegressionMetric, TaskType } from '@/api/types'
import { CLASSIFICATION_METRICS, REGRESSION_METRICS } from '@/api/types'
import { useStudio } from '@/context/StudioContext'

const DEFAULT_BUILTIN_CONTEXT =
  'Predict whether the tumor is malignant (target column diagnosis), aim for at least 0.9 f1.'

type Tab = 'agent' | 'structured'

function errorMessage(e: unknown): string {
  if (e instanceof ApiUnreachableError || e instanceof ApiError) return e.message
  return 'Something went wrong.'
}

export function NewRunPage() {
  const [tab, setTab] = useState<Tab>('agent')

  return (
    <div>
      <h2 className="flex items-center gap-3 text-[32px] leading-tight font-bold text-[var(--color-text-primary)]">
        <span aria-hidden>🚀</span>
        Initialize New Pipeline Run
      </h2>
      <p className="mt-3 text-[15px] text-[var(--color-text-secondary)]">
        Provide your dataset and plain-English intent. The{' '}
        <strong className="text-[var(--color-text-primary)]">Requirement Agent</strong> inspects schema
        peeks (never raw rows) and builds a validated{' '}
        <code className="font-mono text-[13px]">ProblemSpec</code> for the autonomous orchestrator.
      </p>
      <hr className="my-6 border-[var(--color-border-subtle)]" />

      <DatasetPeekCard />

      <section className="mt-8 rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)]">
        <div className="flex gap-8 border-b border-[var(--color-border-subtle)] px-6">
          <TabButton active={tab === 'agent'} onClick={() => setTab('agent')} icon={Sparkles}>
            Requirement Agent (Natural Language Context)
          </TabButton>
          <TabButton active={tab === 'structured'} onClick={() => setTab('structured')} icon={SlidersHorizontal}>
            Structured ProblemSpec Form
          </TabButton>
        </div>
        <div className="p-6">{tab === 'agent' ? <AgentForm /> : <StructuredForm />}</div>
      </section>
    </div>
  )
}

function TabButton({
  active,
  onClick,
  icon: Icon,
  children,
}: {
  active: boolean
  onClick: () => void
  icon: typeof Sparkles
  children: string
}) {
  return (
    <button
      onClick={onClick}
      className={clsx(
        '-mb-px flex items-center gap-2.5 border-b-2 py-4 text-[15px] font-medium transition',
        active
          ? 'border-[var(--color-accent-secondary)] text-[var(--color-accent-secondary)]'
          : 'border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]',
      )}
    >
      <Icon className="h-[18px] w-[18px]" />
      {children}
    </button>
  )
}

// ---------------------------------------------------------------------------
// Dataset peek
// ---------------------------------------------------------------------------

const COLLAPSED_ROWS = 7

function formatPct(n: number): string {
  return `${Number.isInteger(n) ? n : n.toString()}%`
}

function DatasetPeekCard() {
  const { activeLabel, benchmark, upload, setUpload, peek, peekError, peekLoading } = useStudio()
  const fileInput = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => setExpanded(false), [activeLabel])

  const onFile = async (file: File) => {
    setUploading(true)
    setUploadError(null)
    try {
      setUpload(await api.uploadDataset(file))
    } catch (e) {
      setUploadError(errorMessage(e))
    } finally {
      setUploading(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const rows = peek ? (expanded ? peek.columns : peek.columns.slice(0, COLLAPSED_ROWS)) : []

  return (
    <section className="rounded-2xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)] p-6">
      <div className="flex items-start justify-between gap-4">
        <h3 className="flex items-center gap-3 text-lg font-bold text-[var(--color-text-primary)]">
          <Database className="h-5 w-5 text-[var(--color-accent-secondary)]" />
          Active Dataset Peek: {activeLabel}
        </h3>
        {peek && (
          <span className="shrink-0 rounded-full bg-[var(--color-bg-surface-3)] px-3.5 py-1.5 font-mono text-xs text-[var(--color-text-secondary)]">
            {peek.row_count} rows · {peek.column_count} columns
          </span>
        )}
      </div>
      <p className="mt-3 text-[15px] text-[var(--color-text-secondary)]">
        {upload
          ? 'Uploaded file — only this schema summary is shown to the agents, never raw rows.'
          : benchmark.description}
      </p>

      <div className="mt-5 overflow-hidden rounded-xl border border-[var(--color-border-subtle)] bg-[var(--color-bg-canvas)]">
        {peekLoading ? (
          <div className="flex items-center justify-center gap-2 py-12 text-sm text-[var(--color-text-tertiary)]">
            <Loader2 className="h-4 w-4 animate-spin" /> Reading schema…
          </div>
        ) : peekError ? (
          <div className="flex items-start gap-2 p-5 text-sm text-rose-300">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{peekError}</span>
          </div>
        ) : (
          <div className={clsx(expanded && 'max-h-[28rem] overflow-y-auto')}>
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-[var(--color-bg-canvas)]">
                <tr className="border-b border-[var(--color-border-subtle)] font-mono text-xs text-[var(--color-text-secondary)]">
                  <th className="px-4 py-4 font-medium">Column Name</th>
                  <th className="px-4 py-4 font-medium">Data Type</th>
                  <th className="px-4 py-4 font-medium">Missing %</th>
                  <th className="px-4 py-4 font-medium">Cardinality</th>
                  <th className="px-4 py-4 font-medium">Sample Values</th>
                </tr>
              </thead>
              <tbody className="font-mono text-[13px]">
                {rows.map((c) => (
                  <tr key={c.name} className="border-b border-[var(--color-border-subtle)] last:border-0">
                    <td className="px-4 py-3.5 font-semibold text-[var(--color-text-primary)]">{c.name}</td>
                    <td className="px-4 py-3.5 text-[var(--color-text-tertiary)]">{c.dtype}</td>
                    <td className="px-4 py-3.5 text-[var(--color-text-tertiary)]">{formatPct(c.missing_pct)}</td>
                    <td className="px-4 py-3.5 text-[var(--color-text-tertiary)]">{c.unique_count}</td>
                    <td className="px-4 py-3.5 text-[var(--color-text-primary)]">{c.example_value.join(', ')}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            {peek && peek.columns.length > COLLAPSED_ROWS && (
              <button
                onClick={() => setExpanded((v) => !v)}
                className="w-full border-t border-[var(--color-border-subtle)] py-2.5 text-xs font-medium text-[var(--color-accent-secondary)] hover:bg-[var(--color-bg-surface)]"
              >
                {expanded ? 'Show fewer columns' : `Show all ${peek.columns.length} columns`}
              </button>
            )}
          </div>
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-between gap-3 border-t border-[var(--color-border-subtle)] pt-5">
        <p className="text-[15px] text-[var(--color-text-secondary)]">
          Want to test your own data? Upload a .csv file:
          {upload && (
            <button
              onClick={() => setUpload(null)}
              className="ml-3 text-[13px] font-medium text-[var(--color-accent-secondary)] hover:underline"
            >
              Reset to benchmark
            </button>
          )}
        </p>
        <div className="flex items-center gap-3">
          {uploadError && <span className="text-sm text-rose-400">{uploadError}</span>}
          <button
            onClick={() => fileInput.current?.click()}
            disabled={uploading}
            className="flex items-center gap-2 rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-4 py-2.5 text-sm font-semibold text-[var(--color-text-primary)] transition hover:border-[var(--color-accent-secondary)] disabled:opacity-60"
          >
            {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
            Upload CSV
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.parquet"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0]
              if (file) onFile(file)
            }}
          />
        </div>
      </div>
    </section>
  )
}

// ---------------------------------------------------------------------------
// Forms
// ---------------------------------------------------------------------------

const inputClass =
  'w-full rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-3.5 py-2.5 text-[15px] text-[var(--color-text-primary)] outline-none focus:border-[var(--color-accent-primary)]'

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="flex items-start gap-2 rounded-lg border border-rose-500/30 bg-rose-500/10 p-3 text-sm text-rose-300">
      <AlertCircle className="mt-0.5 h-4 w-4 shrink-0" />
      <span>{message}</span>
    </div>
  )
}

function SubmitButton({ submitting, disabled, label }: { submitting: boolean; disabled: boolean; label: string }) {
  return (
    <button
      type="submit"
      disabled={submitting || disabled}
      className="flex items-center gap-2 rounded-lg bg-[var(--color-accent-primary)] px-5 py-2.5 text-sm font-semibold text-white transition hover:bg-[var(--color-accent-primary-hover)] disabled:cursor-not-allowed disabled:opacity-50"
    >
      {submitting ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
      {submitting ? 'Starting…' : label}
    </button>
  )
}

function AgentForm() {
  const navigate = useNavigate()
  const { activeSource, upload, maxTrials, maxLoopBacks } = useStudio()
  const [context, setContext] = useState(DEFAULT_BUILTIN_CONTEXT)
  const [touched, setTouched] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [clarification, setClarification] = useState<{
    question_for_user: string
    missing_fields: string[]
  } | null>(null)

  // The seeded example is about the breast-cancer benchmark; once another
  // dataset is active and the user hasn't typed anything, don't leave a
  // misleading prompt behind.
  useEffect(() => {
    if (!touched) setContext(activeSource === 'builtin:breast_cancer' ? DEFAULT_BUILTIN_CONTEXT : '')
  }, [activeSource, touched])

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    setClarification(null)
    try {
      const response = await api.createRun({
        context: context.trim(),
        file_path: activeSource,
        constraints: { max_tuning_trials: maxTrials, max_loop_backs: maxLoopBacks },
      })
      if (isClarificationResponse(response)) {
        setClarification(response)
        return
      }
      navigate(`/runs/${response.run_id}`)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-4">
      <label className="block text-[15px] font-semibold text-[var(--color-text-primary)]">
        Describe your modeling problem in plain English (context):
        <textarea
          value={context}
          onChange={(e) => {
            setTouched(true)
            setContext(e.target.value)
          }}
          rows={4}
          placeholder={
            upload
              ? 'e.g. Predict whether a customer churns (target column churned), aim for at least 0.85 f1.'
              : undefined
          }
          className={clsx(inputClass, 'mt-3 font-normal')}
        />
      </label>
      <p className="text-xs text-[var(--color-text-tertiary)]">
        Name the target column exactly as it appears in the peek above — the agent validates it against
        the real schema and asks for clarification rather than guess. Submitting starts the run
        immediately (there is no separate preview step).
      </p>

      {clarification && (
        <div className="rounded-lg border border-amber-500/30 bg-amber-500/5 p-4">
          <p className="text-sm font-medium text-amber-300">Needs clarification — not an error.</p>
          <p className="mt-1 text-sm text-[var(--color-text-secondary)]">{clarification.question_for_user}</p>
          <p className="mt-2 font-mono text-xs text-amber-400/80">
            missing: {clarification.missing_fields.join(', ')}
          </p>
        </div>
      )}
      {error && <ErrorBox message={error} />}

      <SubmitButton submitting={submitting} disabled={!context.trim()} label="Launch Pipeline" />
    </form>
  )
}

function StructuredForm() {
  const navigate = useNavigate()
  const { activeSource, peek, benchmark, upload, maxTrials, maxLoopBacks } = useStudio()

  const [taskType, setTaskType] = useState<TaskType>(benchmark.task_type ?? 'classification')
  const [targetColumn, setTargetColumn] = useState('')
  const [successMetric, setSuccessMetric] = useState<string>('f1')
  const [metricThreshold, setMetricThreshold] = useState(0.9)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const columns = peek?.columns.map((c) => c.name) ?? []
  const metricOptions: readonly string[] = taskType === 'classification' ? CLASSIFICATION_METRICS : REGRESSION_METRICS

  // Default the target to the benchmark's label column once the peek loads.
  useEffect(() => {
    if (columns.length === 0) return
    if (!columns.includes(targetColumn)) {
      setTargetColumn(upload ? columns[columns.length - 1] : columns.includes('diagnosis') ? 'diagnosis' : columns[columns.length - 1])
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [peek])

  const spec: ProblemSpec = {
    task_type: taskType,
    target_column: targetColumn.trim(),
    success_metric: successMetric,
    metric_threshold: metricThreshold,
    data_source: activeSource,
    constraints: {
      max_tuning_trials: maxTrials,
      max_loop_backs: maxLoopBacks,
      max_training_seconds: null,
      interpretability_required: false,
    },
    notes: null,
  }

  const submit = async (e: React.FormEvent) => {
    e.preventDefault()
    setSubmitting(true)
    setError(null)
    try {
      const response = await api.createRun({ problem_spec: spec })
      if (isClarificationResponse(response)) {
        setError(response.question_for_user)
        return
      }
      navigate(`/runs/${response.run_id}`)
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-5">
      <div className="grid gap-5 md:grid-cols-2">
        <Field label="task_type">
          <select
            value={taskType}
            onChange={(e) => {
              const next = e.target.value as TaskType
              setTaskType(next)
              setSuccessMetric(next === 'classification' ? 'f1' : 'rmse')
            }}
            className={inputClass}
          >
            <option value="classification">classification</option>
            <option value="regression">regression</option>
          </select>
        </Field>
        <Field label="target_column">
          {columns.length > 0 ? (
            <select value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className={clsx(inputClass, 'font-mono')}>
              {columns.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>
          ) : (
            <input value={targetColumn} onChange={(e) => setTargetColumn(e.target.value)} className={clsx(inputClass, 'font-mono')} />
          )}
        </Field>
        <Field label="success_metric">
          <select
            value={successMetric}
            onChange={(e) => setSuccessMetric(e.target.value as ClassificationMetric | RegressionMetric)}
            className={inputClass}
          >
            {metricOptions.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </Field>
        <Field label="metric_threshold">
          <input
            type="number"
            step="0.01"
            value={metricThreshold}
            onChange={(e) => setMetricThreshold(Number(e.target.value))}
            className={clsx(inputClass, 'font-mono')}
          />
        </Field>
      </div>

      <p className="font-mono text-xs text-[var(--color-text-tertiary)]">
        data_source={activeSource} · max_tuning_trials={maxTrials} · max_loop_backs={maxLoopBacks}{' '}
        (set in the sidebar)
      </p>

      {error && <ErrorBox message={error} />}

      <SubmitButton submitting={submitting} disabled={!targetColumn.trim()} label="Launch Pipeline" />
    </form>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="block">
      <span className="mb-1.5 block font-mono text-xs text-[var(--color-text-secondary)]">{label}</span>
      {children}
    </label>
  )
}
