import { Link, useLocation } from 'react-router-dom'
import clsx from 'clsx'
import {
  BookOpen,
  ChevronDown,
  Clock,
  Database,
  PlayCircle,
  Settings2,
  Sparkles,
  Palette,
} from 'lucide-react'
import { Logo } from './Logo'
import { LIMITS, useStudio } from '@/context/StudioContext'
import { STREAMLIT_URL } from '@/lib/links'

interface NavItem {
  to: string
  label: string
  icon: typeof Sparkles
  emoji: string
  isActive: (pathname: string) => boolean
}

const NAV: NavItem[] = [
  { to: '/', label: 'New Run', icon: Sparkles, emoji: '➕', isActive: (p) => p === '/' || p === '/new-run' },
  { to: '/live', label: 'Live Run', icon: PlayCircle, emoji: '⚡', isActive: (p) => p === '/live' || p.startsWith('/runs/') },
  { to: '/runs', label: 'Run History', icon: Clock, emoji: '📜', isActive: (p) => p === '/runs' },
  { to: '/architecture', label: 'Architecture & Spec', icon: BookOpen, emoji: '🏛️', isActive: (p) => p === '/architecture' },
]

export function Sidebar() {
  const { pathname } = useLocation()
  const studio = useStudio()
  const { benchmark, benchmarks, peek, upload } = studio

  const datasetCaption = upload
    ? `${upload.filename} · uploaded`
    : [
        peek ? `${peek.row_count} rows · ${peek.column_count} cols` : null,
        benchmark.task_type,
      ]
        .filter(Boolean)
        .join(' · ')

  return (
    <aside className="flex h-screen w-[22.25rem] shrink-0 flex-col overflow-y-auto border-r border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)] px-4 pt-5 pb-6">
      <div className="flex items-center gap-3 px-1">
        <Logo />
        <div className="leading-tight">
          <p className="text-[17px] font-bold text-[var(--color-text-primary)]">Multi-Agent ML</p>
          <p className="font-mono text-[11px] text-[var(--color-text-tertiary)]">
            AutoML &amp; LLM Supervisors
          </p>
        </div>
      </div>

      {/* Theme switch: AI Studio is this app; Streamlit is the other UI. */}
      <div className="mt-5 flex items-center justify-between rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] p-1.5 pl-3 text-sm">
        <span className="flex items-center gap-1.5 font-medium text-[var(--color-text-primary)]">
          <Palette className="h-3.5 w-3.5 text-rose-400" />
          UI Theme
        </span>
        <div className="flex items-center gap-1 text-[13px]">
          <a
            href={STREAMLIT_URL}
            target="_blank"
            rel="noreferrer"
            className="rounded-md px-3 py-1 text-[var(--color-text-secondary)] transition hover:text-[var(--color-text-primary)]"
          >
            Streamlit
          </a>
          <span className="rounded-md bg-[var(--color-accent-primary)] px-3 py-1 font-semibold text-white">
            AI Studio
          </span>
        </div>
      </div>

      <hr className="my-5 border-[var(--color-border-subtle)]" />

      <SectionLabel>Navigation</SectionLabel>
      <nav className="mt-2.5 space-y-1.5">
        {NAV.map((item) => {
          const active = item.isActive(pathname)
          return (
            <Link
              key={item.to}
              to={item.to}
              className={clsx(
                'flex items-center gap-3 rounded-lg px-3.5 py-2.5 text-[15px] font-medium transition',
                active
                  ? 'bg-[var(--color-accent-primary)] text-white shadow-[0_0_0_1px_rgba(255,255,255,0.04)]'
                  : 'text-[var(--color-text-primary)] hover:bg-[var(--color-bg-surface-2)]',
              )}
            >
              <item.icon className={clsx('h-[18px] w-[18px]', active ? 'text-white' : 'text-[var(--color-text-secondary)]')} />
              <span>
                <span className="mr-1.5 text-[13px]">{item.emoji}</span>
                {item.label}
              </span>
            </Link>
          )
        })}
      </nav>

      <div className="mt-8">
        <SectionLabel icon={Database}>Active Benchmark</SectionLabel>
        <div className="relative mt-2.5">
          <select
            value={benchmark.source}
            onChange={(e) => studio.setBenchmarkSource(e.target.value)}
            className="w-full appearance-none rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-3.5 py-3 pr-9 text-sm text-[var(--color-text-primary)] outline-none focus:border-[var(--color-accent-primary)]"
          >
            {benchmarks.map((b) => (
              <option key={b.source} value={b.source}>
                {b.label}
              </option>
            ))}
          </select>
          <ChevronDown className="pointer-events-none absolute top-1/2 right-3 h-4 w-4 -translate-y-1/2 text-[var(--color-text-secondary)]" />
        </div>
        {datasetCaption && (
          <p className="mt-2 text-[13px] text-[var(--color-text-secondary)]">{datasetCaption}</p>
        )}
      </div>

      <div className="mt-8">
        <SectionLabel icon={Settings2}>Pipeline Constraints</SectionLabel>
        <div className="mt-4 space-y-5">
          <Slider
            label="Max Loop-Backs:"
            value={studio.maxLoopBacks}
            {...LIMITS.loopBacks}
            onChange={studio.setMaxLoopBacks}
          />
          <Slider
            label="Optuna CV Trials:"
            value={studio.maxTrials}
            {...LIMITS.trials}
            onChange={studio.setMaxTrials}
          />
          <Slider
            label="CV Folds:"
            value={studio.cvFolds}
            {...LIMITS.cvFolds}
            onChange={studio.setCvFolds}
            disabled
            hint="Preview only — the pipeline's fold count (cv=3 in tuning_stage.py) isn't a run setting yet."
          />
        </div>
      </div>
    </aside>
  )
}

function SectionLabel({
  children,
  icon: Icon,
}: {
  children: string
  icon?: typeof Database
}) {
  return (
    <p className="flex items-center gap-2 font-mono text-xs font-semibold tracking-[0.12em] text-[var(--color-text-secondary)] uppercase">
      {Icon && <Icon className="h-4 w-4" />}
      {children}
    </p>
  )
}

function Slider({
  label,
  value,
  min,
  max,
  onChange,
  disabled,
  hint,
}: {
  label: string
  value: number
  min: number
  max: number
  onChange: (n: number) => void
  disabled?: boolean
  hint?: string
}) {
  const pct = ((value - min) / (max - min)) * 100
  return (
    <div title={hint}>
      <div className="mb-1.5 flex items-baseline justify-between text-[15px] text-[var(--color-text-primary)]">
        <span>{label}</span>
        <span className="font-mono text-sm font-semibold">{value}</span>
      </div>
      <input
        type="range"
        className="studio-range"
        min={min}
        max={max}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ['--pct' as string]: `${pct}%` }}
        aria-label={label}
      />
      {hint && <p className="mt-1.5 text-[11px] leading-snug text-[var(--color-text-tertiary)]">{hint}</p>}
    </div>
  )
}
