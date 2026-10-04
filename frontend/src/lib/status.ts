/**
 * Single source of truth for status -> color/icon/label mapping.
 * Every screen imports from here rather than deciding colors locally —
 * that's what keeps the semantic meaning (green=success, blue=active,
 * amber=warning/loop-back, red=failed/blocked, purple=LLM, gray=pending)
 * consistent across the whole app instead of drifting per-page.
 */
import {
  CheckCircle2,
  CircleDashed,
  Clock,
  Loader2,
  OctagonAlert,
  RotateCcw,
  XCircle,
} from 'lucide-react'
import type { RunStatus } from '@/api/types'

export type Tone = 'success' | 'active' | 'warning' | 'danger' | 'llm' | 'neutral'

export const toneClasses: Record<Tone, { text: string; bg: string; border: string; dot: string }> = {
  success: {
    text: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/30',
    dot: 'bg-emerald-400',
  },
  active: {
    text: 'text-sky-400',
    bg: 'bg-sky-500/10',
    border: 'border-sky-500/30',
    dot: 'bg-sky-400',
  },
  warning: {
    text: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/30',
    dot: 'bg-amber-400',
  },
  danger: {
    text: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/30',
    dot: 'bg-rose-400',
  },
  // Reserved specifically for "the LLM did something here" — never used
  // decoratively elsewhere in the app.
  llm: {
    text: 'text-violet-400',
    bg: 'bg-violet-500/10',
    border: 'border-violet-500/30',
    dot: 'bg-violet-400',
  },
  neutral: {
    text: 'text-slate-400',
    bg: 'bg-slate-500/10',
    border: 'border-slate-500/30',
    dot: 'bg-slate-400',
  },
}

interface StatusMeta {
  label: string
  tone: Tone
  icon: typeof CheckCircle2
  spin?: boolean
}

/** Maps the REAL RunStatus values (schemas/pipeline_run.py) to a tone/icon.
 *  "created" through "evaluated" are all in-progress stage markers — none
 *  of them are terminal. Only "done" and "escalated" are terminal
 *  (see TERMINAL_STATUSES in api/types.ts). */
export function runStatusMeta(status: RunStatus): StatusMeta {
  switch (status) {
    case 'done':
      return { label: 'Done', tone: 'success', icon: CheckCircle2 }
    case 'escalated':
      return { label: 'Escalated', tone: 'danger', icon: OctagonAlert }
    case 'created':
      return { label: 'Created', tone: 'neutral', icon: Clock }
    case 'data_ready':
      return { label: 'Data ready', tone: 'active', icon: Loader2, spin: true }
    case 'features_ready':
      return { label: 'Features ready', tone: 'active', icon: Loader2, spin: true }
    case 'tuned':
      return { label: 'Tuned', tone: 'active', icon: Loader2, spin: true }
    case 'evaluated':
      return { label: 'Evaluated', tone: 'active', icon: Loader2, spin: true }
  }
}

export type StageState = 'pending' | 'done' | 'skipped'

export function stageStateMeta(state: StageState): StatusMeta {
  switch (state) {
    case 'done':
      return { label: 'Done', tone: 'success', icon: CheckCircle2 }
    case 'skipped':
      return { label: 'Skipped', tone: 'neutral', icon: RotateCcw }
    case 'pending':
      return { label: 'Pending', tone: 'neutral', icon: CircleDashed }
  }
}

export function passFailMeta(passFail: 'pass' | 'fail'): StatusMeta {
  return passFail === 'pass'
    ? { label: 'Pass', tone: 'success', icon: CheckCircle2 }
    : { label: 'Fail', tone: 'danger', icon: XCircle }
}
