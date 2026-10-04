import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { TrialRecord } from '@/api/types'
import { EmptyState } from '@/components/ui/States'

export function TuningChart({ history }: { history: TrialRecord[] }) {
  if (history.length === 0) {
    return <EmptyState title="No trial history yet" />
  }

  const best = history.reduce((a, b) => (b.score > a.score ? b : a))
  const data = history.map((t) => ({ trial: t.trial_number, score: t.score }))

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" />
        <XAxis
          dataKey="trial"
          stroke="var(--color-text-tertiary)"
          fontSize={11}
          label={{ value: 'Trial', position: 'insideBottom', offset: -2, fill: 'var(--color-text-tertiary)', fontSize: 11 }}
        />
        <YAxis stroke="var(--color-text-tertiary)" fontSize={11} width={56} />
        <Tooltip
          contentStyle={{
            backgroundColor: 'var(--color-bg-surface-2)',
            border: '1px solid var(--color-border-default)',
            borderRadius: 8,
            fontSize: 12,
          }}
          labelFormatter={(v) => `Trial ${v}`}
          formatter={(value) => [Number(value).toFixed(4), 'CV score']}
        />
        <Line
          type="monotone"
          dataKey="score"
          stroke="var(--color-accent-secondary)"
          strokeWidth={2}
          dot={{ r: 2.5, fill: 'var(--color-accent-secondary)' }}
        />
        <ReferenceDot
          x={best.trial_number}
          y={best.score}
          r={5}
          fill="var(--color-accent-primary)"
          stroke="none"
        />
      </LineChart>
    </ResponsiveContainer>
  )
}
