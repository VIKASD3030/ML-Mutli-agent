import { Zap } from 'lucide-react'
import clsx from 'clsx'
import { useHealth } from '@/hooks/useHealth'
import { STREAMLIT_URL } from '@/lib/links'

export function TopBar() {
  const connected = useHealth()

  return (
    <header className="flex h-[73px] shrink-0 items-center justify-between border-b border-[var(--color-border-subtle)] bg-[var(--color-bg-surface)] px-8">
      <div className="flex items-center gap-4">
        <Zap className="h-5 w-5 fill-orange-400 text-orange-400" />
        <h1 className="text-lg font-bold text-[var(--color-text-primary)]">Multi-Agent ML Pipeline</h1>
        <span
          className={clsx(
            'rounded-full border px-3 py-1 font-mono text-xs',
            connected === false
              ? 'border-rose-500/30 bg-rose-500/10 text-rose-300'
              : 'border-[var(--color-border-default)] bg-[var(--color-bg-surface-3)] text-[var(--color-text-secondary)]',
          )}
        >
          API · {connected === null ? 'Checking…' : connected ? 'Live' : 'Offline'}
        </span>
      </div>

      <div className="flex items-center gap-4">
        <span className="rounded-lg border border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-4 py-2 font-mono text-xs text-[var(--color-text-secondary)]">
          UI Mode: <span className="font-semibold text-[var(--color-accent-secondary)]">Modern ML Studio</span>
        </span>
        <a
          href={STREAMLIT_URL}
          target="_blank"
          rel="noreferrer"
          className="rounded-lg bg-[var(--color-accent-primary)] px-4 py-2 text-sm font-semibold text-white transition hover:bg-[var(--color-accent-primary-hover)]"
        >
          Switch to Streamlit
        </a>
      </div>
    </header>
  )
}
