import { SectionHeader } from '@/components/ui/SectionHeader'
import { Panel } from '@/components/ui/Panel'
import { API_BASE_URL } from '@/api/client'
import { useHealth } from '@/hooks/useHealth'

/**
 * §31: "Only expose settings the app actually supports. Do not build fake
 * provider/API-key configuration UI with no backing functionality." The
 * frontend holds zero LLM credentials — every OpenAI call happens
 * server-side using the backend's own OPENAI_API_KEY — so there is no
 * provider/key settings section here at all, deliberately.
 */
export function SettingsPage() {
  const connected = useHealth()

  return (
    <div className="max-w-xl">
      <SectionHeader title="Settings" />

      <Panel className="space-y-4">
        <div>
          <p className="mb-1 text-sm font-medium text-[var(--color-text-primary)]">API base URL</p>
          <p className="rounded-md border border-[var(--color-border-subtle)] bg-[var(--color-bg-surface-2)] px-3 py-2 font-mono text-sm text-[var(--color-text-secondary)]">
            {API_BASE_URL}
          </p>
          <p className="mt-2 text-xs text-[var(--color-text-tertiary)]">
            Set via VITE_API_BASE_URL at build time (see .env.example) — not editable at runtime
            in this build, since Vite inlines env vars at compile time.
          </p>
        </div>

        <div>
          <p className="mb-1 text-sm font-medium text-[var(--color-text-primary)]">Connection</p>
          <p className={connected ? 'text-sm text-emerald-400' : 'text-sm text-rose-400'}>
            {connected === null ? 'Checking…' : connected ? '● Connected' : '● Disconnected'}
          </p>
        </div>
      </Panel>

      <p className="mt-4 text-xs text-[var(--color-text-tertiary)]">
        No LLM provider or API key configuration lives here — every model call happens
        server-side in the backend's agents, using the backend's own OPENAI_API_KEY. This
        frontend never holds, sends, or needs any LLM credential.
      </p>
    </div>
  )
}
