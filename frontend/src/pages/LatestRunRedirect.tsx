import { Navigate } from 'react-router-dom'
import { Link } from 'react-router-dom'
import { Zap } from 'lucide-react'
import { useRuns } from '@/hooks/useRuns'
import { EmptyState, LoadingState } from '@/components/ui/States'

/** "Live Run" in the sidebar: jump to the most recently updated run. */
export function LatestRunRedirect() {
  const { runs, loading, error } = useRuns()

  if (loading) return <LoadingState label="Finding your latest run…" />
  if (error) return <EmptyState title="Couldn't load runs" description={error.message} />
  if (!runs || runs.length === 0) {
    return (
      <EmptyState
        icon={Zap}
        title="No runs yet"
        description="Start a pipeline run and it will show up here live."
        action={
          <Link to="/" className="text-sm font-medium text-[var(--color-accent-secondary)]">
            Initialize a run →
          </Link>
        }
      />
    )
  }
  const latest = [...runs].sort((a, b) => b.updated_at.localeCompare(a.updated_at))[0]
  return <Navigate to={`/runs/${latest.run_id}`} replace />
}
