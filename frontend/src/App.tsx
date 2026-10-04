import { Navigate, Route, Routes } from 'react-router-dom'
import { AppShell } from '@/components/layout/AppShell'
import { StudioProvider } from '@/context/StudioContext'
import { OverviewPage } from '@/pages/Overview'
import { NewRunPage } from '@/pages/NewRun'
import { LatestRunRedirect } from '@/pages/LatestRunRedirect'
import { LiveRunPage } from '@/pages/LiveRun'
import { RunHistoryPage } from '@/pages/RunHistory'
import { PerformancePage } from '@/pages/Performance'
import { LlmUsagePage } from '@/pages/LlmUsage'
import { ArchitecturePage } from '@/pages/Architecture'
import { SettingsPage } from '@/pages/Settings'

export default function App() {
  return (
    <StudioProvider>
      <AppShell>
        <Routes>
          <Route path="/" element={<NewRunPage />} />
          <Route path="/new-run" element={<Navigate to="/" replace />} />
          <Route path="/live" element={<LatestRunRedirect />} />
          <Route path="/runs" element={<RunHistoryPage />} />
          <Route path="/runs/:runId" element={<LiveRunPage />} />
          <Route path="/architecture" element={<ArchitecturePage />} />
          {/* Still routed, no longer in the sidebar (the reference design
              has four entries). */}
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/performance" element={<PerformancePage />} />
          <Route path="/llm-usage" element={<LlmUsagePage />} />
          <Route path="/settings" element={<SettingsPage />} />
        </Routes>
      </AppShell>
    </StudioProvider>
  )
}
