import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { api } from '@/api/client'
import type { DatasetInfo, DatasetPeek, UploadResponse } from '@/api/types'

/** The fallback entry used until GET /datasets answers (or if the API is
 *  down) — the same builtin the backend always lists first. */
const FALLBACK_BENCHMARK: DatasetInfo = {
  source: 'builtin:breast_cancer',
  label: 'Breast Cancer Diagnostic (Wisconsin)',
  description:
    'Real-world clinical benchmark with 30 continuous numeric features computed from digitized FNA images of breast masses.',
  task_type: 'classification',
}

export const LIMITS = {
  loopBacks: { min: 1, max: 5, default: 3 },
  trials: { min: 5, max: 50, default: 15 },
  cvFolds: { min: 2, max: 10, default: 5 },
} as const

interface StudioState {
  benchmarks: DatasetInfo[]
  benchmark: DatasetInfo
  setBenchmarkSource: (source: string) => void

  /** A user-uploaded file overrides the benchmark as the active dataset. */
  upload: UploadResponse | null
  setUpload: (u: UploadResponse | null) => void

  /** What the run will actually use, and a label for it. */
  activeSource: string
  activeLabel: string
  peek: DatasetPeek | null
  peekError: string | null
  peekLoading: boolean

  maxLoopBacks: number
  setMaxLoopBacks: (n: number) => void
  maxTrials: number
  setMaxTrials: (n: number) => void
  /** UI-only for now: the pipeline's CV fold count is a constant in
   *  pipeline/tuning_stage.py, not a ProblemSpec field. */
  cvFolds: number
  setCvFolds: (n: number) => void
}

const Ctx = createContext<StudioState | null>(null)

export function StudioProvider({ children }: { children: ReactNode }) {
  const [benchmarks, setBenchmarks] = useState<DatasetInfo[]>([FALLBACK_BENCHMARK])
  const [benchmarkSource, setBenchmarkSource] = useState(FALLBACK_BENCHMARK.source)
  const [upload, setUpload] = useState<UploadResponse | null>(null)

  const [maxLoopBacks, setMaxLoopBacks] = useState<number>(LIMITS.loopBacks.default)
  const [maxTrials, setMaxTrials] = useState<number>(LIMITS.trials.default)
  const [cvFolds, setCvFolds] = useState<number>(LIMITS.cvFolds.default)

  const [peek, setPeek] = useState<DatasetPeek | null>(null)
  const [peekError, setPeekError] = useState<string | null>(null)
  const [peekLoading, setPeekLoading] = useState(true)

  useEffect(() => {
    api
      .listDatasets()
      .then((items) => items.length > 0 && setBenchmarks(items))
      .catch(() => {
        // Keep the fallback; the peek request below surfaces the real error.
      })
  }, [])

  const benchmark = benchmarks.find((b) => b.source === benchmarkSource) ?? benchmarks[0]
  const activeSource = upload?.file_path ?? benchmark.source
  const activeLabel = upload?.filename ?? benchmark.label

  useEffect(() => {
    let cancelled = false
    setPeekLoading(true)
    setPeekError(null)
    api
      .peekDataset(activeSource)
      .then((p) => !cancelled && setPeek(p))
      .catch((e: Error) => {
        if (cancelled) return
        setPeek(null)
        setPeekError(e.message)
      })
      .finally(() => !cancelled && setPeekLoading(false))
    return () => {
      cancelled = true
    }
  }, [activeSource])

  const value = useMemo<StudioState>(
    () => ({
      benchmarks,
      benchmark,
      setBenchmarkSource: (source) => {
        setUpload(null)
        setBenchmarkSource(source)
      },
      upload,
      setUpload,
      activeSource,
      activeLabel,
      peek,
      peekError,
      peekLoading,
      maxLoopBacks,
      setMaxLoopBacks,
      maxTrials,
      setMaxTrials,
      cvFolds,
      setCvFolds,
    }),
    [benchmarks, benchmark, upload, activeSource, activeLabel, peek, peekError, peekLoading, maxLoopBacks, maxTrials, cvFolds],
  )

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useStudio(): StudioState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useStudio must be used inside <StudioProvider>')
  return ctx
}
