import { useCallback, useState } from 'react'
import { UploadCloud, FileCheck2, Loader2, X } from 'lucide-react'
import { api, ApiError, ApiUnreachableError } from '@/api/client'
import type { UploadResponse } from '@/api/types'

/**
 * Real upload via POST /uploads. There is deliberately NO dataset schema
 * preview shown after upload (rows/columns/dtypes) — DatasetPeek, which
 * would contain that, is computed entirely server-side inside
 * RequirementAgent and is never returned by any endpoint (verified against
 * agents/requirement_agent.py and tools/data_peek_tools.py). What's shown
 * here — filename and byte size — is the complete, real UploadResponse.
 */
export function FileUpload({
  onUploaded,
  onCleared,
}: {
  onUploaded: (result: UploadResponse) => void
  onCleared: () => void
}) {
  const [state, setState] = useState<
    | { status: 'idle' }
    | { status: 'uploading'; filename: string }
    | { status: 'done'; result: UploadResponse }
    | { status: 'error'; message: string }
  >({ status: 'idle' })

  const upload = useCallback(
    async (file: File) => {
      setState({ status: 'uploading', filename: file.name })
      try {
        const result = await api.uploadDataset(file)
        setState({ status: 'done', result })
        onUploaded(result)
      } catch (e) {
        const message =
          e instanceof ApiUnreachableError || e instanceof ApiError
            ? e.message
            : 'Upload failed.'
        setState({ status: 'error', message })
      }
    },
    [onUploaded],
  )

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) upload(file)
  }

  if (state.status === 'done') {
    return (
      <div className="flex items-center justify-between rounded-lg border border-emerald-500/30 bg-emerald-500/5 px-4 py-3">
        <div className="flex items-center gap-2 text-sm text-emerald-300">
          <FileCheck2 className="h-4 w-4" />
          <span className="font-medium">{state.result.filename}</span>
          <span className="text-[var(--color-text-tertiary)]">
            ({(state.result.size_bytes / 1024).toFixed(1)} KB)
          </span>
        </div>
        <button
          onClick={() => {
            setState({ status: 'idle' })
            onCleared()
          }}
          className="text-[var(--color-text-tertiary)] hover:text-[var(--color-text-primary)]"
        >
          <X className="h-4 w-4" />
        </button>
      </div>
    )
  }

  return (
    <div>
      <label
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
        className="flex cursor-pointer flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-[var(--color-border-default)] bg-[var(--color-bg-surface-2)] px-6 py-10 text-center transition hover:border-[var(--color-accent-primary)]/50"
      >
        {state.status === 'uploading' ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin text-[var(--color-accent-secondary)]" />
            <p className="text-sm text-[var(--color-text-secondary)]">
              Uploading {state.filename}…
            </p>
          </>
        ) : (
          <>
            <UploadCloud className="h-5 w-5 text-[var(--color-text-tertiary)]" />
            <p className="text-sm text-[var(--color-text-secondary)]">
              Drag a file here, or click to browse
            </p>
            <p className="text-xs text-[var(--color-text-tertiary)]">CSV or Parquet, up to 50MB</p>
          </>
        )}
        <input
          type="file"
          accept=".csv,.parquet"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0]
            if (file) upload(file)
          }}
        />
      </label>
      {state.status === 'error' && (
        <p className="mt-2 text-sm text-rose-400">{state.message}</p>
      )}
    </div>
  )
}
