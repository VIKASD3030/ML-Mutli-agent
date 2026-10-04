/**
 * The ONLY place this app talks to a network. Every call here goes straight
 * to the real FastAPI backend over plain HTTP — nothing here constructs or
 * holds an LLM client of any kind, and nothing here ever will. All LLM
 * calls happen server-side, inside agents/*.py, using the backend's own
 * OPENAI_API_KEY. The frontend does not know, and does not need to know,
 * which LLM provider (or how many) the backend uses.
 */
import type {
  ClarificationResponse,
  CreateRunResponse,
  DatasetInfo,
  DatasetPeek,
  RunCreateRequest,
  RunDetail,
  RunSummary,
  UploadResponse,
} from './types'

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export class ApiError extends Error {
  status: number
  constructor(message: string, status: number) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

/** Thrown specifically when the backend cannot be reached at all (server
 * down, wrong URL, CORS misconfiguration) — distinct from ApiError so the
 * UI can show "is the API running?" instead of a generic failure message. */
export class ApiUnreachableError extends Error {
  baseUrl: string
  constructor(baseUrl: string) {
    super(`Could not connect to the API at ${baseUrl} — is it running?`)
    this.name = 'ApiUnreachableError'
    this.baseUrl = baseUrl
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init)
  } catch {
    // fetch() throws TypeError for network failures, DNS errors, CORS
    // rejections, and connection refused — all of which mean the same
    // thing to a user: the backend isn't reachable at this URL.
    throw new ApiUnreachableError(API_BASE_URL)
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = await response.json()
      // FastAPI's HTTPException shape is {"detail": "..."} or
      // {"detail": [{"msg": "...", ...}, ...]} for pydantic validation
      // errors — handle both rather than showing "[object Object]".
      if (typeof body.detail === 'string') {
        detail = body.detail
      } else if (Array.isArray(body.detail)) {
        detail = body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ')
      }
    } catch {
      // Response body wasn't JSON — fall back to statusText.
    }
    throw new ApiError(detail, response.status)
  }

  return response.json() as Promise<T>
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  uploadDataset: async (file: File): Promise<UploadResponse> => {
    const formData = new FormData()
    // Field name MUST be "file" — it is the parameter name on
    // upload_dataset(file: UploadFile) in api/routes/uploads.py, and
    // FastAPI binds the multipart field to the backend by that name.
    formData.append('file', file)
    return request<UploadResponse>('/uploads', { method: 'POST', body: formData })
  },

  createRun: (body: RunCreateRequest): Promise<CreateRunResponse> =>
    request<CreateRunResponse>('/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  listDatasets: (): Promise<DatasetInfo[]> => request<DatasetInfo[]>('/datasets'),

  peekDataset: (source: string): Promise<DatasetPeek> =>
    request<DatasetPeek>(`/datasets/peek?source=${encodeURIComponent(source)}`),

  getRun: (runId: string): Promise<RunDetail> => request<RunDetail>(`/runs/${runId}`),

  listRuns: (): Promise<RunSummary[]> => request<RunSummary[]>('/runs'),
}

export function isClarificationResponse(
  r: CreateRunResponse,
): r is ClarificationResponse {
  return r.status === 'needs_clarification'
}
