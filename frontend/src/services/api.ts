/**
 * The only place this app talks to the network: plain HTTP to the FastAPI
 * backend. No LLM client lives in the browser; every model call happens
 * server-side with the backend's own OPENAI_API_KEY.
 */
import type {
  WireCreateRunRequest,
  WireCreateRunResponse,
  WireDatasetInfo,
  WireDatasetPeek,
  WireRunDetail,
  WireRunSummary,
  WireUpload,
} from './apiTypes';

export const API_BASE_URL: string =
  import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`${API_BASE_URL}${path}`, init);
  } catch {
    // fetch() throws TypeError for network failures and CORS rejections alike.
    throw new ApiError(`Could not reach the API at ${API_BASE_URL} — is the backend running?`, 0);
  }

  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      if (typeof body.detail === 'string') detail = body.detail;
      else if (Array.isArray(body.detail)) {
        detail = body.detail.map((d: { msg?: string }) => d.msg ?? JSON.stringify(d)).join('; ');
      }
    } catch {
      // non-JSON error body: keep statusText
    }
    throw new ApiError(detail, response.status);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<{ status: string }>('/health'),

  listDatasets: () => request<WireDatasetInfo[]>('/datasets'),

  peekDataset: (source: string) =>
    request<WireDatasetPeek>(`/datasets/peek?source=${encodeURIComponent(source)}`),

  uploadDataset: (file: File) => {
    const form = new FormData();
    // Field name must be "file" (upload_dataset(file: UploadFile) in api/routes/uploads.py).
    form.append('file', file);
    return request<WireUpload>('/uploads', { method: 'POST', body: form });
  },

  createRun: (body: WireCreateRunRequest) =>
    request<WireCreateRunResponse>('/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    }),

  getRun: (runId: string) => request<WireRunDetail>(`/runs/${runId}`),

  listRuns: () => request<WireRunSummary[]>('/runs'),
};

export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : 'Something went wrong.';
}
