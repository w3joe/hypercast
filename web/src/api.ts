import type {
  ArchitectureRecord,
  ArchitectureSpec,
  Catalog,
  ComputeCapabilities,
  EvaluationSpec,
  ExecutionSpec,
  Job,
  ValidationResult,
  InternalGraph,
  GraphSpec, GraphRecord, GraphViewState, GraphValidation, GraphDescription,
} from './types'
import { offlineApi } from './offlineApi'

export const isOfflineDemo = import.meta.env.VITE_OFFLINE_DEMO === 'true'

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...options,
    headers: { 'Content-Type': 'application/json', ...options?.headers },
  })
  if (!response.ok) {
    const body = await response.json().catch(() => ({ detail: response.statusText }))
    throw new Error(body.detail || 'Request failed')
  }
  return response.json() as Promise<T>
}

const remoteApi = {
  demoSession: () => request<import('./types').DemoSession>('/api/v1/demo/session'),
  demoRequestCode: (email: string) => request<{ challenge_id: string }>('/api/v1/demo/request-code', { method: 'POST', body: JSON.stringify({ email }) }),
  demoVerify: (challenge_id: string, code: string) => request<{ authenticated: boolean }>('/api/v1/demo/verify', { method: 'POST', body: JSON.stringify({ challenge_id, code }) }),
  demoLogout: () => request('/api/v1/demo/logout', { method: 'POST', body: '{}' }),
  forecasts: (id: string, window: number, horizon: number, seed: number, fold: number, lead: number, trial = '') =>
    request<import('./ForecastReplay').ForecastResponse>(`/api/v1/jobs/${encodeURIComponent(id)}/forecasts?${new URLSearchParams({ window: String(window), horizon: String(horizon), seed: String(seed), fold: String(fold), lead: String(lead), trial })}`),
  comparisonArchives: () => request<import('./types').ArchivedComparisonJob[]>('/api/v1/comparison-archives'),
  archivedForecasts: (id: string, window: number, horizon: number, seed: number, fold: number, lead: number) =>
    request<import('./ForecastReplay').ForecastResponse>(`/api/v1/comparison-archives/${encodeURIComponent(id)}/forecasts?${new URLSearchParams({ window: String(window), horizon: String(horizon), seed: String(seed), fold: String(fold), lead: String(lead) })}`),
  trainedWeights: (id: string) => request<Record<string, import('./WeightInspector').WeightSnapshot>>(`/api/v1/jobs/${encodeURIComponent(id)}/weights`),
  previewWeights: (architecture: GraphSpec, window: number, horizon: number) => request<import('./WeightInspector').WeightSnapshot>('/api/v1/architectures/weights', { method: 'POST', body: JSON.stringify({ architecture, window, horizon }) }),
  editGraph: (architecture: GraphSpec, edit: { action: string; id: string; kind: string; params: Record<string, unknown> }, cells: { window: number; horizon: number }[]) =>
    request<{ spec: GraphSpec; warnings: string[] }>('/api/v1/architectures/edit', { method: 'POST', body: JSON.stringify({ architecture, edit, cells }) }),
  describeGraph: (architecture: GraphSpec, window: number, horizon: number) =>
    request<GraphDescription>('/api/v1/architectures/describe', { method: 'POST', body: JSON.stringify({ architecture, window, horizon }) }),
  convert: (architecture: ArchitectureSpec | GraphSpec, window: number, horizon: number) =>
    request<GraphSpec>('/api/v1/architectures/convert', { method: 'POST', body: JSON.stringify({ architecture, window, horizon }) }),
  validateGraph: (architecture: GraphSpec, window: number, horizon: number) =>
    request<GraphValidation>('/api/v1/architectures/validate', { method: 'POST', body: JSON.stringify({ architecture, window, horizon }) }),
  graphRecords: () => request<GraphRecord[]>('/api/v1/architectures'),
  saveGraph: (architecture: GraphSpec, view: GraphViewState, id?: string) =>
    request<GraphRecord>('/api/v1/architectures', { method: 'POST', body: JSON.stringify({ ...architecture, view, id }) }),
  internalGraph: (architecture: ArchitectureSpec, layerId: string, window: number, horizon: number) =>
    request<InternalGraph>('/api/v1/architectures/internal-graph', {
      method: 'POST', body: JSON.stringify({ architecture, layer_id: layerId, window, horizon }),
    }),
  catalog: () => request<Catalog>('/api/v1/catalog'),
  compute: () => request<ComputeCapabilities>('/api/v1/compute'),
  validate: (architecture: ArchitectureSpec, window: number, horizon: number) =>
    request<ValidationResult>('/api/v1/architectures/validate', {
      method: 'POST',
      body: JSON.stringify({ architecture, window, horizon }),
    }),
  architectures: () => request<ArchitectureRecord[]>('/api/v1/architectures'),
  saveArchitecture: (architecture: ArchitectureSpec & { id?: string }) =>
    request<ArchitectureRecord>('/api/v1/architectures', {
      method: 'POST',
      body: JSON.stringify(architecture),
    }),
  jobs: () => request<Job[]>('/api/v1/jobs'),
  legacyRuns: () => request<Record<string, string | number | null>[]>('/api/runs'),
  submit: (architecture: ArchitectureSpec | GraphSpec, evaluation: EvaluationSpec, execution: ExecutionSpec) =>
    request<Job>('/api/v1/jobs', {
      method: 'POST',
      body: JSON.stringify({ architecture, evaluation, execution }),
    }),
  cancel: (jobId: string) =>
    request<Job>(`/api/v1/jobs/${jobId}/cancel`, { method: 'POST' }),
  finalTest: (jobId: string) =>
    request<Job>(`/api/v1/jobs/${jobId}/final-test`, { method: 'POST' }),
  log: async (jobId: string) => {
    const response = await fetch(`/api/v1/jobs/${jobId}/log`)
    if (!response.ok) throw new Error('Unable to load training log')
    return response.text()
  },
}

export const api = (isOfflineDemo ? offlineApi : remoteApi) as typeof remoteApi
