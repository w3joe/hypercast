import type { Job, ComparisonJob } from './types'

export const comparisonColors = ['#011476', '#009dc0', '#6b3eda', '#bb751b', '#4078cb', '#bb527a']
export type ComparisonMetric = 'mae_ratio' | 'mse_ratio' | 'mae' | 'mse'
export const comparisonMetrics: Record<ComparisonMetric, { label: string; ratio: boolean }> = {
  mae_ratio: { label: 'MAE / persistence', ratio: true },
  mse_ratio: { label: 'MSE / persistence', ratio: true },
  mae: { label: 'MAE (target units)', ratio: false },
  mse: { label: 'MSE (target units²)', ratio: false },
}
export const comparisonRunLabel = (job: ComparisonJob) => 'archive' in job ? job.status.architecture_name : `${job.status.architecture_name} · ${job.id.slice(-8)}`
export const comparisonRunTag = (job: ComparisonJob) => 'archive' in job ? 'Archived study' : job.id.slice(-8)
/** Stable backbone colours across selections, metrics, and comparison charts. */
export function comparisonColor(job: ComparisonJob) {
  const key = 'archive' in job ? job.archive.backbone : job.status.architecture_name
  let hash = 0
  for (const char of key.toLowerCase()) hash = (Math.imul(hash, 31) + char.charCodeAt(0)) | 0
  return `hsl(${(hash >>> 0) % 360} 43% 42%)`
}
type Row = Job['runs'][number]
export const numeric = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v)
const canonical = (v: unknown): unknown => Array.isArray(v) ? v.map(canonical) : v && typeof v === 'object'
  ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.localeCompare(b)).map(([k, x]) => [k, canonical(x)])) : v

// Explicit dataset/split metadata defines a cohort. Filenames alone cannot verify provenance.
export function comparisonProtocol(job: ComparisonJob) {
  const e = job.request.evaluation as typeof job.request.evaluation & Record<string, unknown>
  if (!e.protocol || !e.data_path || !e.target_column) return `unknown-provenance:${job.id}`
  return JSON.stringify(canonical([e.protocol, e.data_path, e.target_column, job.status.phase, e.preset,
    e.folds ?? null, e.train_fraction ?? null, e.validation_fraction ?? null, e.nested_stopping ?? null,
    e.data_sha256 ?? null, e.split_sha256 ?? null, e.target_scale ?? null, e.split_metadata ?? null]))
}
export function comparisonCells(jobs: ComparisonJob[]) {
  const cells = new Map<string, { key: string; protocol: string; window: number; horizon: number; label: string }>()
  for (const job of jobs) for (const row of [...job.summary, ...job.runs]) {
    if (!numeric(row.window) || !numeric(row.horizon) || row.window < 1 || row.horizon < 1) continue
    const protocol = comparisonProtocol(job), key = JSON.stringify([protocol, row.window, row.horizon])
    cells.set(key, { key, protocol, window: row.window, horizon: row.horizon,
      label: `${job.request.evaluation.target_column} · ${row.window}/${row.horizon} · ${job.status.preset} · ${job.request.evaluation.data_path}` })
  }
  return [...cells.values()].sort((a, b) => a.window - b.window || a.horizon - b.horizon)
}
export function metricValue(row: Row, metric: ComparisonMetric): number | null {
  if (metric === 'mae' || metric === 'mse') return numeric(row[metric]) && row[metric] >= 0 ? row[metric] : null
  const error = row[metric === 'mae_ratio' ? 'mae' : 'mse'], baseline = row[metric === 'mae_ratio' ? 'persistence_mae' : 'persistence_mse']
  if (numeric(error) && error >= 0 && numeric(baseline)) return baseline > 0 ? error / baseline : null
  return numeric(row[metric]) && row[metric] >= 0 ? row[metric] : null
}
export const replicateKey = (row: Row) => row.seed == null || row.fold == null ? null : JSON.stringify([
  row.seed, row.fold, row.split ?? null, row.train_samples ?? null, row.validation_samples ?? null, row.test_samples ?? null,
])
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length

export function comparisonData(jobs: ComparisonJob[], window: number, horizon: number, metric: ComparisonMetric, paired: boolean) {
  const sets = jobs.map(job => {
    const rows = job.runs.filter(r => r.window === window && r.horizon === horizon && metricValue(r, metric) !== null)
    const counts = new Map<string, number>()
    rows.forEach(r => { const key = replicateKey(r); if (key) counts.set(key, (counts.get(key) ?? 0) + 1) })
    return { job, rows, unique: new Set([...counts].filter(([, n]) => n === 1).map(([key]) => key)) }
  })
  const shared = sets.length ? [...sets[0].unique].filter(key => sets.every(s => s.unique.has(key))) : []
  return sets.map(({ job, rows }) => {
    const included = paired ? rows.filter(r => shared.includes(replicateKey(r) ?? '')) : rows
    const values = included.map(r => metricValue(r, metric)!).filter(numeric)
    const parameters = included.map(r => r.parameters).filter((p): p is number => numeric(p) && Number.isInteger(p) && p >= 0)
    return { id: job.id, name: job.status.architecture_name, label: comparisonRunLabel(job), tag: comparisonRunTag(job),
      color: comparisonColor(job), variant: 'archive' in job ? job.archive.variant : null,
      mean: values.length ? mean(values) : null,
      min: values.length ? Math.min(...values) : null, max: values.length ? Math.max(...values) : null,
      n: values.length, available: rows.length, seeds: new Set(included.map(r => r.seed).filter(v => v != null)).size,
      folds: new Set(included.map(r => r.fold).filter(v => v != null)).size,
      parameters: parameters.length && parameters.length === included.length && parameters.every(p => p === parameters[0]) ? parameters[0] : null,
      included,
      training: included.map(r => r.train_seconds).filter(numeric),
    }
  })
}

export function paretoIds(data: ReturnType<typeof comparisonData>) {
  const valid = data.filter(r => r.mean !== null && r.parameters !== null)
  return new Set(valid.filter(r => !valid.some(o => o.id !== r.id && o.mean! <= r.mean! && o.parameters! <= r.parameters! && (o.mean! < r.mean! || o.parameters! < r.parameters!))).map(r => r.id))
}

/** Prefer a group with several comparable models over the most recent singleton. */
export function comparisonGroups(jobs: ComparisonJob[]) {
  const groups = new Map<string, { key: string; jobs: ComparisonJob[] }>()
  for (const job of jobs) {
    const key = comparisonProtocol(job)
    const group = groups.get(key) ?? { key, jobs: [] }
    group.jobs.push(job); groups.set(key, group)
  }
  return [...groups.values()].sort((a, b) => b.jobs.length - a.jobs.length)
}

export function defaultComparisonCell(jobs: ComparisonJob[]) {
  return [...comparisonCells(jobs)].sort((a, b) => {
    const coverage = (cell: typeof a) => jobs.filter(job => job.runs.some(row => row.window === cell.window && row.horizon === cell.horizon)).length
    return coverage(b) - coverage(a) || a.window - b.window || a.horizon - b.horizon
  })[0]
}

/** Match within each cell; do not treat changing windows or horizons as replicates. */
export function comparisonProfile(jobs: ComparisonJob[], metric: ComparisonMetric, paired: boolean) {
  return comparisonCells(jobs).map(cell => {
    const scores = comparisonData(jobs, cell.window, cell.horizon, metric, paired)
    return { window: cell.window, horizon: cell.horizon, label: `${cell.window} / ${cell.horizon}`, scores }
  })
}

export function meanRecorded(rows: Job['runs'], key: string): { value: number | null; n: number } {
  const values = rows.map(r => r[key]).filter(numeric)
  return { value: values.length ? mean(values) : null, n: values.length }
}

export function trainingRecipe(job: ComparisonJob) {
  const e = job.request.evaluation
  return JSON.stringify(canonical([e.epochs, e.learning_rate, e.batch_size, e.loss, e.shuffle,
    e.adam_beta1, e.adam_beta2, e.adam_epsilon, e.adam_amsgrad,
    e.early_stopping_patience, e.early_stopping_min_delta, e.restore_best_weights]))
}

/** A single evaluation cell can still be compared across its forecast leads. */
export function comparisonLeadProfile(jobs: ComparisonJob[], window: number, horizon: number, metric: ComparisonMetric, paired: boolean) {
  const cell = comparisonData(jobs, window, horizon, metric, paired)
  const parentKey = (row: Row) => JSON.stringify([row.seed, row.fold, row.split ?? null, row.trial_id ?? null])
  return Array.from({ length: horizon }, (_, i) => {
    const lead = i + 1
    const leadJobs = jobs.map((job, index) => {
      const parents = new Set(cell[index].included.map(parentKey))
      return { ...job, runs: job.per_lead.filter(r => r.window === window && r.horizon === horizon && r.lead === lead && parents.has(parentKey(r))) }
    })
    return { window, horizon, label: `Lead ${lead}`, scores: comparisonData(leadJobs, window, horizon, metric, paired) }
  })
}
