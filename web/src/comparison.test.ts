import { describe, expect, it } from 'vitest'
import { comparisonCells, comparisonData, comparisonProtocol, metricValue, paretoIds, comparisonGroups, comparisonProfile, defaultComparisonCell, meanRecorded, comparisonLeadProfile } from './comparison'
import type { Job } from './types'

export function comparisonJob(id: string, runs: Job['runs'] = []): Job {
  return { id, request: { evaluation: { protocol: 'chronological-v1', data_path: 'copper.csv', target_column: 'Copper', preset: 'standard', epochs: 50, learning_rate: .001, loss: 'mae' } },
    status: { state: 'complete', phase: 'validation', preset: 'standard', architecture_name: id },
    runs, summary: [{ window: 32, horizon: 5 }], per_lead: [] } as unknown as Job
}
const row = (seed: number, mae: number, extra: Partial<Job['runs'][number]> = {}) => ({ window: 32, horizon: 5, seed, fold: 1, split: 'validation', train_samples: 100, validation_samples: 20, mae, persistence_mae: 2, parameters: 100, ...extra }) as Job['runs'][number]

describe('research comparison calculations', () => {
  it('matches seeds, folds and split sizes before averaging and excludes other cells', () => {
    const a = comparisonJob('a', [row(1, 1), row(2, 2), row(3, 100), row(1, 1000, { horizon: 10 })])
    const b = comparisonJob('b', [row(1, 2), row(2, 4), row(3, 1, { validation_samples: 30 })])
    const data = comparisonData([a, b], 32, 5, 'mae_ratio', true)
    expect(data.map(r => [r.mean, r.min, r.max, r.n, r.available])).toEqual([[.75, .5, 1, 2, 3], [1.5, 1, 2, 2, 3]])
  })
  it('does not count duplicate replicate keys as independent repetitions', () => {
    const data = comparisonData([comparisonJob('a', [row(1, 1), row(1, 3)]), comparisonJob('b', [row(1, 2)])], 32, 5, 'mae', true)
    expect(data.every(r => r.n === 0 && r.mean === null)).toBe(true)
  })
  it('keeps missing scores, zero denominators, and no shared replicates unavailable', () => {
    expect(metricValue({ mae: 0, persistence_mae: 2 }, 'mae_ratio')).toBe(0)
    expect(metricValue({ mae: 1, persistence_mae: 0, mae_ratio: 1 }, 'mae_ratio')).toBeNull()
    expect(metricValue({ mae: null }, 'mae')).toBeNull()
    expect(metricValue({ mae: Infinity }, 'mae')).toBeNull()
    const data = comparisonData([comparisonJob('a', [row(1, 1)]), comparisonJob('b', [row(2, 1)])], 32, 5, 'mae', true)
    expect(data.map(r => r.mean)).toEqual([null, null])
    expect(comparisonData([comparisonJob('a', [row(1, 1)])], 32, 5, 'mae', false)[0].n).toBe(1)
  })
  it('separates datasets, phases, presets and explicitly recorded fold plans', () => {
    const a = comparisonJob('a'), b = comparisonJob('b')
    expect(comparisonProtocol(a)).toBe(comparisonProtocol(b))
    b.request.evaluation.data_path = 'etth1.csv'
    expect(comparisonCells([a, b])).toHaveLength(2)
    b.request.evaluation.data_path = 'copper.csv'; b.status.phase = 'final_test'
    expect(comparisonCells([a, b])).toHaveLength(2)
    b.status.phase = 'validation'; b.request.evaluation.preset = 'robust'
    expect(comparisonCells([a, b])).toHaveLength(2)
    b.request.evaluation.preset = 'standard'
    Object.assign(b.request.evaluation, { folds: [{ train_fraction: .7 }] })
    expect(comparisonCells([a, b])).toHaveLength(2)
  })
  it('does not merge jobs with missing provenance', () => {
    const a = comparisonJob('a'), b = comparisonJob('b')
    a.request.evaluation.data_path = ''; b.request.evaluation.data_path = ''
    expect(comparisonCells([a, b])).toHaveLength(2)
  })
  it('calculates the error–parameter frontier, preserving trade-offs and ties', () => {
    const jobs = [comparisonJob('small', [row(1, 2, { parameters: 10 })]), comparisonJob('accurate', [row(1, 1, { parameters: 100 })]), comparisonJob('dominated', [row(1, 3, { parameters: 100 })]), comparisonJob('tie', [row(1, 1, { parameters: 100 })])]
    expect([...paretoIds(comparisonData(jobs, 32, 5, 'mae', true))]).toEqual(['small', 'accurate', 'tie'])
  })
})


describe('evaluation navigation and numeric summaries', () => {
  it('prefers a group with multiple models and the cell with widest actual score coverage', () => {
    const a = comparisonJob('a', [row(1, 1)]), b = comparisonJob('b', [row(1, 2)])
    const singleton = comparisonJob('robust', [row(1, 1)])
    singleton.request.evaluation.preset = 'robust'
    a.summary.push({ window: 1, horizon: 1 } as Job['summary'][number])
    expect(comparisonGroups([singleton, a, b])[0].jobs.map(j => j.id)).toEqual(['a', 'b'])
    expect(defaultComparisonCell([a, b])).toMatchObject({ window: 32, horizon: 5 })
  })
  it('matches every setting separately and leaves absent cells unavailable', () => {
    const a = comparisonJob('a', [row(1, 1), row(2, 100), row(7, 4, { window: 64, horizon: 10 })])
    const b = comparisonJob('b', [row(1, 2), row(7, 6, { window: 64, horizon: 10 })])
    const cells = comparisonProfile([a, b], 'mae', true)
    expect(cells.map(c => c.scores.map(r => [r.mean, r.n]))).toEqual([[[1, 1], [2, 1]], [[4, 1], [6, 1]]])
    b.runs.pop()
    expect(comparisonProfile([a, b], 'mae', true)[1].scores.map(r => r.mean)).toEqual([null, null])
    expect(comparisonProfile([a, b], 'mae', false)[1].scores.map(r => r.mean)).toEqual([4, null])
  })
  it('derives extra numbers from included repeats with explicit missing-value coverage', () => {
    const a = comparisonJob('a', [row(1, 1, { train_seconds: 4, bias: -.5 }), row(2, 2, { train_seconds: 100, bias: 100 })])
    const b = comparisonJob('b', [row(1, 2)])
    const included = comparisonData([a, b], 32, 5, 'mae', true)[0].included
    expect(meanRecorded(included, 'train_seconds')).toEqual({ value: 4, n: 1 })
    expect(meanRecorded(included, 'bias')).toEqual({ value: -.5, n: 1 })
    expect(meanRecorded(included, 'missing')).toEqual({ value: null, n: 0 })
    expect(meanRecorded([{ value: 0 }, { value: null }, { value: Infinity }], 'value')).toEqual({ value: 0, n: 1 })
  })
  it('does not infer a whole-model parameter count from incomplete repeat metadata', () => {
    const data = comparisonData([comparisonJob('a', [row(1, 1), row(2, 2, { parameters: null })])], 32, 5, 'mae', true)
    expect(data[0].parameters).toBeNull()
  })
})


it('uses only eligible parent repeats in the lead chart and rematches missing leads', () => {
  const a = comparisonJob('a', [row(1, 1), row(2, 900)]), b = comparisonJob('b', [row(1, 2)])
  a.per_lead = [row(1, 1, { lead: 1 }), row(1, 5, { lead: 2 }), row(2, 900, { lead: 1 })]
  b.per_lead = [row(1, 2, { lead: 1 }), row(1, 6, { lead: 2 })]
  expect(comparisonLeadProfile([a, b], 32, 5, 'mae', true).map(c => c.scores.map(r => r.mean))).toEqual([[1, 2], [5, 6], [null, null], [null, null], [null, null]])
  b.per_lead.pop()
  expect(comparisonLeadProfile([a, b], 32, 5, 'mae', true)[1].scores.map(r => r.mean)).toEqual([null, null])
})
