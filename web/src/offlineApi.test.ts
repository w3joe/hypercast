import { beforeEach, describe, expect, it, vi } from 'vitest'
import { offlineApi, resetOfflineApiForTests } from './offlineApi'
import type { Catalog, GraphSpec } from './types'

const graph: GraphSpec = {
  schema_version: 2,
  revision: 'test',
  name: 'Offline model',
  sources: { s0: { schema_version: 1, name: 'Preset', input: { representation: 'levels', feature_order: [0, 1, 2, 3] }, layers: [], head: { type: 'direct', zero_initialize: false }, preset_id: 'preset' } },
  nodes: [
    { id: 'input', kind: 'source', source_ref: { source: 's0', node: 'input' }, params: {} },
    { id: 'dense', kind: 'source', source_ref: { source: 's0', node: 'dense' }, params: {} },
  ],
  edges: [{ source: 'input', target: 'dense', port: 'args/0' }],
  groups: [],
  output: 'dense',
  preset_id: 'preset',
}

const catalog = {
  schema_version: 1, categories: [], input_representations: [], head_types: [], algebras: ['quaternion'], algebra_dimensions: { quaternion: 4 }, activations: [], presets: [], evaluation_presets: {}, evaluation_defaults: {},
  offline: { reference_window: 10, reference_horizon: 1, preset_count: 1 },
} as unknown as Catalog

const manifest = {
  schema_version: 1,
  reference_cell: { window: 10, horizon: 1 },
  catalog,
  presets: { preset: { graph, parameters: 64, warnings: [], graph_nodes: {
    input: { label: 'Input', ports: [], shape: [2, 10, 4], settings: {}, category: 'placeholder' },
    dense: { label: 'Linear', ports: ['args/0'], shape: [2, 10, 4], settings: { out_features: 4, bias: true }, category: 'call_module' },
  } } },
}

const archivedResults = {
  schema_version: 1,
  source: 'tslib15-20260915',
  result_count: 120,
  fit_count: 360,
  results: Array.from({ length: 120 }, (_, index) => ({
    id: `archive-${index}`,
    request: { evaluation: {} },
    status: { state: 'complete', architecture_name: `Archived ${index}` },
    summary: [],
    runs: [{ seed: 1 }, { seed: 2 }, { seed: 3 }],
    per_lead: [],
    archive: { source: 'tslib15-20260915' },
  })),
}

beforeEach(() => {
  const values = new Map<string, string>()
  vi.stubGlobal('localStorage', {
    getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => values.set(key, value),
    removeItem: (key: string) => values.delete(key),
    clear: () => values.clear(),
    key: (index: number) => [...values.keys()][index] ?? null,
    get length() { return values.size },
  })
  localStorage.clear()
  resetOfflineApiForTests()
})

describe('offline API', () => {
  it('loads the static manifest without requesting an API route', async () => {
    const fetcher = vi.fn(async (path: RequestInfo | URL) => {
      expect(String(path)).toBe('/offline-manifest.json')
      return new Response(JSON.stringify(manifest), { status: 200 })
    })
    vi.stubGlobal('fetch', fetcher)
    expect((await offlineApi.catalog()).offline?.preset_count).toBe(1)
    expect((await offlineApi.convert(manifest.presets.preset.graph)).nodes).toHaveLength(2)
    expect((await offlineApi.describeGraph(graph)).parameters).toBe(64)
    expect(fetcher).toHaveBeenCalledTimes(1)
    expect(fetcher.mock.calls.some(([path]) => String(path).startsWith('/api'))).toBe(false)
  })

  it('saves and reopens editable graphs in browser storage', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify(manifest), { status: 200 })))
    const editable = { ...graph, locked: false, preset_id: undefined, name: 'My browser design' }
    const record = await offlineApi.saveGraph(editable, { positions: {}, collapsed: [], direction: 'LR' })
    expect(record.id).toBeTruthy()
    expect((await offlineApi.graphRecords())[0].spec.name).toBe('My browser design')
    const updated = await offlineApi.saveGraph({ ...editable, name: 'Updated' }, record.view!, record.id)
    expect(updated.id).toBe(record.id)
    expect((await offlineApi.graphRecords())).toHaveLength(1)
  })

  it('rejects compute actions with an offline explanation', async () => {
    vi.stubGlobal('fetch', vi.fn(async (path: RequestInfo | URL) => {
      expect(String(path)).toBe('/offline-results.json')
      return new Response(JSON.stringify(archivedResults), { status: 200 })
    }))
    await expect(offlineApi.submit()).rejects.toThrow(/no compute connection/i)
    await expect(offlineApi.previewWeights()).rejects.toThrow(/no compute connection/i)
    const examples = await offlineApi.jobs()
    expect(examples).toHaveLength(120)
    expect(examples.every(job => job.status.state === 'complete')).toBe(true)
    expect(examples.reduce((total, job) => total + job.runs.length, 0)).toBe(360)
    await expect(offlineApi.comparisonArchives()).resolves.toEqual([])
  })
})
