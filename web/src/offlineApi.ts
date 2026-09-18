import { prepareDenseSwap } from './layerSwap'
import type {
  ArchitectureSpec,
  ArchivedComparisonJob,
  Catalog,
  GraphNodeInfo,
  GraphRecord,
  GraphSpec,
  GraphValidation,
  GraphViewState,
} from './types'

type PresetEntry = {
  graph: GraphSpec
  graph_nodes: Record<string, GraphNodeInfo>
  parameters: number
  warnings: string[]
  auto_fit_connections: unknown[]
}

type OfflineManifest = {
  schema_version: 1
  reference_cell: { window: number; horizon: number }
  catalog: Catalog
  presets: Record<string, PresetEntry>
}

const STORAGE_KEY = 'hypercast.offline.graphs.v1'
const MAX_RECORDS = 5
let manifestPromise: Promise<OfflineManifest> | undefined
let resultsPromise: Promise<ArchivedComparisonJob[]> | undefined

function unavailable(): never {
  throw new Error('This browser-only playground has no compute connection. Export the architecture and open it in a connected Hypercast installation to run it.')
}

function manifest() {
  manifestPromise ??= fetch('/offline-manifest.json', { cache: 'force-cache' }).then(async response => {
    if (!response.ok) throw new Error('The offline architecture catalog could not be loaded.')
    return response.json() as Promise<OfflineManifest>
  })
  return manifestPromise
}

function archivedResults() {
  resultsPromise ??= fetch('/offline-results.json', { cache: 'force-cache' }).then(async response => {
    if (!response.ok) throw new Error('The read-only research results could not be loaded.')
    const data = await response.json() as { schema_version: number; result_count: number; fit_count: number; results: ArchivedComparisonJob[] }
    if (data.schema_version !== 1 || data.result_count !== 120 || data.fit_count !== 360 || data.results.length !== data.result_count) {
      throw new Error('The deployed research archive is incomplete.')
    }
    return data.results
  })
  return resultsPromise
}

function assertGraph(value: ArchitectureSpec | GraphSpec): GraphSpec {
  if (value.schema_version !== 2) throw new Error('This legacy architecture is not included in the offline preset catalog.')
  const graph = structuredClone(value) as GraphSpec
  if (!Array.isArray(graph.nodes) || !Array.isArray(graph.edges) || !Array.isArray(graph.groups) || !graph.sources || typeof graph.output !== 'string') {
    throw new Error('The imported graph is missing nodes, connections, groups, sources, or its output.')
  }
  const ids = new Set<string>()
  for (const node of graph.nodes) {
    if (!node || typeof node.id !== 'string' || typeof node.kind !== 'string' || !node.params || typeof node.params !== 'object') throw new Error('Every graph node needs an id, kind, and parameter object.')
    if (ids.has(node.id)) throw new Error(`Duplicate graph node id: ${node.id}`)
    ids.add(node.id)
  }
  if (!ids.has(graph.output)) throw new Error('The forecast output does not refer to a graph node.')
  for (const edge of graph.edges) {
    if (!ids.has(edge.source) || !ids.has(edge.target) || typeof edge.port !== 'string') throw new Error('A graph connection has an invalid endpoint or port.')
  }
  return graph
}

function readRecords(): GraphRecord[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '[]')
    return Array.isArray(parsed) ? parsed.filter(record => record && typeof record.id === 'string' && record.spec) : []
  } catch {
    return []
  }
}

function writeRecords(records: GraphRecord[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(records.slice(0, MAX_RECORDS)))
  } catch {
    throw new Error('Browser storage is full. Export a saved design, then replace an older browser save.')
  }
}

function referenceMetadata(data: OfflineManifest, graph: GraphSpec) {
  const result: Record<string, GraphNodeInfo> = {}
  const sourceMetadata = new Map<string, Map<string, GraphNodeInfo>>()
  for (const entry of Object.values(data.presets)) {
    for (const node of entry.graph.nodes) {
      if (!node.source_ref || !entry.graph_nodes[node.id]) continue
      const signature = JSON.stringify(entry.graph.sources[node.source_ref.source])
      const bySourceNode = sourceMetadata.get(signature) ?? new Map<string, GraphNodeInfo>()
      bySourceNode.set(node.source_ref.node, entry.graph_nodes[node.id])
      sourceMetadata.set(signature, bySourceNode)
    }
  }
  for (const node of graph.nodes) {
    const source = node.source_ref ? graph.sources[node.source_ref.source] : undefined
    const reference = source ? sourceMetadata.get(JSON.stringify(source))?.get(node.source_ref!.node) : undefined
    result[node.id] = reference ? structuredClone(reference) : {
      label: node.label ?? node.kind,
      ports: ['add', 'multiply', 'concat'].includes(node.kind) ? ['a', 'b'] : node.kind === 'source' ? [...new Set(graph.edges.filter(edge => edge.target === node.id).map(edge => edge.port))] : ['x'],
      shape: null,
      settings: structuredClone(node.params),
      category: node.kind === 'source' ? 'source' : 'custom',
    }
  }
  for (const node of graph.nodes) {
    if (node.kind !== 'hyper_dense' || node.params.shape_mode !== 'preserve') continue
    const edge = graph.edges.find(item => item.target === node.id && item.port === 'x')
    const inputShape = edge ? result[edge.source]?.shape : null
    const algebra = String(node.params.algebra ?? 'quaternion')
    const dimension = data.catalog.algebra_dimensions[algebra]
    if (!Array.isArray(inputShape) || inputShape.length < 2 || !inputShape.every(value => typeof value === 'number') || !dimension) continue
    const width = inputShape.at(-1) as number
    const units = Math.ceil(width / dimension)
    const padded = units * dimension
    result[node.id] = {
      ...result[node.id],
      label: node.label ?? 'HyperDense',
      shape: [...inputShape],
      settings: structuredClone(node.params),
      shape_fit: { axis: -1, input_width: width, padded_width: padded, units, output_width: width, padding: padded - width, crop: padded - width },
    }
  }
  return result
}

function exactReference(data: OfflineManifest, graph: GraphSpec) {
  const entry = graph.preset_id ? data.presets[graph.preset_id] : undefined
  if (!entry) return undefined
  const comparable = (value: GraphSpec) => JSON.stringify({ nodes: value.nodes, edges: value.edges, output: value.output, sources: value.sources })
  return comparable(entry.graph) === comparable(graph) ? entry : undefined
}

export const offlineApi = {
  catalog: async () => (await manifest()).catalog,
  convert: async (architecture: ArchitectureSpec | GraphSpec) => {
    const data = await manifest()
    if (architecture.schema_version === 2) return assertGraph(architecture)
    const presetId = architecture.preset_id
    if (!presetId || !data.presets[presetId]) throw new Error('Offline conversion is available for the bundled presets and schema 2 graph imports.')
    return structuredClone(data.presets[presetId].graph)
  },
  describeGraph: async (architecture: GraphSpec) => {
    const data = await manifest()
    const graph = assertGraph(architecture)
    const exact = exactReference(data, graph)
    return {
      graph_nodes: exact ? structuredClone(exact.graph_nodes) : referenceMetadata(data, graph),
      parameters: exact?.parameters ?? null,
      warnings: exact?.warnings ?? [],
      reference: Boolean(exact),
    }
  },
  validateGraph: async (architecture: GraphSpec, window: number, horizon: number): Promise<GraphValidation> => {
    const data = await manifest()
    const graph = assertGraph(architecture)
    const exact = exactReference(data, graph)
    if (!exact || window !== data.reference_cell.window || horizon !== data.reference_cell.horizon) unavailable()
    return { valid: true, spec: graph, parameters: exact.parameters, graph_nodes: structuredClone(exact.graph_nodes), warnings: [...exact.warnings] }
  },
  graphRecords: async () => readRecords(),
  saveGraph: async (architecture: GraphSpec, view: GraphViewState, id?: string) => {
    const graph = assertGraph(architecture)
    const records = readRecords()
    const record: GraphRecord = { id: id || crypto.randomUUID(), spec: graph, view: structuredClone(view) }
    writeRecords([record, ...records.filter(item => item.id !== record.id)])
    return record
  },
  editGraph: async (architecture: GraphSpec, edit: { action: string; id: string; kind: string; params: Record<string, unknown> }) => {
    if (edit.action !== 'replace' || !['dense', 'hyper_dense'].includes(edit.kind)) throw new Error('This edit is unavailable in the browser-only playground.')
    const data = await manifest()
    const graph = assertGraph(architecture)
    const info = referenceMetadata(data, graph)
    return { spec: prepareDenseSwap(graph, edit.id, edit.kind as 'dense' | 'hyper_dense', info, String(edit.params.algebra ?? 'quaternion')), warnings: [] }
  },
  jobs: async () => structuredClone(await archivedResults()),
  legacyRuns: async () => [],
  comparisonArchives: async () => [],
  architectures: async () => [],
  compute: async () => unavailable(),
  demoSession: async () => unavailable(),
  demoRequestCode: async () => unavailable(),
  demoVerify: async () => unavailable(),
  demoLogout: async () => unavailable(),
  forecasts: async () => unavailable(),
  archivedForecasts: async () => unavailable(),
  trainedWeights: async () => unavailable(),
  previewWeights: async () => unavailable(),
  internalGraph: async () => unavailable(),
  validate: async () => unavailable(),
  saveArchitecture: async () => unavailable(),
  submit: async () => unavailable(),
  cancel: async () => unavailable(),
  finalTest: async () => unavailable(),
  log: async () => unavailable(),
}

export function resetOfflineApiForTests() {
  manifestPromise = undefined
  resultsPromise = undefined
}
