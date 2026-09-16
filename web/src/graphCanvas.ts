import type { GraphSpec, GraphNodeSpec, GraphNodeInfo, GraphViewState, GraphEdgeSpec } from './types'

export const uid = (prefix = 'node') => `${prefix}-${crypto.randomUUID().slice(0, 8)}`
export const emptyView = (): GraphViewState => ({ positions: {}, collapsed: [], direction: 'LR', renderer: 'flow', version: 1 })
export function safeView(raw: unknown, graph: GraphSpec): GraphViewState {
  const value = raw as Partial<GraphViewState> | undefined
  // Manual placements belong to the previous freeform editor. Arrange those
  // views afresh and refit rather than restoring a camera at an obsolete node.
  const hasManualPositions = value?.positions && Object.keys(value.positions).length > 0
  const camera = value?.renderer === 'flow' && value.version === 1 && !hasManualPositions ? value.camera : undefined
  return { ...emptyView(), collapsed: Array.isArray(value?.collapsed) ? value.collapsed.filter(id => graph.groups.some(g => g.id === id)) : graph.groups.map(g => g.id),
    ...(Array.isArray(value?.expandedStages) ? { expandedStages: value.expandedStages.filter((id): id is string => typeof id === 'string') } : {}),
    ...(camera && [camera.panX, camera.panY, camera.zoom].every(Number.isFinite) && camera.zoom > 0 ? { camera: { panX: camera.panX, panY: camera.panY, zoom: Math.min(3, Math.max(.025, camera.zoom)) } } : {}) }
}
export const edgeId = (e: GraphEdgeSpec) => JSON.stringify([e.source, e.target, e.port])
export const groupId = (id: string) => `group:${id}`
export function portsFor(node: GraphNodeSpec, graph: GraphSpec, info: Record<string, GraphNodeInfo>): string[] {
  return info[node.id]?.ports ?? (node.kind === 'source'
    ? [...new Set(graph.edges.filter(e => e.target === node.id).map(e => e.port))]
    : ['add', 'multiply', 'concat'].includes(node.kind) ? ['a', 'b'] : ['x'])
}
export function connectGraph(graph: GraphSpec, edge: GraphEdgeSpec, replacing?: string): GraphSpec {
  const edges = graph.edges.filter(e => edgeId(e) !== replacing)
  if (edges.some(e => e.target === edge.target && e.port === edge.port)) throw new Error('This input already has a connection. Reconnect or delete that edge first.')
  if (!graph.nodes.some(n => n.id === edge.source) || !graph.nodes.some(n => n.id === edge.target)) throw new Error('Connection endpoint is missing.')
  const visited = new Set<string>()
  const pending = [edge.target]
  while (pending.length) {
    const id = pending.pop()!
    if (id === edge.source) throw new Error('This connection would create a cycle. Use a recurrent operation for recurrence.')
    if (visited.has(id)) continue
    visited.add(id)
    pending.push(...edges.filter(e => e.source === id).map(e => e.target))
  }
  return { ...graph, edges: [...edges, edge] }
}
// One atomic edit: detach a unary layer, heal its old chain, then split the target edge.
// Ambiguous branches and graph endpoints are deliberately left for explicit wiring.
export function insertOnEdge(graph: GraphSpec, node: GraphNodeSpec, targetEdge: string, info: Record<string, GraphNodeInfo> = {}): GraphSpec {
  if (graph.locked) throw new Error('Clone to edit this model.')
  const edge = graph.edges.find(e => edgeId(e) === targetEdge)
  if (!edge) throw new Error('The connection is no longer available.')
  if (edge.source === node.id || edge.target === node.id) throw new Error('Choose a connection between other layers.')
  const ports = portsFor(node, graph, info)
  if (ports.length !== 1 || ['placeholder', 'output', 'get_attr'].includes(info[node.id]?.category ?? '')) throw new Error('Only single-input layers can be inserted into a connection. Wire multi-input operations explicitly.')
  if (graph.output === node.id) throw new Error('Choose another forecast output before moving this layer into a connection.')
  const incoming = graph.edges.filter(e => e.target === node.id)
  const outgoing = graph.edges.filter(e => e.source === node.id)
  if (incoming.length > 1 || outgoing.length > 1 || (!incoming.length && outgoing.length)) throw new Error('This layer has branching connections. Reconnect its branches explicitly before moving it.')
  const sourceGroup = graph.nodes.find(n => n.id === edge.source)?.group
  const targetGroup = graph.nodes.find(n => n.id === edge.target)?.group
  const inserted = { ...node, group: sourceGroup === targetGroup ? targetGroup : undefined }
  let next: GraphSpec = { ...graph,
    nodes: graph.nodes.some(n => n.id === node.id) ? graph.nodes.map(n => n.id === node.id ? inserted : n) : [...graph.nodes, inserted],
    edges: graph.edges.filter(e => e.source !== node.id && e.target !== node.id && edgeId(e) !== targetEdge),
  }
  if (incoming.length && outgoing.length) next = connectGraph(next, { ...outgoing[0], source: incoming[0].source })
  next = connectGraph(next, { source: edge.source, target: node.id, port: ports[0] })
  return connectGraph(next, { ...edge, source: node.id })
}

export function duplicateGraph(graph: GraphSpec, ids: string[]): GraphSpec {
  const mapping = new Map(ids.map(id => [id, uid()]))
  const refs = new Map<string, string>()
  const nodes = graph.nodes.filter(n => mapping.has(n.id)).map(n => {
    if (n.module_ref && !refs.has(n.module_ref)) refs.set(n.module_ref, uid('weights'))
    return { ...structuredClone(n), id: mapping.get(n.id)!, module_ref: n.module_ref ? refs.get(n.module_ref) : undefined, label: `${n.label ?? n.kind} copy` }
  })
  return { ...graph, nodes: [...graph.nodes, ...nodes], edges: [...graph.edges,
    ...graph.edges.filter(e => mapping.has(e.target)).map(e => ({ ...e, source: mapping.get(e.source) ?? e.source, target: mapping.get(e.target)! }))] }
}
export function appendGraph(graph: GraphSpec, template: GraphSpec, info: Record<string, GraphNodeInfo>): GraphSpec {
  const prefix = uid('model'), group = `${prefix}-group`
  const mapping = Object.fromEntries(template.nodes.map(n => [n.id, `${prefix}:${n.id}`]))
  const sourceMapping = Object.fromEntries(Object.keys(template.sources).map(k => [k, uid('source')]))
  const refs = new Map<string, string>()
  const nodes = template.nodes.map(n => {
    if (info[n.id]?.category === 'placeholder') return { id: mapping[n.id], kind: 'activation', params: { kind: 'linear' }, group, label: 'Model input · connect history here' }
    if (n.module_ref && !refs.has(n.module_ref)) refs.set(n.module_ref, uid('weights'))
    return { ...structuredClone(n), id: mapping[n.id], group,
      source_ref: n.source_ref ? { source: sourceMapping[n.source_ref.source], node: n.source_ref.node } : undefined,
      module_ref: n.module_ref ? refs.get(n.module_ref) : undefined }
  })
  return { ...graph, sources: { ...graph.sources, ...Object.fromEntries(Object.entries(template.sources).map(([k, v]) => [sourceMapping[k], v])) },
    nodes: [...graph.nodes, ...nodes], edges: [...graph.edges, ...template.edges.map(e => ({ ...e, source: mapping[e.source], target: mapping[e.target] }))],
    groups: [...graph.groups, { id: group, label: template.name }] }
}
