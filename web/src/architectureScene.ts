import ELK from 'elkjs/lib/elk.bundled.js'
import type { ElkNode } from 'elkjs/lib/elk-api'
import type { GraphSpec, GraphNodeInfo, GraphViewState } from './types'
import { edgeId, groupId, portsFor } from './graphCanvas'

export type Point = { x: number; y: number }
export type ScenePort = { id: string; nodeId: string; label: string; port?: string }
export type SceneBlock = { id: string; nodeIds: string[]; label: string; shape: string; kind: string; color: string; stage?: string; disconnected: boolean; missingInputs: string[]; output: boolean; x: number; y: number; width: number; height: number; inputs: ScenePort[]; outputs: ScenePort[] }
export type SceneRoute = { id: string; source: string; target: string; port: string; sourceBlock: string; targetBlock: string; sourceHandle: string; targetHandle: string; points: Point[]; insert: Point; label: string }
export type ArchitectureScene = { blocks: SceneBlock[]; routes: SceneRoute[]; width: number; height: number; disconnectedY?: number }
export const operationColors = { dense: '#426cba', hyper: '#6b3eda', convolution: '#bd861e', pooling: '#ba6582', recurrent: '#007f9b', attention: '#8f59c2', structural: '#8391ad', input: '#009dc0', output: '#011476', stage: '#011476' }
const elk = new ELK()
// React Flow uses handle IDs inside DOM selectors while connecting. Encode the
// presentation ID, retaining canonical operation IDs and ports in ScenePort.
const inputId = (node: string, port: string) => encodeURIComponent(JSON.stringify(['in', node, port]))
const outputId = (node: string) => encodeURIComponent(JSON.stringify(['out', node]))
export function operationStyle(label: string) {
  const s = label.toLowerCase()
  const kind = /hyper|quaternion|octonion/.test(s) ? 'hyper' : /attention|transformer|patchtst|informer|autoformer/.test(s) ? 'attention' : /conv|tcn/.test(s) ? 'convolution' : /pool/.test(s) ? 'pooling' : /gru|lstm|rnn/.test(s) ? 'recurrent' : /dense|linear/.test(s) ? 'dense' : /input|placeholder/.test(s) ? 'input' : 'structural'
  return { kind, color: operationColors[kind] }
}
export function formatShape(shape: unknown): string { return shape == null ? 'Shape unavailable' : Array.isArray(shape) && shape.every(x => typeof x === 'number') ? shape.join(' × ') : JSON.stringify(shape) }

/** Project immutable executable operations into rectangular cards and exact ports. */
export async function buildArchitectureScene(graph: GraphSpec, info: Record<string, GraphNodeInfo>, view: GraphViewState, overview?: { tuckedNodeIds: Set<string>; stages: { id: string; primaryId: string }[] }): Promise<ArchitectureScene> {
  const byId = new Map(graph.nodes.map(n => [n.id, n]))
  const needed = new Set<string>(), pending = [graph.output], incoming = new Map<string, string[]>()
  for (const e of graph.edges) incoming.set(e.target, [...incoming.get(e.target) ?? [], e.source])
  while (pending.length) { const id = pending.pop()!; if (needed.has(id)) continue; needed.add(id); pending.push(...incoming.get(id) ?? []) }
  const collapsed = new Set(view.collapsed.filter(id => graph.groups.some(g => g.id === id)))
  const liveGroups = new Set(graph.nodes.filter(n => needed.has(n.id)).map(n => n.group))
  const visibleId = (id: string) => { const n = byId.get(id); return n?.group && collapsed.has(n.group) && (needed.has(id) || !liveGroups.has(n.group) || overview?.tuckedNodeIds.has(id)) ? groupId(n.group) : id }
  const visible = new Map<string, string[]>()
  for (const n of graph.nodes) visible.set(visibleId(n.id), [...visible.get(visibleId(n.id)) ?? [], n.id])
  const blocks: SceneBlock[] = [...visible].map(([id, ids]) => {
    const members = new Set(ids), node = byId.get(ids[0])!, stage = id.startsWith('group:') ? node.group : undefined
    const primary = byId.get(overview?.stages.find(s => s.id === stage)?.primaryId ?? '') ?? node
    const label = stage ? graph.groups.find(g => g.id === stage)!.label : node.label ?? info[node.id]?.label ?? node.kind
    const style = operationStyle(`${label} ${info[primary.id]?.label ?? ''} ${primary.kind}`)
    const boundary = graph.edges.filter(e => members.has(e.source) && !members.has(e.target))
    const external = [...new Set(boundary.map(e => formatShape(info[e.source]?.shape)))]
    const inputs = ids.flatMap(nodeId => [...new Set([...portsFor(byId.get(nodeId)!, graph, info), ...graph.edges.filter(e => e.target === nodeId).map(e => e.port)])].filter(port => !graph.edges.some(e => e.target === nodeId && e.port === port && members.has(e.source))).map(port => ({ id: inputId(nodeId, port), nodeId, port, label: stage ? `${info[nodeId]?.label ?? nodeId} · ${port}` : port })))
    const missingInputs = inputs.filter(p => !graph.edges.some(e => e.target === p.nodeId && e.port === p.port)).map(p => `${p.nodeId} · ${p.port}`)
    const outputNodes = [...new Set(boundary.map(e => e.source))]
    if (!outputNodes.length) outputNodes.push(...ids.filter(id => id === graph.output || !graph.edges.some(e => e.source === id && members.has(e.target))))
    const outputs = outputNodes.map(nodeId => ({ id: outputId(nodeId), nodeId, label: info[nodeId]?.label ?? byId.get(nodeId)?.label ?? nodeId }))
    const output = members.has(graph.output), disconnected = !ids.some(id => needed.has(id))
    return { id, nodeIds: ids, label, stage, kind: output ? 'output' : style.kind, color: output ? operationColors.output : style.color,
      shape: stage && external.length ? external.length > 1 ? `${external.length} output tensors` : external[0] : formatShape(info[output ? graph.output : primary.id]?.shape),
      inputs, outputs, missingInputs, output, disconnected, x: 0, y: 0, width: 224, height: Math.max(missingInputs.length ? 166 : 144, 46 + Math.max(inputs.length, outputs.length) * 22) }
  })
  const specs = graph.edges.filter(e => byId.has(e.source) && byId.has(e.target) && visibleId(e.source) !== visibleId(e.target))
  const routes: SceneRoute[] = []
  let width = 0, height = 0, disconnectedY: number | undefined
  const makeRoute = (e: typeof specs[number], points: Point[]): SceneRoute => {
    let longest = 0, index = 0
    for (let i = 0; i < points.length - 1; i++) { const distance = Math.hypot(points[i + 1].x - points[i].x, points[i + 1].y - points[i].y); if (distance > longest) { longest = distance; index = i } }
    return { ...e, id: edgeId(e), sourceBlock: visibleId(e.source), targetBlock: visibleId(e.target), sourceHandle: outputId(e.source), targetHandle: inputId(e.target, e.port), points,
      insert: { x: (points[index].x + points[index + 1].x) / 2, y: (points[index].y + points[index + 1].y) / 2 }, label: `${info[e.source]?.label ?? e.source} → ${info[e.target]?.label ?? e.target} · ${e.port}` }
  }
  for (const disconnected of [false, true]) {
    const subset = blocks.filter(b => b.disconnected === disconnected), ids = new Set(subset.map(b => b.id))
    if (!subset.length) continue
    if (disconnected) disconnectedY = height + 32
    const yOffset = disconnected ? height + 92 : 38
    const localEdges = specs.filter(e => ids.has(visibleId(e.source)) && ids.has(visibleId(e.target)))
    const layout = await elk.layout({ id: 'root', layoutOptions: { 'elk.algorithm': 'layered', 'elk.direction': 'RIGHT', 'elk.edgeRouting': 'ORTHOGONAL', 'elk.spacing.nodeNode': '65', 'elk.layered.spacing.nodeNodeBetweenLayers': '54', 'elk.layered.considerModelOrder.strategy': 'NODES_AND_EDGES' },
      children: subset.map(b => ({ id: b.id, width: b.width, height: b.height, layoutOptions: { 'elk.portConstraints': 'FIXED_POS' }, ports: [
        ...b.inputs.map((p, i) => ({ id: p.id, x: 0, y: (i + 1) * b.height / (b.inputs.length + 1), width: 0, height: 0, layoutOptions: { 'elk.port.side': 'WEST' } })),
        ...b.outputs.map((p, i) => ({ id: p.id, x: b.width, y: (i + 1) * b.height / (b.outputs.length + 1), width: 0, height: 0, layoutOptions: { 'elk.port.side': 'EAST' } })),
      ] })), edges: localEdges.map(e => ({ id: edgeId(e), sources: [outputId(e.source)], targets: [inputId(e.target, e.port)] })) } as ElkNode)
    for (const n of layout.children ?? []) { const b = subset.find(b => b.id === n.id)!; b.x = (n.x ?? 0) + 38; b.y = (n.y ?? 0) + yOffset }
    for (const edge of layout.edges ?? []) {
      const section = edge.sections?.[0], spec = localEdges.find(e => edgeId(e) === edge.id)!
      if (section) routes.push(makeRoute(spec, [section.startPoint, ...section.bendPoints ?? [], section.endPoint].map(p => ({ x: p.x + 38, y: p.y + yOffset }))))
    }
    width = Math.max(width, (layout.width ?? 0) + 76); height = yOffset + (layout.height ?? 0) + 38
  }
  const portPosition = (b: SceneBlock, handle: string, output: boolean) => {
    const ports = output ? b.outputs : b.inputs, index = Math.max(0, ports.findIndex(p => p.id === handle))
    return { x: b.x + (output ? b.width : 0), y: b.y + b.height * (index + 1) / (ports.length + 1) }
  }
  for (const [index, e] of specs.filter(e => !routes.some(r => r.id === edgeId(e))).entries()) {
    const source = blocks.find(b => b.id === visibleId(e.source))!, target = blocks.find(b => b.id === visibleId(e.target))!
    const start = portPosition(source, outputId(e.source), true), end = portPosition(target, inputId(e.target, e.port), false), lane = (disconnectedY ?? height) + 20 + index * 6
    routes.push(makeRoute(e, [start, { x: start.x + 26, y: start.y }, { x: start.x + 26, y: lane }, { x: end.x - 26, y: lane }, { x: end.x - 26, y: end.y }, end]))
  }
  width = Math.max(width, ...blocks.map(b => b.x + b.width + 38)); height = Math.max(height, ...blocks.map(b => b.y + b.height + 38))
  return { blocks, routes, width: Math.max(400, width), height: Math.max(260, height), disconnectedY }
}
