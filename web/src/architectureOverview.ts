import type { GraphNodeInfo, GraphNodeSpec, GraphSpec, GraphViewState } from './types'

export type OverviewStage = { id: string; label: string; nodeIds: string[]; primaryId: string; originalGroup?: string }
export type ArchitectureOverview = { graph: GraphSpec; stages: OverviewStage[]; membership: Map<string, string>; tuckedNodeIds: Set<string> }

const friendly: Record<string, string> = { causal_conv: 'Conv1D', dense: 'Dense', hyper_dense: 'HyperDense', max_pool: 'Max pool', mean_pool: 'Mean pool', flatten: 'Flatten', dropout: 'Dropout', lstm: 'LSTM', gru: 'GRU', tcn: 'TCN', tsmixer: 'TSMixer', patchtst: 'PatchTST', transformer: 'Transformer', layer_norm: 'Layer norm' }
const native = (n: GraphNodeSpec) => n.kind === 'source' && n.source_ref?.node === n.id

/** A presentation-only partition. The executable graph is never edited here. */
export function architectureOverview(graph: GraphSpec, info: Record<string, GraphNodeInfo>): ArchitectureOverview {
  const byId = new Map(graph.nodes.map(n => [n.id, n]))
  const live = new Set<string>(), pending = [graph.output]
  const incoming = new Map<string, string[]>()
  for (const e of graph.edges) incoming.set(e.target, [...incoming.get(e.target) ?? [], e.source])
  while (pending.length) { const id = pending.pop()!; if (live.has(id)) continue; live.add(id); pending.push(...incoming.get(id) ?? []) }
  const stages = new Map<string, OverviewStage>(), membership = new Map<string, string>()
  const tuckedNodeIds = new Set(graph.nodes.filter(n => !live.has(n.id) && native(n) && !n.module_ref && /^(getitem|getattr|zeros_like)(?:_\d+)?$/.test(n.id) && info[n.id]?.category !== 'call_module').map(n => n.id))
  const stageName = (label: string, nodes: GraphNodeSpec[]) => {
    const hyper = nodes.find(n => n.kind === 'hyper_dense' || /hyperdense/i.test(info[n.id]?.label ?? ''))
    if (hyper && /dense|hyper/i.test(label)) return 'HyperDense'
    if (label === 'activation') return info[nodes[0]?.id]?.label ?? 'Activation'
    return friendly[label] ?? label
  }
  for (const g of graph.groups) {
    const members = graph.nodes.filter(n => n.group === g.id && (live.has(n.id) || tuckedNodeIds.has(n.id)))
    // Disconnected user layers remain separate, including edits inside a live stage.
    if (!members.some(n => live.has(n.id))) continue
    const primary = members.find(n => n.kind === 'hyper_dense' || /hyperdense/i.test(info[n.id]?.label ?? ''))
      ?? (/conv/.test(g.label) ? members.find(n => /conv[123]d/i.test(info[n.id]?.label ?? n.kind)) : undefined)
      ?? members.find(n => info[n.id]?.category === 'call_module') ?? members[0]
    stages.set(g.id, { id: g.id, label: stageName(g.label, members), nodeIds: members.map(n => n.id), primaryId: primary.id, originalGroup: g.id })
    members.forEach(n => membership.set(n.id, g.id))
  }
  // Ungrouped real operations start as singletons. Only known source scaffolding
  // is folded automatically; a newly added draft never disappears into a stage.
  for (const n of graph.nodes) if (!membership.has(n.id) && live.has(n.id)) {
    const id = `overview:${n.id}`
    stages.set(id, { id, label: n.label ?? info[n.id]?.label ?? friendly[n.kind] ?? n.kind, nodeIds: [n.id], primaryId: n.id })
    membership.set(n.id, id)
  }
  const neighbors = (id: string, direction: 'in' | 'out') => new Set(graph.edges.filter(e => live.has(e.source) && live.has(e.target) && membership.get(direction === 'in' ? e.target : e.source) === id).map(e => membership.get(direction === 'in' ? e.source : e.target)).filter((other): other is string => Boolean(other) && other !== id))
  const linear = (a: string, b: string) => neighbors(a, 'out').size === 1 && neighbors(a, 'out').has(b) && neighbors(b, 'in').size === 1 && neighbors(b, 'in').has(a)
  const merge = (keep: OverviewStage, other: OverviewStage) => {
    keep.nodeIds.push(...other.nodeIds); other.nodeIds.forEach(id => membership.set(id, keep.id)); stages.delete(other.id)
  }
  const input = graph.nodes.find(n => native(n) && info[n.id]?.category === 'placeholder')
  const inputStage = input && stages.get(membership.get(input.id)!)
  if (inputStage) {
    inputStage.label = 'Input'
    let changed = true
    while (changed) {
      changed = false
      for (const s of stages.values()) {
        const n = byId.get(s.primaryId)!
        if (s.id !== inputStage.id && !s.originalGroup && native(n) && /^getitem(?:_\d+)?$/.test(n.id) && neighbors(s.id, 'in').size === 1 && neighbors(s.id, 'in').has(inputStage.id)) { merge(inputStage, s); changed = true }
      }
    }
    for (const id of tuckedNodeIds) if (!membership.has(id) && !byId.get(id)?.group) { inputStage.nodeIds.push(id); membership.set(id, inputStage.id) }
  }
  const terminal = byId.get(graph.output)
  const forecast = terminal && info[terminal.id]?.category === 'output' ? stages.get(membership.get(terminal.id)!) : undefined
  if (forecast) {
    forecast.label = 'Forecast'
    const before = [...neighbors(forecast.id, 'in')][0], predecessor = before && stages.get(before)
    if (predecessor && !predecessor.originalGroup && predecessor.nodeIds.length === 1 && native(byId.get(predecessor.primaryId)!) && linear(predecessor.id, forecast.id)) {
      forecast.primaryId = predecessor.primaryId; merge(forecast, predecessor)
    }
  }
  // Convolution + activation, Dense + activation/dropout, just like a paper
  // figure. Never merge across a branch or a multi-input boundary.
  for (const s of [...stages.values()]) {
    if (!stages.has(s.id) || !s.originalGroup) continue
    const sourceLabel = graph.groups.find(g => g.id === s.originalGroup)?.label ?? ''
    if (!['activation', 'dropout'].includes(sourceLabel)) continue
    const predecessor = stages.get([...neighbors(s.id, 'in')][0])
    if (!predecessor || predecessor === inputStage || !linear(predecessor.id, s.id)) continue
    if (sourceLabel === 'activation') predecessor.label += ` + ${s.label}`
    merge(predecessor, s)
  }
  for (const s of [...stages.values()]) {
    if (!stages.has(s.id) || graph.groups.find(g => g.id === s.originalGroup)?.label !== 'flatten') continue
    const next = stages.get([...neighbors(s.id, 'out')][0])
    if (next && linear(s.id, next.id) && next !== inputStage) merge(next, s)
  }
  // Preserve standalone drafts and wholly disconnected user groups as editable
  // stages, rather than mistaking them for native tracing helpers.
  for (const g of graph.groups) {
    const members = graph.nodes.filter(n => n.group === g.id && !membership.has(n.id))
    if (!members.length || stages.has(g.id)) continue
    stages.set(g.id, { id: g.id, label: stageName(g.label, members), nodeIds: members.map(n => n.id), primaryId: members[0].id, originalGroup: g.id })
    members.forEach(n => membership.set(n.id, g.id))
  }
  for (const s of stages.values()) if (!s.originalGroup && s.nodeIds.length === 1 && s !== inputStage && s !== forecast) { membership.delete(s.nodeIds[0]); stages.delete(s.id) }
  const order = new Map(graph.nodes.map((n, i) => [n.id, i]))
  const summaries = [...stages.values()].sort((a, b) => Math.min(...a.nodeIds.map(id => order.get(id)!)) - Math.min(...b.nodeIds.map(id => order.get(id)!)))
  for (const s of summaries) s.nodeIds.sort((a, b) => order.get(a)! - order.get(b)!)
  return { graph: { ...graph, groups: summaries.map(({ id, label }) => ({ id, label })), nodes: graph.nodes.map(n => ({ ...n, group: membership.get(n.id) })) }, stages: summaries, membership, tuckedNodeIds }
}

export function overviewView(overview: ArchitectureOverview, view: GraphViewState): GraphViewState {
  return { ...view, collapsed: overview.stages.filter(s => !(view.expandedStages ?? []).includes(s.id)).map(s => s.id) }
}
