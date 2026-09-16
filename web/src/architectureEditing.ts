import type { GraphSpec, GraphNodeSpec, GraphNodeInfo } from './types'
import { portsFor } from './graphCanvas'

/** Layer shortcuts hide tensor plumbing in lists, never in the executable graph. */
export function isLayerOperation(node: GraphNodeSpec, info?: GraphNodeInfo) {
  return node.kind !== 'source' || info?.category === 'call_module'
}

export function replacementPorts(node: GraphNodeSpec, graph: GraphSpec, info: Record<string, GraphNodeInfo>) {
  return [...new Set([...portsFor(node, graph, info), ...graph.edges.filter(e => e.target === node.id).map(e => e.port)])]
}

/** Replace one operation in its existing slot, preserving every incident wire. */
export function replaceLayer(graph: GraphSpec, id: string, replacement: GraphNodeSpec, info: Record<string, GraphNodeInfo> = {}): GraphSpec {
  if (graph.locked) throw new Error('Clone to edit this model.')
  const previous = graph.nodes.find(n => n.id === id)
  if (!previous) throw new Error('The selected layer no longer exists.')
  if (graph.nodes.some(n => n.id === replacement.id && n.id !== id)) throw new Error('The replacement ID is already in use.')
  const oldPorts = replacementPorts(previous, graph, info), newPorts = portsFor(replacement, graph, {})
  if (oldPorts.length !== newPorts.length) throw new Error(`Choose a layer with ${oldPorts.length} input${oldPorts.length === 1 ? '' : 's'} to keep these connections.`)
  const mapping = new Map(oldPorts.filter(p => newPorts.includes(p)).map(p => [p, p]))
  const available = newPorts.filter(p => !mapping.has(p))
  oldPorts.filter(p => !mapping.has(p)).forEach((p, i) => mapping.set(p, available[i]))
  return { ...graph,
    nodes: graph.nodes.map(n => n.id === id ? { ...replacement, group: previous.group } : n),
    output: graph.output === id ? replacement.id : graph.output,
    edges: graph.edges.map(edge => ({ ...edge,
      source: edge.source === id ? replacement.id : edge.source,
      target: edge.target === id ? replacement.id : edge.target,
      port: edge.target === id ? mapping.get(edge.port)! : edge.port,
    })),
  }
}
