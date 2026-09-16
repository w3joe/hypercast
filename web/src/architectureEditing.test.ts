import { describe, expect, it } from 'vitest'
import { isLayerOperation, replaceLayer, replacementPorts } from './architectureEditing'
import type { GraphSpec, GraphNodeInfo } from './types'

const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'Branched model', sources: {}, groups: [{ id: 'core', label: 'Core' }], output: 'end', nodes: [
  { id: 'input', kind: 'source', params: {} }, { id: 'a', kind: 'source', params: {}, group: 'core', module_ref: 'shared' },
  ...['b', 'skip', 'end'].map(id => ({ id, kind: 'dense', params: {} })),
], edges: [
  { source: 'input', target: 'a', port: 'args/0' }, { source: 'a', target: 'b', port: 'x' },
  { source: 'a', target: 'skip', port: 'left' }, { source: 'b', target: 'end', port: 'x' },
] }
const dropout = { id: 'new', kind: 'dropout', params: { p: .1 } }

describe('guided architecture replacement', () => {
  it('replaces in the original slot and group while preserving all outgoing branches and destination ports', () => {
    const before = JSON.stringify(graph)
    const next = replaceLayer(graph, 'a', dropout)
    expect(next.nodes[1]).toEqual({ ...dropout, group: 'core' })
    expect(next.nodes.map(n => n.id)).toEqual(['input', 'new', 'b', 'skip', 'end'])
    expect(next.edges).toEqual([
      { source: 'input', target: 'new', port: 'x' }, { source: 'new', target: 'b', port: 'x' },
      { source: 'new', target: 'skip', port: 'left' }, graph.edges[3],
    ])
    expect(JSON.stringify(graph)).toBe(before)
  })
  it('updates the forecast output when replacing its operation', () => {
    expect(replaceLayer(graph, 'end', dropout).output).toBe('new')
  })
  it('preserves missing inputs in invalid drafts instead of inventing wires', () => {
    const draft = { ...graph, edges: graph.edges.filter(e => e.target !== 'a') }
    const info = { a: { ports: ['args/0'] } as GraphNodeInfo }
    const next = replaceLayer(draft, 'a', dropout, info)
    expect(next.edges.some(e => e.target === 'new')).toBe(false)
    expect(next.edges.filter(e => e.source === 'new')).toHaveLength(2)
  })
  it('keeps named multi-input sources even when edges arrive in the opposite order', () => {
    const joined = { ...graph, nodes: graph.nodes.map(n => n.id === 'a' ? { ...n, kind: 'add' } : n), edges: [
      { source: 'input', target: 'a', port: 'b' }, { source: 'skip', target: 'a', port: 'a' },
    ] }
    expect(replaceLayer(joined, 'a', { ...dropout, kind: 'multiply' }).edges).toEqual([
      { source: 'input', target: 'new', port: 'b' }, { source: 'skip', target: 'new', port: 'a' },
    ])
    expect(() => replaceLayer(joined, 'a', dropout)).toThrow('2 inputs')
  })
  it('includes missing expected ports and stale connected ports when checking replacement compatibility', () => {
    expect(replacementPorts(graph.nodes[1], graph, { a: { ports: ['missing'] } as GraphNodeInfo })).toEqual(['missing', 'args/0'])
    expect(() => replaceLayer(graph, 'a', dropout, { a: { ports: ['missing'] } as GraphNodeInfo })).toThrow('2 inputs')
  })
  it('rejects locked graphs, stale selections and duplicate IDs', () => {
    expect(() => replaceLayer({ ...graph, locked: true }, 'a', dropout)).toThrow('Clone to edit')
    expect(() => replaceLayer(graph, 'missing', dropout)).toThrow('no longer exists')
    expect(() => replaceLayer(graph, 'a', { ...dropout, id: 'b' })).toThrow('already in use')
  })
  it('keeps custom drafts and actual modules in the layer list while leaving tensor operations accessible separately', () => {
    const node = { id: 'a', kind: 'source', params: {} }
    const info = { label: 'a', ports: [], settings: {}, category: 'call_function' } as GraphNodeInfo
    expect(isLayerOperation(node, info)).toBe(false)
    expect(isLayerOperation(node, { ...info, category: 'call_module' })).toBe(true)
    expect(isLayerOperation({ ...node, kind: 'dropout' })).toBe(true)
  })
})
