import { describe, expect, it } from 'vitest'
import { appendGraph, connectGraph, duplicateGraph, edgeId, emptyView, insertOnEdge, safeView } from './graphCanvas'
import type { GraphSpec } from './types'
const graph: GraphSpec = { schema_version: 2, revision: 'test', name: 'test', sources: {},
  nodes: [{ id: 'input', kind: 'source', params: {} }, { id: 'dense', kind: 'dense', params: { units: 4 }, group: 'core', module_ref: 'shared' }, { id: 'out', kind: 'activation', params: { kind: 'gelu' } }],
  edges: [{ source: 'input', target: 'dense', port: 'x' }, { source: 'dense', target: 'out', port: 'x' }], output: 'out', groups: [{ id: 'core', label: 'Model' }] }
describe('canonical canvas topology', () => {
  it('rejects cycles and duplicate drivers and supports reconnecting an edge', () => {
    expect(() => connectGraph(graph, { source: 'out', target: 'input', port: 'x' })).toThrow(/cycle/)
    expect(() => connectGraph(graph, { source: 'input', target: 'out', port: 'x' })).toThrow(/already/)
    const next = connectGraph(graph, { source: 'input', target: 'out', port: 'x' }, edgeId(graph.edges[1]))
    expect(next.edges[1].source).toBe('input')
    expect(graph.edges[1].source).toBe('dense')
  })
  it('duplicates independent weights and remaps internal connections', () => {
    const next = duplicateGraph(graph, ['dense', 'out'])
    expect(next.nodes).toHaveLength(5)
    const copy = next.nodes[3]
    expect(copy.module_ref).not.toBe('shared')
    expect(next.edges).toContainEqual({ source: copy.id, target: next.nodes[4].id, port: 'x' })
  })
  it('inserts another model as an independently wired subgraph', () => {
    const next = appendGraph(graph, graph, { input: { label: 'Input', category: 'placeholder', ports: [], settings: {}, shape: null } })
    expect(next.nodes).toHaveLength(6)
    expect(next.nodes[3].kind).toBe('activation')
    expect(next.nodes[3].params).toEqual({ kind: 'linear' })
    expect(next.nodes[4].module_ref).not.toBe(graph.nodes[1].module_ref)
    expect(next.output).toBe(graph.output)
    expect(next.groups).toHaveLength(2)
  })
})

describe('guided insertion and relocation', () => {
  const chain: GraphSpec = { ...graph, groups: [], nodes: ['a', 'b', 'c', 'd', 'out'].map(id => ({ id, kind: 'activation', params: {} })),
    edges: [{ source: 'a', target: 'b', port: 'x' }, { source: 'b', target: 'c', port: 'x' }, { source: 'c', target: 'd', port: 'x' }, { source: 'd', target: 'out', port: 'args/0' }] }
  it('splits a connection, preserves its downstream port and leaves the original graph intact', () => {
    const before = structuredClone(chain)
    const next = insertOnEdge(chain, { id: 'new', kind: 'dropout', params: { p: .1 } }, edgeId(chain.edges[3]))
    expect(next.edges).toContainEqual({ source: 'd', target: 'new', port: 'x' })
    expect(next.edges).toContainEqual({ source: 'new', target: 'out', port: 'args/0' })
    expect(next.edges).not.toContainEqual(chain.edges[3])
    expect(next.nodes).toHaveLength(6)
    expect(chain).toEqual(before)
  })
  it('moves an existing layer forward and heals its previous neighbours', () => {
    const next = insertOnEdge(chain, chain.nodes[1], edgeId(chain.edges[3]))
    expect(next.edges).toEqual(expect.arrayContaining([
      { source: 'a', target: 'c', port: 'x' }, { source: 'd', target: 'b', port: 'x' }, { source: 'b', target: 'out', port: 'args/0' },
    ]))
    expect(next.edges).toHaveLength(4)
    expect(next.nodes).toHaveLength(5)
  })
  it('retains automatic fitting when a HyperDense layer moves to a different connection', () => {
    const hyper = { ...chain.nodes[1], kind: 'hyper_dense', params: { shape_mode: 'preserve', algebra: 'quaternion', units: 8 } }
    const original = { ...chain, nodes: chain.nodes.map(n => n.id === hyper.id ? hyper : n) }
    const moved = insertOnEdge(original, hyper, edgeId(original.edges[3]))
    expect(moved.nodes.find(n => n.id === hyper.id)?.params).toEqual(hyper.params)
    expect(moved.edges).toContainEqual({ source: 'd', target: 'b', port: 'x' })
    expect(moved.edges).toContainEqual({ source: 'a', target: 'c', port: 'x' })
    expect(original.edges).toEqual(chain.edges)
  })
  it('moves a layer backward while retaining source-layer port names', () => {
    const info = { d: { label: 'Dense', category: 'call_module', ports: ['args/0'], shape: null, settings: {} } }
    const next = insertOnEdge(chain, { ...chain.nodes[3], kind: 'source' }, edgeId(chain.edges[0]), info)
    expect(next.edges).toEqual(expect.arrayContaining([
      { source: 'a', target: 'd', port: 'args/0' }, { source: 'd', target: 'b', port: 'x' }, { source: 'c', target: 'out', port: 'args/0' },
    ]))
  })
  it('slots a disconnected layer into an expanded group and keeps boundary insertions visible', () => {
    const node = { id: 'new', kind: 'activation', params: {} }
    const grouped = { ...chain, nodes: [...chain.nodes.map(n => ({ ...n, group: 'core' })), node], groups: graph.groups }
    const next = insertOnEdge(grouped, node, edgeId(chain.edges[1]))
    expect(next.nodes.find(n => n.id === 'new')?.group).toBe('core')
    expect(next.nodes).toHaveLength(6)
    expect(insertOnEdge(graph, node, edgeId(graph.edges[0])).nodes.find(n => n.id === 'new')?.group).toBeUndefined()
  })
  it('rejects branches, multi-input operations, endpoints, incident edges and locked presets', () => {
    const branching = { ...chain, edges: [...chain.edges, { source: 'b', target: 'd', port: 'other' }] }
    expect(() => insertOnEdge(branching, chain.nodes[1], edgeId(chain.edges[3]))).toThrow(/branch/)
    expect(() => insertOnEdge(chain, { id: 'new', kind: 'add', params: {} }, edgeId(chain.edges[0]))).toThrow(/single-input/)
    expect(() => insertOnEdge(chain, chain.nodes[4], edgeId(chain.edges[0]))).toThrow(/forecast output/)
    expect(() => insertOnEdge(chain, chain.nodes[1], edgeId(chain.edges[0]))).toThrow(/other layers/)
    expect(() => insertOnEdge({ ...chain, locked: true }, chain.nodes[1], edgeId(chain.edges[3]))).toThrow(/Clone/)
  })
  it('discards legacy pixel positions and preserves versioned camera state', () => {
    const positions = { a: { x: 40, y: 100 } }
    expect(safeView({ positions, collapsed: [] }, chain).positions).toEqual({})
    expect(safeView({ positions, collapsed: [], direction: 'LR' }, chain).positions).toEqual({})
    expect(safeView({ renderer: 'flow', version: 1, camera: { panX: 12, panY: 3, zoom: 2 }, collapsed: [] }, chain).camera).toEqual({ panX: 12, panY: 3, zoom: 2 })
  })
})
